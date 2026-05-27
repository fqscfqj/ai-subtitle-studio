from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QEvent, QModelIndex, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent, QFont, QKeySequence, QShortcut, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .config import has_ffmpeg, load_settings, save_settings
from .constants import (
    DEFAULT_VAD_MAX_SEGMENT_SECONDS,
    DEFAULT_VAD_MIN_SILENCE_MS,
    DEFAULT_VAD_MIN_SPEECH_MS,
    DEFAULT_VAD_SPEECH_PAD_MS,
    DEFAULT_VAD_THRESHOLD,
    MEDIA_EXTENSIONS,
    STATUS_COLORS,
    STATUS_LABELS,
    SUBTITLE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from .media import discover_supported_files
from .models import AppSettings, TaskState
from .providers import transcription as transcription_provider
from .queue_manager import TaskQueueManager
from .utils import normalize_language_code, normalize_path_key, parse_context_bias


# ──────────────────────────────────────────────────────────────
#  自定义控件
# ──────────────────────────────────────────────────────────────

class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


class CollapsibleGroupBox(QWidget):
    """可折叠的分组容器，带 ▶/▼ 箭头动画。"""

    def __init__(self, title: str, parent: Optional[QWidget] = None, collapsed: bool = False) -> None:
        super().__init__(parent)
        self._collapsed = collapsed

        self._header = QPushButton(f"  ▼  {title}" if not collapsed else f"  ▶  {title}")
        self._header.setObjectName("collapsibleHeader")
        self._header.setCheckable(True)
        self._header.setChecked(not collapsed)
        self._header.setStyleSheet(
            "QPushButton#collapsibleHeader { text-align: left; font-weight: 700; font-size: 13px; "
            "padding: 8px 12px; background: #dce6f2; border: 1px solid #b8c7d9; border-radius: 8px; "
            "color: #1a2a3a; }"
            "QPushButton#collapsibleHeader:hover { background: #d0ddef; }"
        )
        self._header.clicked.connect(self._toggle)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._content.setVisible(not collapsed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._header)
        layout.addWidget(self._content)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def add_widget(self, widget: QWidget) -> None:
        self._content_layout.addWidget(widget)

    def _toggle(self) -> None:
        self._collapsed = not self._collapsed
        self._content.setVisible(not self._collapsed)
        text = self._header.text()
        if self._collapsed:
            self._header.setText(text.replace("▼", "▶"))
        else:
            self._header.setText(text.replace("▶", "▼"))


class DropFrame(QFrame):
    """拖拽区域，支持 hover 高亮。"""
    files_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("dropFrame")
        layout = QVBoxLayout(self)
        self._label = QLabel("＋  将视频/音频/字幕文件或文件夹拖拽到这里")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color: #4f83c2; font-size: 14px; font-weight: 600;")
        layout.addWidget(self._label)
        self._base_style = self.styleSheet()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_hover(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._set_hover(False)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._set_hover(False)
        paths: List[str] = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local:
                paths.append(local)
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()

    def _set_hover(self, hover: bool) -> None:
        if hover:
            self._label.setStyleSheet("color: #1e5ba8; font-size: 14px; font-weight: 700;")
            self.setStyleSheet(
                "#dropFrame { border: 2px solid #2e78c7; border-radius: 10px; "
                "background: #dce8f6; min-height: 84px; }"
            )
        else:
            self._label.setStyleSheet("color: #4f83c2; font-size: 14px; font-weight: 600;")
            self.setStyleSheet("")


class TaskTableWidget(QTableWidget):
    """支持右键菜单和拖拽排序的任务表格。"""
    reorder_requested = Signal()  # 拖拽排序完成信号

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, 6, parent)
        self.setHorizontalHeaderLabels(["来源文件", "状态", "进度", "耗时", "输出文件", "消息"])
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setDragDropMode(QTableWidget.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.verticalHeader().setDefaultSectionSize(34)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        # 由 MainWindow 注入菜单，这里仅转发
        menu = QMenu(self)
        # 获取父级 MainWindow
        mw = self.window()
        if hasattr(mw, "_build_context_menu"):
            mw._build_context_menu(menu)
        if menu.actions():
            menu.exec(event.globalPos())

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        super().dropEvent(event)
        self.reorder_requested.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Subtitle Studio 字幕工作台")
        self.resize(1280, 860)

        # ── 队列管理器 ──
        self.qm = TaskQueueManager(self)
        self.qm.task_progress.connect(self.on_task_progress)
        self.qm.task_finished.connect(self.on_task_finished)
        self.qm.batch_finished.connect(self.on_batch_finished)

        # ── 进度刷新定时器 ──
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(120)
        self._flush_timer.timeout.connect(self.qm.flush_pending_progress)

        # ── 耗时刷新定时器 ──
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed_times)

        # ── API Key 同步标志 ──
        self._syncing_mistral_api_key = False

        self.init_ui()
        self.apply_style()
        self.load_settings_into_ui()

    def init_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        tabs = QTabWidget()
        tabs.addTab(self._build_task_page(), "任务")
        tabs.addTab(self._build_settings_page(), "设置")
        root_layout.addWidget(tabs)

        self.setCentralWidget(root)

        # ── 键盘快捷键 ──
        QShortcut(QKeySequence("Delete"), self, self.on_remove_selected)
        QShortcut(QKeySequence("F5"), self, self.on_start)
        QShortcut(QKeySequence("Escape"), self, self.on_stop)
        QShortcut(QKeySequence("Ctrl+R"), self, self._retry_selected)

    def _build_task_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ── 拖拽区域 ──
        self.drop_frame = DropFrame()
        self.drop_frame.files_dropped.connect(self.on_drop_paths)
        layout.addWidget(self.drop_frame)

        # ── 按钮栏：左侧文件操作 + 右侧运行控制 ──
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(6)

        self.add_file_btn = QPushButton("添加文件")
        self.add_file_btn.setObjectName("secondaryBtn")
        self.add_file_btn.clicked.connect(self.on_add_file)
        self.add_folder_btn = QPushButton("添加文件夹")
        self.add_folder_btn.setObjectName("secondaryBtn")
        self.add_folder_btn.clicked.connect(self.on_add_folder)
        self.remove_btn = QPushButton("删除所选")
        self.remove_btn.setObjectName("dangerBtn")
        self.remove_btn.clicked.connect(self.on_remove_selected)
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.setObjectName("dangerBtn")
        self.clear_btn.clicked.connect(self.on_clear_all)

        btn_layout.addWidget(self.add_file_btn)
        btn_layout.addWidget(self.add_folder_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch(1)

        self.start_btn = QPushButton("▶ 开始")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setObjectName("warningBtn")
        self.pause_btn.clicked.connect(self._pause_selected)
        self.pause_btn.setEnabled(False)
        self.resume_btn = QPushButton("▶ 恢复")
        self.resume_btn.setObjectName("primaryBtn")
        self.resume_btn.clicked.connect(self._resume_selected)
        self.resume_btn.setEnabled(False)
        self.retry_btn = QPushButton("↻ 重试")
        self.retry_btn.setObjectName("secondaryBtn")
        self.retry_btn.clicked.connect(self._retry_selected)
        self.open_output_btn = QPushButton("打开输出目录")
        self.open_output_btn.setObjectName("secondaryBtn")
        self.open_output_btn.clicked.connect(self.on_open_output_dir)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.resume_btn)
        btn_layout.addWidget(self.retry_btn)
        btn_layout.addWidget(self.open_output_btn)
        layout.addWidget(btn_row)

        # ── 任务表格 ──
        self.task_table = TaskTableWidget()
        self.task_table.reorder_requested.connect(self._on_table_reorder)
        layout.addWidget(self.task_table)

        # ── 总进度条 + 摘要 ──
        self.total_progress = QProgressBar()
        self.total_progress.setRange(0, 100)
        self.total_progress.setValue(0)
        self.total_progress.setFixedHeight(22)
        self.total_progress.setFormat("%p%")
        self.summary_label = QLabel("暂无任务")
        self.summary_label.setStyleSheet("font-weight: 600; color: #374151;")
        layout.addWidget(self.total_progress)
        layout.addWidget(self.summary_label)

        # ── 日志 ──
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(120)
        layout.addWidget(self.log_text)
        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        outer_layout = QVBoxLayout(page)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 可折叠分组
        self._group_transcription = CollapsibleGroupBox("转写设置", collapsed=False)
        self._group_transcription.add_widget(self._build_transcription_group_inner())

        self._group_translation = CollapsibleGroupBox("翻译设置", collapsed=False)
        self._group_translation.add_widget(self._build_translation_group_inner())

        self._group_output = CollapsibleGroupBox("输出设置", collapsed=False)
        self._group_output.add_widget(self._build_output_group_inner())

        self._group_preprocess = CollapsibleGroupBox("预处理（VAD）", collapsed=True)
        self._group_preprocess.add_widget(self._build_preprocess_group_inner())

        for group in (self._group_transcription, self._group_translation, self._group_output, self._group_preprocess):
            group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
            layout.addWidget(group)

        layout.addStretch(1)

        # 保存按钮固定在底部
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 6, 0, 0)
        footer_layout.addStretch(1)
        self.save_settings_btn = QPushButton("保存设置")
        self.save_settings_btn.setObjectName("primaryBtn")
        self.save_settings_btn.clicked.connect(self.on_save_settings)
        footer_layout.addWidget(self.save_settings_btn)

        scroll.setWidget(content)
        outer_layout.addWidget(footer)
        return page

    def _build_transcription_group_inner(self) -> QGroupBox:
        group = QGroupBox("转写设置")
        layout = QGridLayout(group)

        self.transcription_provider_combo = NoWheelComboBox()
        self.transcription_provider_combo.addItem("Mistral", "mistral")
        self.transcription_provider_combo.addItem("Whisper(OpenAI 兼容)", "whisper_openai_compatible")
        self.transcription_provider_combo.addItem("Qwen3 ASR（DashScope）", "qwen3asr")
        self.transcription_provider_combo.currentIndexChanged.connect(self.on_transcription_provider_changed)

        self.mistral_api_key_input = QLineEdit(os.environ.get("MISTRAL_API_KEY", ""))
        self.mistral_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.mistral_api_key_input.textChanged.connect(self.on_transcription_mistral_key_changed)
        self.show_mistral_key_checkbox = QCheckBox("显示")
        self.show_mistral_key_checkbox.toggled.connect(
            lambda checked: self.mistral_api_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        self.mistral_key_row = QWidget()
        mistral_key_layout = QHBoxLayout(self.mistral_key_row)
        mistral_key_layout.setContentsMargins(0, 0, 0, 0)
        mistral_key_layout.addWidget(self.mistral_api_key_input)
        mistral_key_layout.addWidget(self.show_mistral_key_checkbox)

        self.mistral_model_combo = NoWheelComboBox()
        self.mistral_model_combo.setEditable(True)
        self.mistral_model_combo.addItems(["voxtral-mini-latest", "voxtral-small-latest"])

        self.whisper_base_url_input = QLineEdit("https://api.openai.com/v1")
        self.whisper_model_input = QLineEdit("whisper-1")
        self.whisper_api_key_input = QLineEdit(os.environ.get("OPENAI_API_KEY", ""))
        self.whisper_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_whisper_key_checkbox = QCheckBox("显示")
        self.show_whisper_key_checkbox.toggled.connect(
            lambda checked: self.whisper_api_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        self.whisper_key_row = QWidget()
        whisper_key_layout = QHBoxLayout(self.whisper_key_row)
        whisper_key_layout.setContentsMargins(0, 0, 0, 0)
        whisper_key_layout.addWidget(self.whisper_api_key_input)
        whisper_key_layout.addWidget(self.show_whisper_key_checkbox)

        self.qwen3asr_api_key_input = QLineEdit(os.environ.get("DASHSCOPE_API_KEY", ""))
        self.qwen3asr_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_qwen3asr_key_checkbox = QCheckBox("显示")
        self.show_qwen3asr_key_checkbox.toggled.connect(
            lambda checked: self.qwen3asr_api_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        self.qwen3asr_key_row = QWidget()
        qwen3asr_key_layout = QHBoxLayout(self.qwen3asr_key_row)
        qwen3asr_key_layout.setContentsMargins(0, 0, 0, 0)
        qwen3asr_key_layout.addWidget(self.qwen3asr_api_key_input)
        qwen3asr_key_layout.addWidget(self.show_qwen3asr_key_checkbox)

        self.qwen3asr_model_combo = NoWheelComboBox()
        self.qwen3asr_model_combo.setEditable(True)
        self.qwen3asr_model_combo.addItems([
            "qwen3-asr-flash",
            "qwen3-asr-flash-2026-02-10",
        ])

        self.language_mode_combo = NoWheelComboBox()
        self.language_mode_combo.addItems(["自动识别", "指定语言"])
        self.language_mode_combo.currentIndexChanged.connect(self.on_language_mode_changed)
        self.language_input = QLineEdit("zh")
        self.timestamp_combo = NoWheelComboBox()
        self.timestamp_combo.addItems(["none", "segment", "word"])
        self.timestamp_combo.setCurrentText("segment")
        self.diarize_checkbox = QCheckBox("启用说话人分离（仅 Mistral）")
        self.thread_spin = NoWheelSpinBox()
        self.thread_spin.setRange(1, 16)
        self.thread_spin.setValue(3)
        self.context_bias_input = QPlainTextEdit()
        self.context_bias_input.setPlaceholderText("术语提示/上下文偏置，使用逗号或换行分隔")
        self.context_bias_input.setFixedHeight(64)

        self.transcription_provider_label = QLabel("转写提供方")
        self.mistral_api_key_label = QLabel("Mistral API Key")
        self.mistral_model_label = QLabel("Mistral 模型")
        self.whisper_base_url_label = QLabel("Whisper Base URL")
        self.whisper_api_key_label = QLabel("Whisper API Key")
        self.whisper_model_label = QLabel("Whisper 模型")
        self.qwen3asr_api_key_label = QLabel("DashScope API Key")
        self.qwen3asr_model_label = QLabel("Qwen3 ASR 模型")
        self.language_mode_label = QLabel("语言模式")
        self.language_label = QLabel("指定语言")
        self.timestamp_label = QLabel("时间戳粒度")
        self.thread_label = QLabel("任务线程数")
        self.context_bias_label = QLabel("术语提示")

        layout.addWidget(self.transcription_provider_label, 0, 0)
        layout.addWidget(self.transcription_provider_combo, 0, 1)
        layout.addWidget(self.mistral_api_key_label, 1, 0)
        layout.addWidget(self.mistral_key_row, 1, 1)
        layout.addWidget(self.mistral_model_label, 2, 0)
        layout.addWidget(self.mistral_model_combo, 2, 1)
        layout.addWidget(self.whisper_base_url_label, 3, 0)
        layout.addWidget(self.whisper_base_url_input, 3, 1)
        layout.addWidget(self.whisper_api_key_label, 4, 0)
        layout.addWidget(self.whisper_key_row, 4, 1)
        layout.addWidget(self.whisper_model_label, 5, 0)
        layout.addWidget(self.whisper_model_input, 5, 1)
        layout.addWidget(self.qwen3asr_api_key_label, 6, 0)
        layout.addWidget(self.qwen3asr_key_row, 6, 1)
        layout.addWidget(self.qwen3asr_model_label, 7, 0)
        layout.addWidget(self.qwen3asr_model_combo, 7, 1)
        layout.addWidget(self.language_mode_label, 8, 0)
        layout.addWidget(self.language_mode_combo, 8, 1)
        layout.addWidget(self.language_label, 9, 0)
        layout.addWidget(self.language_input, 9, 1)
        layout.addWidget(self.timestamp_label, 10, 0)
        layout.addWidget(self.timestamp_combo, 10, 1)
        layout.addWidget(self.thread_label, 11, 0)
        layout.addWidget(self.thread_spin, 11, 1)
        layout.addWidget(self.context_bias_label, 12, 0)
        layout.addWidget(self.context_bias_input, 12, 1)
        layout.addWidget(self.diarize_checkbox, 13, 0, 1, 2)

        self.transcription_mistral_key_widgets = [
            self.mistral_api_key_label,
            self.mistral_key_row,
        ]
        self.transcription_mistral_only_widgets = [
            self.mistral_model_label,
            self.mistral_model_combo,
            self.diarize_checkbox,
        ]
        self.transcription_whisper_widgets = [
            self.whisper_base_url_label,
            self.whisper_base_url_input,
            self.whisper_api_key_label,
            self.whisper_key_row,
            self.whisper_model_label,
            self.whisper_model_input,
        ]
        self.transcription_qwen3asr_widgets = [
            self.qwen3asr_api_key_label,
            self.qwen3asr_key_row,
            self.qwen3asr_model_label,
            self.qwen3asr_model_combo,
        ]
        return group

    def _build_translation_group_inner(self) -> QGroupBox:
        group = QGroupBox("翻译设置")
        layout = QGridLayout(group)

        self.translation_mode_combo = NoWheelComboBox()
        self.translation_mode_combo.addItem("不翻译", "none")
        self.translation_mode_combo.addItem("Mistral API 翻译", "mistral")
        self.translation_mode_combo.addItem("OpenAI 兼容 API 翻译", "openai")
        self.translation_mode_combo.currentIndexChanged.connect(self.on_translation_mode_changed)

        self.translation_target_input = QLineEdit("zh")
        self.translation_model_input = QLineEdit("mistral-small-latest")
        self.translation_bilingual_checkbox = QCheckBox("SRT 输出双语（原文 + 译文）")
        self.translation_bilingual_checkbox.setChecked(True)
        self.translation_keep_original_checkbox = QCheckBox("翻译后额外输出原文字幕（xxx.orig.srt）")
        self.allow_subtitle_import_checkbox = QCheckBox("允许导入字幕文件并翻译")
        self.allow_subtitle_import_checkbox.setChecked(True)
        self.subtitle_translation_thread_spin = NoWheelSpinBox()
        self.subtitle_translation_thread_spin.setRange(1, 16)
        self.subtitle_translation_thread_spin.setValue(3)

        self.translation_thinking_checkbox = QCheckBox("启用思考模式（Reasoning）")
        self.translation_thinking_checkbox.setToolTip(
            "开启后将使用模型的深度思考能力，可能提升翻译质量但会增加延迟和消耗。\n"
            "支持 DeepSeek、OpenAI 等兼容 reasoning_effort 的模型。\n"
            "思考模式下 temperature 参数将被忽略。"
        )
        self.translation_reasoning_effort_combo = NoWheelComboBox()
        self.translation_reasoning_effort_combo.addItem("low", "low")
        self.translation_reasoning_effort_combo.addItem("medium", "medium")
        self.translation_reasoning_effort_combo.addItem("high", "high")
        self.translation_reasoning_effort_combo.addItem("max", "max")
        self.translation_reasoning_effort_combo.setCurrentText("high")
        self.translation_reasoning_effort_combo.setToolTip(
            "思考强度：low / medium / high / max\n"
            "越高质量越好但延迟和消耗越大。\n"
            "DeepSeek 模型下 low/medium 会映射为 high。"
        )
        self.translation_reasoning_effort_label = QLabel("思考强度")

        self.translation_mistral_api_key_input = QLineEdit(os.environ.get("MISTRAL_API_KEY", ""))
        self.translation_mistral_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.translation_mistral_api_key_input.textChanged.connect(self.on_translation_mistral_key_changed)
        self.show_translation_mistral_key_checkbox = QCheckBox("显示")
        self.show_translation_mistral_key_checkbox.toggled.connect(
            lambda checked: self.translation_mistral_api_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        self.translation_mistral_key_row = QWidget()
        translation_mistral_key_layout = QHBoxLayout(self.translation_mistral_key_row)
        translation_mistral_key_layout.setContentsMargins(0, 0, 0, 0)
        translation_mistral_key_layout.addWidget(self.translation_mistral_api_key_input)
        translation_mistral_key_layout.addWidget(self.show_translation_mistral_key_checkbox)

        self.translation_openai_base_input = QLineEdit("https://api.openai.com/v1")
        self.translation_openai_key_input = QLineEdit(os.environ.get("OPENAI_API_KEY", ""))
        self.translation_openai_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_translation_openai_key_checkbox = QCheckBox("显示")
        self.show_translation_openai_key_checkbox.toggled.connect(
            lambda checked: self.translation_openai_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        self.translation_openai_key_row = QWidget()
        openai_key_layout = QHBoxLayout(self.translation_openai_key_row)
        openai_key_layout.setContentsMargins(0, 0, 0, 0)
        openai_key_layout.addWidget(self.translation_openai_key_input)
        openai_key_layout.addWidget(self.show_translation_openai_key_checkbox)

        self.translation_mode_label = QLabel("翻译模式")
        self.translation_target_label = QLabel("目标语言")
        self.translation_model_label = QLabel("翻译模型")
        self.subtitle_translation_thread_label = QLabel("字幕翻译线程数")
        self.translation_mistral_api_key_label = QLabel("Mistral API Key")
        self.translation_openai_base_label = QLabel("OpenAI 兼容 Base URL")
        self.translation_openai_key_label = QLabel("OpenAI 兼容 API Key")

        layout.addWidget(self.translation_mode_label, 0, 0)
        layout.addWidget(self.translation_mode_combo, 0, 1)
        layout.addWidget(self.translation_target_label, 1, 0)
        layout.addWidget(self.translation_target_input, 1, 1)
        layout.addWidget(self.translation_model_label, 2, 0)
        layout.addWidget(self.translation_model_input, 2, 1)
        layout.addWidget(self.translation_bilingual_checkbox, 3, 0, 1, 2)
        layout.addWidget(self.translation_keep_original_checkbox, 4, 0, 1, 2)
        layout.addWidget(self.allow_subtitle_import_checkbox, 5, 0, 1, 2)
        layout.addWidget(self.subtitle_translation_thread_label, 6, 0)
        layout.addWidget(self.subtitle_translation_thread_spin, 6, 1)
        layout.addWidget(self.translation_thinking_checkbox, 7, 0, 1, 2)
        layout.addWidget(self.translation_reasoning_effort_label, 8, 0)
        layout.addWidget(self.translation_reasoning_effort_combo, 8, 1)
        layout.addWidget(self.translation_mistral_api_key_label, 9, 0)
        layout.addWidget(self.translation_mistral_key_row, 9, 1)
        layout.addWidget(self.translation_openai_base_label, 10, 0)
        layout.addWidget(self.translation_openai_base_input, 10, 1)
        layout.addWidget(self.translation_openai_key_label, 11, 0)
        layout.addWidget(self.translation_openai_key_row, 11, 1)

        self.translation_common_widgets = [
            self.translation_target_label,
            self.translation_target_input,
            self.translation_model_label,
            self.translation_model_input,
            self.translation_bilingual_checkbox,
            self.translation_keep_original_checkbox,
            self.allow_subtitle_import_checkbox,
            self.subtitle_translation_thread_label,
            self.subtitle_translation_thread_spin,
            self.translation_thinking_checkbox,
            self.translation_reasoning_effort_label,
            self.translation_reasoning_effort_combo,
        ]
        self.translation_mistral_widgets = [
            self.translation_mistral_api_key_label,
            self.translation_mistral_key_row,
        ]
        self.translation_openai_widgets = [
            self.translation_openai_base_label,
            self.translation_openai_base_input,
            self.translation_openai_key_label,
            self.translation_openai_key_row,
        ]
        return group

    def _build_output_group_inner(self) -> QGroupBox:
        group = QGroupBox("输出设置")
        layout = QGridLayout(group)

        self.output_mode_combo = NoWheelComboBox()
        self.output_mode_combo.addItem("输出到原文件目录", "source")
        self.output_mode_combo.addItem("输出到指定目录", "custom")
        self.output_mode_combo.currentIndexChanged.connect(self.on_output_mode_changed)
        self.output_dir_input = QLineEdit(str(Path.cwd() / "subtitles"))
        self.output_btn = QPushButton("浏览")
        self.output_btn.clicked.connect(self.on_choose_output_dir)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_dir_input)
        output_layout.addWidget(self.output_btn)

        self.save_srt_checkbox = QCheckBox("保存 .srt")
        self.save_srt_checkbox.setChecked(True)
        self.save_lrc_checkbox = QCheckBox("纯音频保存 .lrc")
        self.save_lrc_checkbox.setChecked(True)
        self.save_txt_checkbox = QCheckBox("保存 .txt")
        self.save_txt_checkbox.setChecked(True)
        self.save_json_checkbox = QCheckBox("保存 .json")
        format_row = QWidget()
        format_layout = QHBoxLayout(format_row)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.addWidget(self.save_srt_checkbox)
        format_layout.addWidget(self.save_lrc_checkbox)
        format_layout.addWidget(self.save_txt_checkbox)
        format_layout.addWidget(self.save_json_checkbox)
        format_layout.addStretch(1)

        layout.addWidget(QLabel("输出目录模式"), 0, 0)
        layout.addWidget(self.output_mode_combo, 0, 1)
        layout.addWidget(QLabel("指定输出目录"), 1, 0)
        layout.addWidget(output_row, 1, 1)
        layout.addWidget(format_row, 2, 0, 1, 2)
        return group

    def _build_preprocess_group_inner(self) -> QGroupBox:
        group = QGroupBox("预处理")
        layout = QGridLayout(group)

        self.enable_vad_checkbox = QCheckBox("启用 Silero VAD 预切分")
        self.enable_vad_checkbox.setChecked(False)
        self.enable_vad_checkbox.toggled.connect(self.on_vad_enabled_changed)

        self.vad_min_speech_spin = NoWheelSpinBox()
        self.vad_min_speech_spin.setRange(1, 60_000)
        self.vad_min_speech_spin.setSingleStep(50)
        self.vad_min_speech_spin.setSuffix(" ms")
        self.vad_min_speech_spin.setValue(DEFAULT_VAD_MIN_SPEECH_MS)

        self.vad_min_silence_spin = NoWheelSpinBox()
        self.vad_min_silence_spin.setRange(1, 60_000)
        self.vad_min_silence_spin.setSingleStep(50)
        self.vad_min_silence_spin.setSuffix(" ms")
        self.vad_min_silence_spin.setValue(DEFAULT_VAD_MIN_SILENCE_MS)

        self.vad_speech_pad_spin = NoWheelSpinBox()
        self.vad_speech_pad_spin.setRange(0, 60_000)
        self.vad_speech_pad_spin.setSingleStep(50)
        self.vad_speech_pad_spin.setSuffix(" ms")
        self.vad_speech_pad_spin.setValue(DEFAULT_VAD_SPEECH_PAD_MS)

        self.vad_max_segment_spin = NoWheelSpinBox()
        self.vad_max_segment_spin.setRange(1, 24 * 3600)
        self.vad_max_segment_spin.setSingleStep(30)
        self.vad_max_segment_spin.setSuffix(" s")
        self.vad_max_segment_spin.setValue(DEFAULT_VAD_MAX_SEGMENT_SECONDS)

        self.vad_threshold_spin = NoWheelDoubleSpinBox()
        self.vad_threshold_spin.setRange(0.0, 1.0)
        self.vad_threshold_spin.setDecimals(2)
        self.vad_threshold_spin.setSingleStep(0.05)
        self.vad_threshold_spin.setValue(DEFAULT_VAD_THRESHOLD)

        self.vad_controls = [
            self.vad_min_speech_spin,
            self.vad_min_silence_spin,
            self.vad_speech_pad_spin,
            self.vad_max_segment_spin,
            self.vad_threshold_spin,
        ]

        self.ffmpeg_hint_label = QLabel("视频任务与 VAD 预切分会自动使用软件内置 ffmpeg，无需单独设置。")
        self.ffmpeg_hint_label.setWordWrap(True)

        layout.addWidget(self.ffmpeg_hint_label, 0, 0, 1, 2)
        layout.addWidget(self.enable_vad_checkbox, 1, 0, 1, 2)
        layout.addWidget(QLabel("最短语音"), 2, 0)
        layout.addWidget(self.vad_min_speech_spin, 2, 1)
        layout.addWidget(QLabel("最短静音"), 3, 0)
        layout.addWidget(self.vad_min_silence_spin, 3, 1)
        layout.addWidget(QLabel("语音补边"), 4, 0)
        layout.addWidget(self.vad_speech_pad_spin, 4, 1)
        layout.addWidget(QLabel("单段最长时长"), 5, 0)
        layout.addWidget(self.vad_max_segment_spin, 5, 1)
        layout.addWidget(QLabel("检测阈值"), 6, 0)
        layout.addWidget(self.vad_threshold_spin, 6, 1)
        return group

    def _set_widgets_visible(self, widgets: List[QWidget], visible: bool) -> None:
        for widget in widgets:
            widget.setVisible(visible)

    def _sync_mistral_api_key_inputs(self, text: str, source: QLineEdit) -> None:
        if self._syncing_mistral_api_key:
            return
        self._syncing_mistral_api_key = True
        try:
            targets = [self.mistral_api_key_input, self.translation_mistral_api_key_input]
            for target in targets:
                if target is source or target.text() == text:
                    continue
                target.setText(text)
        finally:
            self._syncing_mistral_api_key = False

    def on_transcription_mistral_key_changed(self, text: str) -> None:
        self._sync_mistral_api_key_inputs(text, self.mistral_api_key_input)

    def on_translation_mistral_key_changed(self, text: str) -> None:
        self._sync_mistral_api_key_inputs(text, self.translation_mistral_api_key_input)

    def refresh_settings_visibility(self) -> None:
        self.on_language_mode_changed()
        self.on_translation_mode_changed()
        self.on_output_mode_changed()
        self.on_vad_enabled_changed()

    def apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #eef3f8; color: #1c2b39; }
            QTabWidget::pane {
                border: 1px solid #b8c7d9;
                border-radius: 8px;
                background: #eef3f8;
                top: -1px;
            }
            QTabBar::tab {
                background: #d9e6f3;
                color: #1c2b39;
                border: 1px solid #b8c7d9;
                border-bottom: none;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                padding: 8px 20px;
                margin-right: 3px;
                font-weight: 600;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: #f9fbfd;
                color: #1e2d3d;
            }
            QTabBar::tab:!selected:hover { background: #e6f0fa; }
            QScrollArea, QScrollArea > QWidget > QWidget { background: #eef3f8; }
            QGroupBox {
                border: 1px solid #c5d2df;
                border-radius: 10px;
                margin-top: 10px;
                padding: 14px 10px 10px 10px;
                font-weight: 600;
                background: #f9fbfd;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #1e2d3d;
            }
            #dropFrame {
                border: 2px dashed #4f83c2;
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f4f8fc, stop:1 #e1ecf8);
                min-height: 72px;
            }
            QLabel { color: #1c2b39; }
            QCheckBox { color: #1c2b39; background: transparent; spacing: 8px; padding: 3px 2px; }
            QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #7f95ad; border-radius: 4px; background: white; }
            QCheckBox::indicator:hover { border: 1px solid #2e78c7; }
            QCheckBox::indicator:checked { background: #2e78c7; border: 1px solid #2e78c7; }
            QCheckBox::indicator:checked:hover { background: #266ab1; border: 1px solid #266ab1; }
            QCheckBox::indicator:disabled { background: #edf2f7; border: 1px solid #c5d2df; }
            QCheckBox::indicator:checked:disabled { background: #9db5cc; border: 1px solid #9db5cc; }

            QPushButton {
                background: #2e78c7; color: white; border: none; border-radius: 7px;
                padding: 7px 14px; font-weight: 600;
            }
            QPushButton:hover { background: #266ab1; }
            QPushButton:disabled { background: #9db5cc; color: #ebf2f9; }
            QPushButton#primaryBtn { background: #2563eb; padding: 8px 18px; font-size: 13px; }
            QPushButton#primaryBtn:hover { background: #1d4ed8; }
            QPushButton#dangerBtn { background: #dc2626; }
            QPushButton#dangerBtn:hover { background: #b91c1c; }
            QPushButton#dangerBtn:disabled { background: #f3a0a0; color: #fef2f2; }
            QPushButton#warningBtn { background: #d97706; }
            QPushButton#warningBtn:hover { background: #b45309; }
            QPushButton#secondaryBtn { background: #64748b; }
            QPushButton#secondaryBtn:hover { background: #475569; }

            QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #b4c4d4; border-radius: 6px; padding: 5px; background: white; color: #1c2b39;
            }
            QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
                background: #edf2f7; color: #68798a; border: 1px solid #c5d2df;
            }
            QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                border-left: 1px solid #b4c4d4; background: #eef3f8; width: 22px;
            }
            QComboBox QAbstractItemView {
                background: white; color: #1c2b39; selection-background-color: #d9e6f3;
                selection-color: #1c2b39; border: 1px solid #b4c4d4;
            }
            QTableWidget {
                border: 1px solid #b8c7d9; border-radius: 8px;
                background: white; alternate-background-color: #f4f7fb;
                gridline-color: #d9e2ec;
            }
            QTableWidget::item { padding: 4px 6px; }
            QHeaderView::section {
                background: #d0ddf0; padding: 7px 6px; border: none;
                border-right: 1px solid #bfd0e1; color: #1a2a3a; font-weight: 700;
            }
            QProgressBar {
                border: 1px solid #9eb4c8; border-radius: 6px;
                text-align: center; background: #f4f8fc; font-weight: 600; font-size: 11px;
            }
            QProgressBar::chunk { background: #3b82f6; border-radius: 5px; }
            QMenu {
                background: white; border: 1px solid #b8c7d9; border-radius: 6px;
                padding: 4px 0px; color: #1c2b39;
            }
            QMenu::item { padding: 6px 24px 6px 12px; }
            QMenu::item:selected { background: #d9e6f3; color: #1a2a3a; }
            QMenu::item:disabled { color: #9ca3af; }
            """
        )
        self.setFont(QFont("Segoe UI", 10))

    def log(self, message: str) -> None:
        self.log_text.appendPlainText(message)

    def load_settings_into_ui(self) -> None:
        settings = load_settings()
        self.apply_settings_to_ui(settings)
        self.log("已加载本地设置")
        self.refresh_settings_visibility()

    def apply_settings_to_ui(self, settings: AppSettings) -> None:
        provider_index = {"mistral": 0, "whisper_openai_compatible": 1, "qwen3asr": 2}.get(
            settings.transcription.provider, 0
        )
        self.transcription_provider_combo.setCurrentIndex(provider_index)
        self.mistral_api_key_input.setText(settings.transcription.mistral.api_key)
        self.translation_mistral_api_key_input.setText(settings.transcription.mistral.api_key)
        self.mistral_model_combo.setCurrentText(settings.transcription.mistral.model)
        self.whisper_base_url_input.setText(settings.transcription.whisper.base_url)
        self.whisper_api_key_input.setText(settings.transcription.whisper.api_key)
        self.whisper_model_input.setText(settings.transcription.whisper.model)
        self.qwen3asr_api_key_input.setText(settings.transcription.qwen3asr.api_key)
        self.qwen3asr_model_combo.setCurrentText(settings.transcription.qwen3asr.model)
        self.language_mode_combo.setCurrentIndex(1 if settings.transcription.language_mode == "manual" else 0)
        self.language_input.setText(settings.transcription.language)
        self.timestamp_combo.setCurrentText(settings.transcription.timestamp_granularity)
        self.diarize_checkbox.setChecked(settings.transcription.diarize)
        self.thread_spin.setValue(settings.transcription.thread_count)
        self.context_bias_input.setPlainText(settings.transcription.context_bias)

        self.translation_mode_combo.setCurrentIndex({"none": 0, "mistral": 1, "openai": 2}.get(settings.translation.mode, 0))
        self.translation_target_input.setText(settings.translation.target_language)
        self.translation_model_input.setText(settings.translation.model)
        self.translation_bilingual_checkbox.setChecked(settings.translation.bilingual_srt)
        self.translation_keep_original_checkbox.setChecked(settings.translation.keep_original_srt)
        self.allow_subtitle_import_checkbox.setChecked(settings.translation.allow_subtitle_import)
        self.subtitle_translation_thread_spin.setValue(settings.translation.subtitle_translation_thread_count)
        self.translation_openai_base_input.setText(settings.translation.openai_base_url)
        self.translation_openai_key_input.setText(settings.translation.openai_api_key)
        self.translation_thinking_checkbox.setChecked(settings.translation.thinking_enabled)
        self.translation_reasoning_effort_combo.setCurrentText(settings.translation.reasoning_effort)

        self.output_mode_combo.setCurrentIndex(1 if settings.output.mode == "custom" else 0)
        self.output_dir_input.setText(str(settings.output.output_dir))
        self.save_srt_checkbox.setChecked(settings.output.save_srt)
        self.save_lrc_checkbox.setChecked(settings.output.save_lrc)
        self.save_txt_checkbox.setChecked(settings.output.save_txt)
        self.save_json_checkbox.setChecked(settings.output.save_json)
        self.enable_vad_checkbox.setChecked(settings.vad.enabled)
        self.vad_min_speech_spin.setValue(settings.vad.min_speech_ms)
        self.vad_min_silence_spin.setValue(settings.vad.min_silence_ms)
        self.vad_speech_pad_spin.setValue(settings.vad.speech_pad_ms)
        self.vad_max_segment_spin.setValue(settings.vad.max_segment_seconds)
        self.vad_threshold_spin.setValue(settings.vad.threshold)

    def collect_settings_from_ui(self) -> AppSettings:
        settings = AppSettings()
        settings.transcription.provider = self.transcription_provider_combo.currentData()
        settings.transcription.mistral.api_key = self.mistral_api_key_input.text().strip()
        settings.transcription.mistral.model = self.mistral_model_combo.currentText().strip() or "voxtral-mini-latest"
        settings.transcription.whisper.base_url = self.whisper_base_url_input.text().strip() or "https://api.openai.com/v1"
        settings.transcription.whisper.api_key = self.whisper_api_key_input.text().strip()
        settings.transcription.whisper.model = self.whisper_model_input.text().strip() or "whisper-1"
        settings.transcription.qwen3asr.api_key = self.qwen3asr_api_key_input.text().strip()
        settings.transcription.qwen3asr.model = self.qwen3asr_model_combo.currentText().strip() or "qwen3-asr-flash"
        settings.transcription.language_mode = "manual" if self.language_mode_combo.currentIndex() == 1 else "auto"
        settings.transcription.language = normalize_language_code(self.language_input.text().strip())
        settings.transcription.timestamp_granularity = self.timestamp_combo.currentText().strip() or "none"
        settings.transcription.diarize = self.diarize_checkbox.isChecked()
        settings.transcription.thread_count = self.thread_spin.value()
        settings.transcription.context_bias = parse_context_bias(self.context_bias_input.toPlainText())

        settings.translation.mode = self.translation_mode_combo.currentData()
        settings.translation.target_language = normalize_language_code(self.translation_target_input.text().strip())
        settings.translation.model = self.translation_model_input.text().strip()
        settings.translation.bilingual_srt = self.translation_bilingual_checkbox.isChecked()
        settings.translation.keep_original_srt = self.translation_keep_original_checkbox.isChecked()
        settings.translation.allow_subtitle_import = self.allow_subtitle_import_checkbox.isChecked()
        settings.translation.subtitle_translation_thread_count = self.subtitle_translation_thread_spin.value()
        settings.translation.openai_base_url = self.translation_openai_base_input.text().strip() or "https://api.openai.com/v1"
        settings.translation.openai_api_key = self.translation_openai_key_input.text().strip()
        settings.translation.thinking_enabled = self.translation_thinking_checkbox.isChecked()
        settings.translation.reasoning_effort = self.translation_reasoning_effort_combo.currentData() or "high"

        settings.output.mode = self.output_mode_combo.currentData()
        settings.output.output_dir = Path(self.output_dir_input.text().strip() or str(Path.cwd() / "subtitles"))
        settings.output.save_srt = self.save_srt_checkbox.isChecked()
        settings.output.save_lrc = self.save_lrc_checkbox.isChecked()
        settings.output.save_txt = self.save_txt_checkbox.isChecked()
        settings.output.save_json = self.save_json_checkbox.isChecked()

        settings.vad.enabled = self.enable_vad_checkbox.isChecked()
        settings.vad.min_speech_ms = self.vad_min_speech_spin.value()
        settings.vad.min_silence_ms = self.vad_min_silence_spin.value()
        settings.vad.speech_pad_ms = self.vad_speech_pad_spin.value()
        settings.vad.max_segment_seconds = self.vad_max_segment_spin.value()
        settings.vad.threshold = self.vad_threshold_spin.value()
        return settings

    def on_save_settings(self) -> None:
        try:
            settings = self.collect_settings_from_ui()
            path = save_settings(settings)
            self.log(f"设置已保存到：{path}")
            QMessageBox.information(self, "保存成功", f"设置已保存到：\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", f"无法保存设置：{exc}")

    def on_transcription_provider_changed(self) -> None:
        provider = self.transcription_provider_combo.currentData()
        translation_mode = self.translation_mode_combo.currentData()
        use_mistral = provider == "mistral"
        use_whisper = provider == "whisper_openai_compatible"
        use_qwen3asr = provider == "qwen3asr"
        show_mistral_key = use_mistral or translation_mode == "mistral"

        self._set_widgets_visible(self.transcription_mistral_key_widgets, show_mistral_key)
        self._set_widgets_visible(self.transcription_mistral_only_widgets, use_mistral)
        self._set_widgets_visible(self.transcription_whisper_widgets, use_whisper)
        self._set_widgets_visible(self.transcription_qwen3asr_widgets, use_qwen3asr)

        self.mistral_api_key_input.setEnabled(show_mistral_key)
        self.show_mistral_key_checkbox.setEnabled(show_mistral_key)
        self.mistral_model_combo.setEnabled(use_mistral)
        self.whisper_base_url_input.setEnabled(use_whisper)
        self.whisper_api_key_input.setEnabled(use_whisper)
        self.show_whisper_key_checkbox.setEnabled(use_whisper)
        self.whisper_model_input.setEnabled(use_whisper)
        self.qwen3asr_api_key_input.setEnabled(use_qwen3asr)
        self.show_qwen3asr_key_checkbox.setEnabled(use_qwen3asr)
        self.qwen3asr_model_combo.setEnabled(use_qwen3asr)
        self.diarize_checkbox.setEnabled(use_mistral)
        if not use_mistral:
            self.diarize_checkbox.setChecked(False)

    def on_translation_mode_changed(self) -> None:
        mode = self.translation_mode_combo.currentData()
        enable_translation = mode != "none"
        use_mistral = mode == "mistral"
        use_openai = mode == "openai"
        current_model = self.translation_model_input.text().strip()

        self._set_widgets_visible(self.translation_common_widgets, enable_translation)
        self._set_widgets_visible(self.translation_mistral_widgets, use_mistral)
        self._set_widgets_visible(self.translation_openai_widgets, use_openai)

        self.translation_target_input.setEnabled(enable_translation)
        self.translation_model_input.setEnabled(enable_translation)
        self.translation_bilingual_checkbox.setEnabled(enable_translation)
        self.translation_keep_original_checkbox.setEnabled(enable_translation)
        self.allow_subtitle_import_checkbox.setEnabled(enable_translation)
        self.subtitle_translation_thread_spin.setEnabled(enable_translation)
        self.translation_thinking_checkbox.setEnabled(enable_translation)
        self.translation_reasoning_effort_combo.setEnabled(enable_translation)
        self.translation_reasoning_effort_label.setEnabled(enable_translation)
        self.translation_mistral_api_key_input.setEnabled(use_mistral)
        self.show_translation_mistral_key_checkbox.setEnabled(use_mistral)
        self.translation_openai_base_input.setEnabled(use_openai)
        self.translation_openai_key_input.setEnabled(use_openai)
        self.show_translation_openai_key_checkbox.setEnabled(use_openai)

        if mode == "mistral" and current_model in {"", "gpt-4o-mini"}:
            self.translation_model_input.setText("mistral-small-latest")
        if mode == "openai" and current_model in {"", "mistral-small-latest"}:
            self.translation_model_input.setText("gpt-4o-mini")
        self.on_transcription_provider_changed()

    def on_language_mode_changed(self) -> None:
        manual = self.language_mode_combo.currentIndex() == 1
        self.language_input.setEnabled(manual)
        if manual:
            self.language_input.setPlaceholderText("语言代码，例如 zh / en")
        else:
            self.language_input.setPlaceholderText("自动识别时无需填写")

    def on_output_mode_changed(self) -> None:
        custom = self.output_mode_combo.currentData() == "custom"
        self.output_dir_input.setEnabled(custom)
        self.output_btn.setEnabled(custom)

    def on_vad_enabled_changed(self) -> None:
        enabled = self.enable_vad_checkbox.isChecked()
        for widget in self.vad_controls:
            widget.setEnabled(enabled)

    def on_choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir_input.text())
        if directory:
            self.output_dir_input.setText(directory)

    def on_add_file(self) -> None:
        filters = (
            "媒体/字幕文件 (*.mp4 *.mov *.mkv *.avi *.wmv *.webm *.m4v *.flv *.ts "
            "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma *.srt *.vtt *.txt)"
        )
        files, _ = QFileDialog.getOpenFileNames(self, "选择媒体或字幕文件", str(Path.cwd()), filters)
        if files:
            self.add_paths(files)

    def on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", str(Path.cwd()))
        if folder:
            self.add_paths([folder])

    def on_drop_paths(self, paths: List[str]) -> None:
        self.add_paths(paths)

    def add_paths(self, raw_paths: List[str]) -> None:
        allow_subtitle_import = self.allow_subtitle_import_checkbox.isChecked()
        discovered: List[Path] = []
        skipped_subtitle = 0
        for raw in raw_paths:
            path = Path(raw)
            if path.is_dir():
                for item in discover_supported_files(path):
                    if item.suffix.lower() in SUBTITLE_EXTENSIONS and not allow_subtitle_import:
                        skipped_subtitle += 1
                        continue
                    discovered.append(item)
            elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                if path.suffix.lower() in SUBTITLE_EXTENSIONS and not allow_subtitle_import:
                    skipped_subtitle += 1
                    continue
                discovered.append(path)

        if not discovered:
            if skipped_subtitle > 0 and not allow_subtitle_import:
                self.log("字幕导入开关已关闭，已忽略字幕文件")
            else:
                self.log("未找到支持的媒体或字幕文件")
            return

        added = 0
        for path in discovered:
            row = self.task_table.rowCount()
            task_id = self.qm.add_task(path, row)
            if task_id is None:
                continue
            self.task_table.insertRow(row)
            self.task_table.setItem(row, 0, QTableWidgetItem(str(path)))
            status_item = QTableWidgetItem("排队中")
            status_item.setForeground(QColor(STATUS_COLORS.get("Queued", "#6b7280")))
            self.task_table.setItem(row, 1, status_item)
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            self.task_table.setCellWidget(row, 2, progress_bar)
            self.task_table.setItem(row, 3, QTableWidgetItem("-"))
            self.task_table.setItem(row, 4, QTableWidgetItem("-"))
            self.task_table.setItem(row, 5, QTableWidgetItem("就绪"))
            added += 1

        self.log(f"已添加 {added} 个文件")
        if skipped_subtitle > 0 and not allow_subtitle_import:
            self.log(f"已忽略 {skipped_subtitle} 个字幕文件（导入开关已关闭）")
        self._update_summary()

    # ──────────────────────────────────────────────────────────
    #  任务操作
    # ──────────────────────────────────────────────────────────

    def on_remove_selected(self) -> None:
        if self.qm.is_running:
            QMessageBox.information(self, "任务进行中", "任务运行时无法删除行")
            return
        rows = sorted({idx.row() for idx in self.task_table.selectionModel().selectedRows()}, reverse=True)
        if not rows:
            return
        remove_ids = self.qm.get_selected_task_ids(rows)
        for row in rows:
            self.task_table.removeRow(row)
        self.qm.remove_tasks(remove_ids)
        self._rebuild_row_mapping()
        self.log(f"已删除 {len(rows)} 行所选任务")
        self._update_summary()

    def on_clear_all(self) -> None:
        if self.qm.is_running:
            QMessageBox.information(self, "任务进行中", "请先停止运行中的任务再清空")
            return
        self.task_table.setRowCount(0)
        self.qm.clear_all()
        self.total_progress.setValue(0)
        self._update_summary()
        self.log("已清空所有任务")

    def _rebuild_row_mapping(self) -> None:
        path_to_row: Dict[str, int] = {}
        for row in range(self.task_table.rowCount()):
            item = self.task_table.item(row, 0)
            if not item:
                continue
            path_to_row[normalize_path_key(Path(item.text()))] = row
        self.qm.rebuild_row_mapping(path_to_row)

    def on_start(self) -> None:
        if self.qm.is_running:
            return
        if not self.qm.tasks:
            QMessageBox.information(self, "没有任务", "请先添加文件")
            return
        try:
            settings = self._validate_settings()
        except Exception as exc:
            QMessageBox.warning(self, "设置无效", str(exc))
            return

        run_ids = [tid for tid, t in self.qm.tasks.items() if t.status in {"Queued", "Failed", "Cancelled"}]
        if not run_ids:
            QMessageBox.information(self, "没有可执行任务", "当前没有可运行的排队/失败/取消任务")
            return

        has_media = any(self.qm.tasks[tid].source_path.suffix.lower() in MEDIA_EXTENSIONS for tid in run_ids)
        has_subtitle = any(self.qm.tasks[tid].source_path.suffix.lower() in SUBTITLE_EXTENSIONS for tid in run_ids)
        has_video = any(self.qm.tasks[tid].source_path.suffix.lower() in VIDEO_EXTENSIONS for tid in run_ids)
        if (has_video or settings.vad.enabled) and not has_ffmpeg():
            QMessageBox.warning(self, "缺少 ffmpeg",
                "视频任务或 VAD 预切分需要 ffmpeg。当前运行环境未检测到内置或可用的 ffmpeg。")
            return
        if has_subtitle and settings.translation.mode == "none":
            QMessageBox.warning(self, "翻译未启用", "导入字幕任务需要启用翻译模式")
            return
        if has_subtitle and not settings.translation.allow_subtitle_import:
            QMessageBox.warning(self, "字幕导入已关闭", "请在设置中开启\u201c允许导入字幕文件并翻译\u201d")
            return
        if settings.transcription.provider == "mistral" and settings.transcription.timestamp_granularity != "none" and settings.transcription.language_mode == "manual":
            self.log("Mistral 启用时间戳粒度后，language 参数将被忽略")

        count = self.qm.start_batch(settings)
        if count == 0:
            return

        for tid in self.qm.active_run_ids:
            self._update_task_row(tid, "Queued", 0, "等待执行", "-")

        self._flush_timer.start()
        self._elapsed_timer.start()
        self._update_button_states()

        self.log(f"已启动 {count} 个任务，线程数：{settings.transcription.thread_count}")
        if has_media and settings.transcription.provider == "whisper_openai_compatible":
            self.log("转写后端：Whisper(OpenAI 兼容)")
        self.total_progress.setValue(0)
        self._update_summary()

    def on_stop(self) -> None:
        if not self.qm.is_running:
            return
        canceled = self.qm.stop_all()
        self.log(f"已请求停止，取消了 {canceled} 个排队任务")

    def on_batch_finished(self) -> None:
        self._flush_timer.stop()
        self._elapsed_timer.stop()
        self._update_button_states()
        summary = self.qm.get_summary()
        self.log(f"任务结束：成功={summary['done']}，失败={summary['failed']}，取消={summary['canceled']}")
        self.total_progress.setValue(0)
        self._update_summary()

    # ──────────────────────────────────────────────────────────
    #  单任务操作（暂停/恢复/取消/重试）
    # ──────────────────────────────────────────────────────────

    def _pause_selected(self) -> None:
        for row in {idx.row() for idx in self.task_table.selectionModel().selectedRows()}:
            tid = self.qm.get_task_id_by_row(row)
            if tid:
                self.qm.pause_task(tid)

    def _resume_selected(self) -> None:
        for row in {idx.row() for idx in self.task_table.selectionModel().selectedRows()}:
            tid = self.qm.get_task_id_by_row(row)
            if tid:
                self.qm.resume_task(tid)

    def _cancel_selected(self) -> None:
        for row in {idx.row() for idx in self.task_table.selectionModel().selectedRows()}:
            tid = self.qm.get_task_id_by_row(row)
            if tid:
                self.qm.cancel_task(tid)

    def _retry_selected(self) -> None:
        rows = {idx.row() for idx in self.task_table.selectionModel().selectedRows()}
        if rows:
            tids = self.qm.get_selected_task_ids(list(rows))
            reset = self.qm.retry_failed(tids)
        else:
            reset = self.qm.retry_failed()
        for tid in reset:
            self._update_task_row(tid, "Queued", 0, "等待重试", "-")
        if reset:
            self.log(f"已重置 {len(reset)} 个任务为排队状态")
            self._update_summary()

    # ──────────────────────────────────────────────────────────
    #  右键菜单
    # ──────────────────────────────────────────────────────────

    def _build_context_menu(self, menu: QMenu) -> None:
        rows = {idx.row() for idx in self.task_table.selectionModel().selectedRows()}
        if not rows:
            return

        has_running = has_paused = has_failed = False
        for row in rows:
            tid = self.qm.get_task_id_by_row(row)
            if not tid:
                continue
            s = self.qm.tasks[tid].status
            if s in {"Preparing", "Extracting", "Transcribing", "Translating", "Writing", "Queued"}:
                has_running = True
            if s == "Paused":
                has_paused = True
            if s in {"Failed", "Cancelled"}:
                has_failed = True

        if has_running:
            menu.addAction("取消所选任务", self._cancel_selected)
            menu.addAction("\u23f8 暂停所选任务", self._pause_selected)
        if has_paused:
            menu.addAction("\u25b6 恢复所选任务", self._resume_selected)
        if has_failed:
            menu.addAction("\u21bb 重试所选任务", self._retry_selected)

        menu.addSeparator()
        selected_tids = self.qm.get_selected_task_ids(list(rows))
        menu.addAction("上移优先级", lambda: self.qm.move_priority(selected_tids, +1))
        menu.addAction("下移优先级", lambda: self.qm.move_priority(selected_tids, -1))

        if not self.qm.is_running:
            menu.addSeparator()
            menu.addAction("删除所选", self.on_remove_selected)

    # ──────────────────────────────────────────────────────────
    #  拖拽排序
    # ──────────────────────────────────────────────────────────

    def _on_table_reorder(self) -> None:
        ordered: List[str] = []
        for row in range(self.task_table.rowCount()):
            item = self.task_table.item(row, 0)
            if not item:
                continue
            tid = self.qm.get_task_id_by_row(row)
            if tid:
                ordered.append(tid)
        if ordered:
            self.qm.reorder_by_rows(ordered)
        self._rebuild_row_mapping()

    # ──────────────────────────────────────────────────────────
    #  表格行更新
    # ──────────────────────────────────────────────────────────

    def _update_task_row(self, task_id: str, status: str, progress: int, message: str, outputs: str = "-") -> None:
        state = self.qm.tasks.get(task_id)
        if not state:
            return
        row = state.row
        if row >= self.task_table.rowCount():
            return

        # 状态列（带颜色）
        status_item = self.task_table.item(row, 1)
        if status_item:
            status_item.setText(STATUS_LABELS.get(status, status))
            status_item.setForeground(QColor(STATUS_COLORS.get(status, "#6b7280")))

        # 进度列
        progress_bar = self.task_table.cellWidget(row, 2)
        if isinstance(progress_bar, QProgressBar):
            progress_bar.setValue(progress)

        # 耗时列
        if state.start_time > 0:
            end = state.end_time if state.end_time > 0 else time.monotonic()
            elapsed_item = self.task_table.item(row, 3)
            if elapsed_item:
                elapsed_item.setText(self._format_duration(end - state.start_time))

        # 输出列
        out_item = self.task_table.item(row, 4)
        if out_item:
            out_item.setText(outputs)

        # 消息列
        msg_item = self.task_table.item(row, 5)
        if msg_item:
            display = message if len(message) <= 80 else message[:80] + "..."
            msg_item.setText(display)
            msg_item.setToolTip(message if len(message) > 80 else "")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds <= 0:
            return "-"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    # ──────────────────────────────────────────────────────────
    #  QueueManager 信号处理
    # ──────────────────────────────────────────────────────────

    def on_task_progress(self, task_id: str, status: str, progress: int, message: str) -> None:
        self._update_task_row(task_id, status, progress, message)
        self.total_progress.setValue(self.qm.get_total_progress())
        self._update_summary()
        self._update_button_states()

    def on_task_finished(self, task_id: str, success: bool, status: str, message: str, outputs: dict) -> None:
        state = self.qm.tasks.get(task_id)
        if not state:
            return
        output_text = " | ".join(v for k, v in outputs.items() if k != "_detail") if outputs else "-"
        self._update_task_row(task_id, status, 100 if success else state.progress, message, output_text)
        if success:
            self.log(f"[{task_id[:8]}] 已完成: {state.source_path.name}")
        else:
            self.log(f"[{task_id[:8]}] {STATUS_LABELS.get(status, status)}: {state.source_path.name} | {message}")
        self.total_progress.setValue(self.qm.get_total_progress())
        self._update_summary()
        self._update_button_states()

    def _update_elapsed_times(self) -> None:
        for tid in self.qm.get_running_task_ids():
            state = self.qm.tasks.get(tid)
            if state and state.start_time > 0:
                item = self.task_table.item(state.row, 3)
                if item:
                    item.setText(self._format_duration(time.monotonic() - state.start_time))

    # ──────────────────────────────────────────────────────────
    #  摘要 & 按钮状态
    # ──────────────────────────────────────────────────────────

    def _update_summary(self) -> None:
        s = self.qm.get_summary()
        total = s["total"]
        if total == 0:
            self.summary_label.setText("暂无任务")
            self.summary_label.setStyleSheet("font-weight: 600; color: #374151;")
            return
        if self.qm.is_running:
            done_batch = s["done"] + s["failed"] + s["canceled"]
            self.summary_label.setText(
                f"当前批次：已完成 {done_batch}/{total} | 运行中={s['running']} | 暂停={s['paused']}"
            )
            self.summary_label.setStyleSheet("font-weight: 600; color: #2563eb;")
        else:
            parts = [f"总数={total}", f"排队={s['queued']}", f"完成={s['done']}"]
            if s["failed"] > 0:
                parts.append(f"失败={s['failed']}")
            if s["canceled"] > 0:
                parts.append(f"取消={s['canceled']}")
            self.summary_label.setText(" | ".join(parts))
            color = "#dc2626" if s["failed"] > 0 else "#374151"
            self.summary_label.setStyleSheet(f"font-weight: 600; color: {color};")

    def _update_button_states(self) -> None:
        running = self.qm.is_running
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.pause_btn.setEnabled(running)
        self.resume_btn.setEnabled(running)
        self.remove_btn.setEnabled(not running)
        self.clear_btn.setEnabled(not running)
        self.retry_btn.setEnabled(not running and bool(self.qm.get_failed_task_ids()))

    # ──────────────────────────────────────────────────────────
    #  输出目录 & 日志
    # ──────────────────────────────────────────────────────────

    def on_open_output_dir(self) -> None:
        if self.output_mode_combo.currentData() == "source":
            selected_rows = self.task_table.selectionModel().selectedRows()
            if selected_rows:
                item = self.task_table.item(selected_rows[0].row(), 0)
                folder = Path(item.text()).parent if item else Path.cwd()
            elif self.qm.tasks:
                folder = next(iter(self.qm.tasks.values())).source_path.parent
            else:
                folder = Path.cwd()
        else:
            folder = Path(self.output_dir_input.text().strip() or str(Path.cwd() / "subtitles"))
            folder.mkdir(parents=True, exist_ok=True)

        if sys.platform.startswith("win"):
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(folder)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(folder)])

    def log(self, message: str) -> None:
        self.log_text.appendPlainText(message)

    # ──────────────────────────────────────────────────────────
    #  设置验证
    # ──────────────────────────────────────────────────────────

    def _validate_settings(self) -> AppSettings:
        settings = self.collect_settings_from_ui()
        if settings.transcription.language_mode == "manual" and not settings.transcription.language:
            raise RuntimeError("已选择指定语言，请填写有效语言代码，例如 zh / en")
        if settings.translation.mode != "none":
            if not settings.translation.target_language:
                raise RuntimeError("请填写目标语言代码，例如 zh / en / ja")
            if not settings.translation.model:
                raise RuntimeError("请填写翻译模型名称")
        if settings.translation.mode == "openai" and not settings.translation.openai_api_key:
            raise RuntimeError("OpenAI 兼容翻译模式需要填写 API Key")
        if settings.translation.mode == "mistral" and not settings.transcription.mistral.api_key:
            raise RuntimeError("Mistral 翻译模式需要填写 MISTRAL_API_KEY")
        if settings.transcription.provider == "mistral" and not settings.transcription.mistral.api_key:
            raise RuntimeError("Mistral 转写需要填写 MISTRAL_API_KEY")
        if settings.transcription.provider == "whisper_openai_compatible":
            if not settings.transcription.whisper.api_key:
                raise RuntimeError("Whisper 转写需要填写第三方/OpenAI 兼容 API Key")
            if not settings.transcription.whisper.model:
                raise RuntimeError("Whisper 转写需要填写模型名称")
        if not (settings.output.save_srt or settings.output.save_lrc or settings.output.save_txt or settings.output.save_json):
            raise RuntimeError("请至少选择一种输出格式")
        if settings.output.mode == "custom":
            settings.output.output_dir.mkdir(parents=True, exist_ok=True)
        if settings.transcription.provider == "mistral" and transcription_provider.Mistral is None:
            details = ""
            if transcription_provider._MISTRAL_IMPORT_ERROR is not None:
                details = (
                    f"（导入错误：{type(transcription_provider._MISTRAL_IMPORT_ERROR).__name__}: "
                    f"{transcription_provider._MISTRAL_IMPORT_ERROR}）"
                )
            raise RuntimeError(f"缺少依赖：mistral{details}")
        return settings

