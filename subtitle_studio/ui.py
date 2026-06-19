from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QKeySequence, QShortcut, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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
from .models import AppSettings
from .providers import transcription as transcription_provider
from .queue_manager import TaskQueueManager
from .theme import ThemeManager, ThemePalette
from .utils import normalize_language_code, normalize_path_key, parse_context_bias


def refresh_widget_style(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def is_valid_language_code(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z]{2,3}(?:[-_][a-z]{2,4})?", value.strip().lower()))


def is_valid_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value.strip())
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


class OptionCardButton(QPushButton):
    def __init__(self, title: str, description: str = "", parent: Optional[QWidget] = None) -> None:
        text = title if not description else f"{title}\n{description}"
        super().__init__(text, parent)
        self.setObjectName("optionCardButton")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(76)


class ModePillButton(QPushButton):
    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("modePillButton")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class DropFrame(QFrame):
    """拖拽区域，支持 hover 高亮。"""

    files_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("dropFrame")
        self.setProperty("dragActive", False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._icon_label = QLabel("⤴")
        self._icon_label.setObjectName("dropIconLabel")
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._title_label = QLabel("拖拽视频 / 音频 / 字幕文件到这里")
        self._title_label.setObjectName("dropTitleLabel")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._hint_label = QLabel("也支持整个文件夹，程序会自动递归识别可处理文件")
        self._hint_label.setObjectName("dropHintLabel")
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setWordWrap(True)

        layout.addWidget(self._icon_label)
        layout.addWidget(self._title_label)
        layout.addWidget(self._hint_label)

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
        self.setProperty("dragActive", hover)
        refresh_widget_style(self)


class TaskTableWidget(QTableWidget):
    """支持右键菜单和拖拽排序的任务表格。"""

    reorder_requested = Signal()

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
        self.verticalHeader().setDefaultSectionSize(36)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setDragDropMode(QTableWidget.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu(self)
        mw = self.window()
        if hasattr(mw, "_build_context_menu"):
            mw._build_context_menu(menu)  # type: ignore[attr-defined]
        if menu.actions():
            menu.exec(event.globalPos())

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        super().dropEvent(event)
        self.reorder_requested.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("mainWindow")
        self.setWindowTitle("Subtitle Studio 字幕工作台")
        self.resize(1320, 900)
        self._palette: ThemePalette = ThemeManager.palette("light")
        self._button_groups: list[QButtonGroup] = []

        self.qm = TaskQueueManager(self)
        self.qm.task_progress.connect(self.on_task_progress)
        self.qm.task_finished.connect(self.on_task_finished)
        self.qm.batch_finished.connect(self.on_batch_finished)

        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(120)
        self._flush_timer.timeout.connect(self.qm.flush_pending_progress)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed_times)

        self.init_ui()
        self.apply_style("light")
        self.load_settings_into_ui()

    def init_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_task_page(), "任务")
        self.tabs.addTab(self._build_settings_page(), "设置")
        root_layout.addWidget(self.tabs)
        self.setCentralWidget(root)

        QShortcut(QKeySequence("Delete"), self, self.on_remove_selected)
        QShortcut(QKeySequence("F5"), self, self.on_start)
        QShortcut(QKeySequence("Escape"), self, self.on_stop)
        QShortcut(QKeySequence("Ctrl+R"), self, self._retry_selected)

    def _build_task_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("taskPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.drop_frame = DropFrame()
        self.drop_frame.files_dropped.connect(self.on_drop_paths)
        layout.addWidget(self.drop_frame)

        toolbar_card = QFrame()
        toolbar_card.setObjectName("taskToolbarCard")
        btn_layout = QHBoxLayout(toolbar_card)
        btn_layout.setContentsMargins(14, 12, 14, 12)
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
        layout.addWidget(toolbar_card)

        self.task_table = TaskTableWidget()
        self.task_table.reorder_requested.connect(self._on_table_reorder)
        layout.addWidget(self.task_table, 1)

        self.total_progress = QProgressBar()
        self.total_progress.setObjectName("totalProgress")
        self.total_progress.setRange(0, 100)
        self.total_progress.setValue(0)
        self.total_progress.setTextVisible(False)
        self.summary_label = QLabel("暂无任务")
        self.summary_label.setObjectName("summaryLabel")
        self.summary_label.setProperty("summaryState", "idle")
        refresh_widget_style(self.summary_label)
        layout.addWidget(self.total_progress)
        layout.addWidget(self.summary_label)

        log_card = QFrame()
        log_card.setObjectName("logCard")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(14, 12, 14, 12)
        log_layout.setSpacing(8)

        log_header = QWidget()
        log_header_layout = QHBoxLayout(log_header)
        log_header_layout.setContentsMargins(0, 0, 0, 0)
        log_header_layout.setSpacing(8)
        log_title = QLabel("运行日志")
        log_title.setObjectName("sectionTitle")
        self.log_autoscroll_checkbox = QCheckBox("自动滚动")
        self.log_autoscroll_checkbox.setChecked(True)
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.setObjectName("secondaryBtn")
        self.clear_log_btn.clicked.connect(self.on_clear_log)
        log_header_layout.addWidget(log_title)
        log_header_layout.addStretch(1)
        log_header_layout.addWidget(self.log_autoscroll_checkbox)
        log_header_layout.addWidget(self.clear_log_btn)

        self.log_text = QPlainTextEdit()
        self.log_text.setObjectName("logText")
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(160)
        self.log_text.setPlaceholderText("运行中的任务进度、告警和错误会显示在这里")

        log_layout.addWidget(log_header)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_card)
        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("settingsPage")
        outer_layout = QVBoxLayout(page)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(12)

        header_card = QFrame()
        header_card.setObjectName("settingsHeaderCard")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(18, 16, 18, 16)
        header_layout.setSpacing(12)

        header_text = QVBoxLayout()
        header_text.setContentsMargins(0, 0, 0, 0)
        header_text.setSpacing(4)
        hero_title = QLabel("设置中心")
        hero_title.setObjectName("heroTitle")
        hero_desc = QLabel("按任务流程组织设置项，常用参数靠前，高级控制收拢到更合理的位置。")
        hero_desc.setObjectName("heroDescription")
        hero_desc.setWordWrap(True)
        header_text.addWidget(hero_title)
        header_text.addWidget(hero_desc)

        theme_box = QHBoxLayout()
        theme_box.setContentsMargins(0, 0, 0, 0)
        theme_box.setSpacing(8)
        theme_label = self._field_label("界面主题")
        self.theme_combo = NoWheelComboBox()
        self.theme_combo.addItem("浅色", "light")
        self.theme_combo.addItem("深色", "dark")
        self.theme_combo.currentIndexChanged.connect(self.on_theme_changed)
        theme_box.addWidget(theme_label)
        theme_box.addWidget(self.theme_combo)

        header_layout.addLayout(header_text, 1)
        header_layout.addLayout(theme_box)
        outer_layout.addWidget(header_card)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)

        self.settings_nav = QListWidget()
        self.settings_nav.setObjectName("settingsNav")
        self.settings_nav.setFixedWidth(180)
        self.settings_nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.settings_nav.setSpacing(4)

        for title in ("🎙 转写", "🌐 翻译", "📤 输出", "🔉 预处理"):
            self.settings_nav.addItem(QListWidgetItem(title))

        self.settings_stack = QStackedWidget()
        self.settings_stack.addWidget(self._wrap_settings_panel(self._build_transcription_panel()))
        self.settings_stack.addWidget(self._wrap_settings_panel(self._build_translation_panel()))
        self.settings_stack.addWidget(self._wrap_settings_panel(self._build_output_panel()))
        self.settings_stack.addWidget(self._wrap_settings_panel(self._build_preprocess_panel()))

        self.settings_nav.currentRowChanged.connect(self.settings_stack.setCurrentIndex)
        self.settings_nav.setCurrentRow(0)

        body_layout.addWidget(self.settings_nav)
        body_layout.addWidget(self.settings_stack, 1)
        outer_layout.addWidget(body, 1)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.addStretch(1)
        self.save_settings_btn = QPushButton("保存设置")
        self.save_settings_btn.setObjectName("primaryBtn")
        self.save_settings_btn.clicked.connect(self.on_save_settings)
        footer_layout.addWidget(self.save_settings_btn)
        outer_layout.addWidget(footer)
        return page

    def _wrap_settings_panel(self, panel: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(panel)
        return scroll

    def _create_settings_panel(self, title: str, description: str) -> tuple[QWidget, QVBoxLayout]:
        panel = QWidget()
        panel.setObjectName("settingsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        desc_label = QLabel(description)
        desc_label.setObjectName("pageDescription")
        desc_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        return panel, layout

    def _build_card(self, title: str, description: str = "") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("settingsCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        layout.addWidget(title_label)
        if description:
            desc_label = QLabel(description)
            desc_label.setObjectName("sectionDescription")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)
        layout.addLayout(body)
        return card, body

    def _form_grid(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)
        grid.setColumnStretch(1, 1)
        return grid

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def _build_secret_row(self, env_key: str = "") -> tuple[QWidget, QLineEdit, QCheckBox]:
        line_edit = QLineEdit(os.environ.get(env_key, "") if env_key else "")
        line_edit.setEchoMode(QLineEdit.EchoMode.Password)
        reveal_checkbox = QCheckBox("显示")
        reveal_checkbox.toggled.connect(
            lambda checked, target=line_edit: target.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(line_edit)
        layout.addWidget(reveal_checkbox)
        return row, line_edit, reveal_checkbox

    def _build_choice_buttons(
        self,
        combo: QComboBox,
        options: list[tuple[str, str, str]],
        compact: bool = False,
    ) -> tuple[QWidget, dict[str, QPushButton]]:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        group = QButtonGroup(container)
        group.setExclusive(True)
        self._button_groups.append(group)
        mapping: dict[str, QPushButton] = {}

        for index, (title, description, data) in enumerate(options):
            button = ModePillButton(title) if compact else OptionCardButton(title, description)
            layout.addWidget(button)
            group.addButton(button, index)
            mapping[data] = button
            button.clicked.connect(
                lambda checked, idx=index, target=combo: target.setCurrentIndex(idx) if checked else None
            )
        return container, mapping

    def _sync_choice_buttons(self, combo: QComboBox, mapping: dict[str, QPushButton]) -> None:
        current = combo.currentData()
        for data, button in mapping.items():
            blocked = button.blockSignals(True)
            button.setChecked(data == current)
            button.blockSignals(blocked)

    def _build_transcription_panel(self) -> QWidget:
        panel, layout = self._create_settings_panel(
            "转写设置",
            "先选择转写引擎，再填写该引擎真正需要的凭据和模型；无关项会自动收起。",
        )

        credentials_card, credentials_body = self._build_card("引擎凭据与模型")
        credentials_grid = self._form_grid()
        credentials_body.addLayout(credentials_grid)

        self.whisper_base_url_label = self._field_label("Whisper Base URL")
        self.whisper_base_url_input = QLineEdit("https://api.openai.com/v1")
        self.whisper_api_key_label = self._field_label("Whisper API Key")
        self.whisper_key_row, self.whisper_api_key_input, self.show_whisper_key_checkbox = self._build_secret_row(
            "OPENAI_API_KEY"
        )
        self.whisper_model_label = self._field_label("Whisper 模型")
        self.whisper_model_input = QLineEdit("whisper-1")

        credentials_grid.addWidget(self.whisper_base_url_label, 0, 0)
        credentials_grid.addWidget(self.whisper_base_url_input, 0, 1)
        credentials_grid.addWidget(self.whisper_api_key_label, 1, 0)
        credentials_grid.addWidget(self.whisper_key_row, 1, 1)
        credentials_grid.addWidget(self.whisper_model_label, 2, 0)
        credentials_grid.addWidget(self.whisper_model_input, 2, 1)
        layout.addWidget(credentials_card)

        advanced_card, advanced_body = self._build_card(
            "识别策略与高级参数",
            "这里放与语言、时间戳、线程和重试行为相关的控制项，常用但不该抢占首页视线。",
        )
        advanced_grid = self._form_grid()
        advanced_body.addLayout(advanced_grid)

        self.language_mode_label = self._field_label("语言模式")
        self.language_mode_combo = NoWheelComboBox()
        self.language_mode_combo.addItems(["自动识别", "指定语言"])
        self.language_mode_combo.currentIndexChanged.connect(self.on_language_mode_changed)
        self.language_label = self._field_label("指定语言")
        self.language_input = QLineEdit("zh")
        self.timestamp_label = self._field_label("时间戳粒度")
        self.timestamp_combo = NoWheelComboBox()
        self.timestamp_combo.addItems(["none", "segment", "word"])
        self.timestamp_combo.setCurrentText("segment")
        self.timestamp_combo.currentIndexChanged.connect(self.on_timestamp_granularity_changed)
        self.thread_label = self._field_label("任务线程数")
        self.thread_spin = NoWheelSpinBox()
        self.thread_spin.setRange(1, 16)
        self.thread_spin.setValue(3)
        self.max_retries_label = self._field_label("失败后重试次数")
        self.max_retries_spin = NoWheelSpinBox()
        self.max_retries_spin.setRange(0, 10)
        self.max_retries_spin.setValue(3)
        self.retry_base_delay_label = self._field_label("重试退避基准（秒）")
        self.retry_base_delay_spin = NoWheelDoubleSpinBox()
        self.retry_base_delay_spin.setRange(0.1, 30.0)
        self.retry_base_delay_spin.setDecimals(1)
        self.retry_base_delay_spin.setSingleStep(0.5)
        self.retry_base_delay_spin.setValue(2.0)
        self.context_bias_label = self._field_label("术语提示")
        self.context_bias_input = QPlainTextEdit()
        self.context_bias_input.setPlaceholderText("术语提示 / 上下文偏置，支持逗号或换行分隔")
        self.context_bias_input.setFixedHeight(80)

        advanced_grid.addWidget(self.language_mode_label, 0, 0)
        advanced_grid.addWidget(self.language_mode_combo, 0, 1)
        advanced_grid.addWidget(self.language_label, 1, 0)
        advanced_grid.addWidget(self.language_input, 1, 1)
        advanced_grid.addWidget(self.timestamp_label, 2, 0)
        advanced_grid.addWidget(self.timestamp_combo, 2, 1)
        advanced_grid.addWidget(self.thread_label, 3, 0)
        advanced_grid.addWidget(self.thread_spin, 3, 1)
        advanced_grid.addWidget(self.max_retries_label, 4, 0)
        advanced_grid.addWidget(self.max_retries_spin, 4, 1)
        advanced_grid.addWidget(self.retry_base_delay_label, 5, 0)
        advanced_grid.addWidget(self.retry_base_delay_spin, 5, 1)
        advanced_grid.addWidget(self.context_bias_label, 6, 0)
        advanced_grid.addWidget(self.context_bias_input, 6, 1)
        layout.addWidget(advanced_card)

        segmentation_card, segmentation_body = self._build_card(
            "智能分段",
            "只在词级时间戳可用时启用；分段完成后，翻译与写出会沿用重建后的片段。",
        )
        self.enable_segmentation_checkbox = QCheckBox("启用智能分段（仅 word 时间戳）")
        self.enable_segmentation_checkbox.toggled.connect(self.on_intelligent_segmentation_changed)
        self.segmentation_hint_label = QLabel("仅在时间戳粒度为 word 时生效；将调用专用 OpenAI 兼容 API 进行语义分段。")
        self.segmentation_hint_label.setObjectName("mutedLabel")
        self.segmentation_hint_label.setWordWrap(True)
        self.segmentation_thinking_checkbox = QCheckBox("启用思考模式（Reasoning）")
        self.segmentation_thinking_checkbox.setToolTip(
            "开启后将使用模型的思考能力来做语义分段；关闭时可手动通过温度控制输出稳定性。"
        )
        self.segmentation_thinking_checkbox.toggled.connect(self.on_segmentation_thinking_changed)

        segmentation_grid = self._form_grid()
        self.segmentation_base_url_label = self._field_label("智能分段 Base URL")
        self.segmentation_base_url_input = QLineEdit("https://api.openai.com/v1")
        self.segmentation_api_key_label = self._field_label("智能分段 API Key")
        self.segmentation_key_row, self.segmentation_api_key_input, self.show_segmentation_key_checkbox = self._build_secret_row(
            "SEGMENTATION_OPENAI_API_KEY"
        )
        self.segmentation_model_label = self._field_label("智能分段模型")
        self.segmentation_model_input = QLineEdit("gpt-4o-mini")
        self.segmentation_reasoning_effort_label = self._field_label("思考强度")
        self.segmentation_reasoning_effort_combo = NoWheelComboBox()
        self.segmentation_reasoning_effort_combo.addItem("low", "low")
        self.segmentation_reasoning_effort_combo.addItem("medium", "medium")
        self.segmentation_reasoning_effort_combo.addItem("high", "high")
        self.segmentation_reasoning_effort_combo.addItem("max", "max")
        self.segmentation_reasoning_effort_combo.setCurrentText("high")
        self.segmentation_temperature_label = self._field_label("分段温度")
        self.segmentation_temperature_spin = NoWheelDoubleSpinBox()
        self.segmentation_temperature_spin.setRange(0.0, 2.0)
        self.segmentation_temperature_spin.setDecimals(1)
        self.segmentation_temperature_spin.setSingleStep(0.1)
        self.segmentation_temperature_spin.setValue(0.1)
        self.segmentation_window_label = self._field_label("窗口最大词数")
        self.segmentation_window_spin = NoWheelSpinBox()
        self.segmentation_window_spin.setRange(50, 500)
        self.segmentation_window_spin.setSingleStep(10)
        self.segmentation_window_spin.setValue(180)

        segmentation_grid.addWidget(self.segmentation_base_url_label, 0, 0)
        segmentation_grid.addWidget(self.segmentation_base_url_input, 0, 1)
        segmentation_grid.addWidget(self.segmentation_api_key_label, 1, 0)
        segmentation_grid.addWidget(self.segmentation_key_row, 1, 1)
        segmentation_grid.addWidget(self.segmentation_model_label, 2, 0)
        segmentation_grid.addWidget(self.segmentation_model_input, 2, 1)
        segmentation_grid.addWidget(self.segmentation_reasoning_effort_label, 3, 0)
        segmentation_grid.addWidget(self.segmentation_reasoning_effort_combo, 3, 1)
        segmentation_grid.addWidget(self.segmentation_temperature_label, 4, 0)
        segmentation_grid.addWidget(self.segmentation_temperature_spin, 4, 1)
        segmentation_grid.addWidget(self.segmentation_window_label, 5, 0)
        segmentation_grid.addWidget(self.segmentation_window_spin, 5, 1)

        segmentation_body.addWidget(self.enable_segmentation_checkbox)
        segmentation_body.addWidget(self.segmentation_hint_label)
        segmentation_body.addWidget(self.segmentation_thinking_checkbox)
        segmentation_body.addLayout(segmentation_grid)
        layout.addWidget(segmentation_card)
        layout.addStretch(1)

        self.segmentation_config_widgets = [
            self.segmentation_base_url_label,
            self.segmentation_base_url_input,
            self.segmentation_api_key_label,
            self.segmentation_key_row,
            self.segmentation_model_label,
            self.segmentation_model_input,
            self.segmentation_thinking_checkbox,
            self.segmentation_reasoning_effort_label,
            self.segmentation_reasoning_effort_combo,
            self.segmentation_temperature_label,
            self.segmentation_temperature_spin,
            self.segmentation_window_label,
            self.segmentation_window_spin,
        ]
        return panel

    def _build_translation_panel(self) -> QWidget:
        panel, layout = self._create_settings_panel(
            "翻译设置",
            "把基础参数、输出选项和高级推理参数拆开显示，避免所有开关在同一块里抢戏。",
        )

        self.translation_mode_combo = NoWheelComboBox(panel)
        self.translation_mode_combo.addItem("不翻译", "none")
        self.translation_mode_combo.addItem("OpenAI 兼容 API 翻译", "openai")
        self.translation_mode_combo.currentIndexChanged.connect(self.on_translation_mode_changed)

        mode_card, mode_body = self._build_card("翻译模式")
        mode_buttons, self.translation_mode_buttons = self._build_choice_buttons(
            self.translation_mode_combo,
            [
                ("不翻译", "只输出转写结果", "none"),
                ("OpenAI 兼容", "适配 DeepSeek / OpenAI / 自建兼容服务", "openai"),
            ],
        )
        mode_body.addWidget(mode_buttons)
        layout.addWidget(mode_card)

        self.translation_basic_card, basic_body = self._build_card("基础参数")
        basic_grid = self._form_grid()
        basic_body.addLayout(basic_grid)
        self.translation_target_label = self._field_label("目标语言")
        self.translation_target_input = QLineEdit("zh")
        self.translation_model_label = self._field_label("翻译模型")
        self.translation_model_input = QLineEdit("gpt-4o-mini")
        self.subtitle_translation_thread_label = self._field_label("字幕翻译线程数")
        self.subtitle_translation_thread_spin = NoWheelSpinBox()
        self.subtitle_translation_thread_spin.setRange(1, 16)
        self.subtitle_translation_thread_spin.setValue(3)
        basic_grid.addWidget(self.translation_target_label, 0, 0)
        basic_grid.addWidget(self.translation_target_input, 0, 1)
        basic_grid.addWidget(self.translation_model_label, 1, 0)
        basic_grid.addWidget(self.translation_model_input, 1, 1)
        basic_grid.addWidget(self.subtitle_translation_thread_label, 2, 0)
        basic_grid.addWidget(self.subtitle_translation_thread_spin, 2, 1)
        layout.addWidget(self.translation_basic_card)

        self.translation_output_card, output_body = self._build_card("翻译输出行为")
        self.translation_bilingual_checkbox = QCheckBox("SRT 输出双语（原文 + 译文）")
        self.translation_bilingual_checkbox.setChecked(True)
        self.translation_keep_original_checkbox = QCheckBox("翻译后额外输出原文字幕（xxx.orig.srt）")
        self.allow_subtitle_import_checkbox = QCheckBox("允许导入字幕文件并翻译")
        self.allow_subtitle_import_checkbox.setChecked(True)
        output_body.addWidget(self.translation_bilingual_checkbox)
        output_body.addWidget(self.translation_keep_original_checkbox)
        output_body.addWidget(self.allow_subtitle_import_checkbox)
        layout.addWidget(self.translation_output_card)

        self.translation_reasoning_card, reasoning_body = self._build_card(
            "思考与高级参数",
            "思考模式适合追求更稳的翻译质量；关闭思考后，可手动调温度来控制输出风格。",
        )
        self.translation_thinking_checkbox = QCheckBox("启用思考模式（Reasoning）")
        self.translation_thinking_checkbox.setToolTip(
            "开启后将使用模型的深度思考能力，可能提升翻译质量，但也会增加延迟和消耗。"
        )
        self.translation_thinking_checkbox.toggled.connect(self.on_translation_thinking_changed)
        reasoning_grid = self._form_grid()
        self.translation_reasoning_effort_label = self._field_label("思考强度")
        self.translation_reasoning_effort_combo = NoWheelComboBox()
        self.translation_reasoning_effort_combo.addItem("low", "low")
        self.translation_reasoning_effort_combo.addItem("medium", "medium")
        self.translation_reasoning_effort_combo.addItem("high", "high")
        self.translation_reasoning_effort_combo.addItem("max", "max")
        self.translation_reasoning_effort_combo.setCurrentText("high")
        self.translation_temperature_label = self._field_label("翻译温度")
        self.translation_temperature_spin = NoWheelDoubleSpinBox()
        self.translation_temperature_spin.setRange(0.0, 2.0)
        self.translation_temperature_spin.setDecimals(1)
        self.translation_temperature_spin.setSingleStep(0.1)
        self.translation_temperature_spin.setValue(0.2)
        self.translation_chunk_size_label = self._field_label("每批翻译条数")
        self.translation_chunk_size_spin = NoWheelSpinBox()
        self.translation_chunk_size_spin.setRange(10, 100)
        self.translation_chunk_size_spin.setSingleStep(5)
        self.translation_chunk_size_spin.setValue(40)
        reasoning_grid.addWidget(self.translation_reasoning_effort_label, 0, 0)
        reasoning_grid.addWidget(self.translation_reasoning_effort_combo, 0, 1)
        reasoning_grid.addWidget(self.translation_temperature_label, 1, 0)
        reasoning_grid.addWidget(self.translation_temperature_spin, 1, 1)
        reasoning_grid.addWidget(self.translation_chunk_size_label, 2, 0)
        reasoning_grid.addWidget(self.translation_chunk_size_spin, 2, 1)
        reasoning_body.addWidget(self.translation_thinking_checkbox)
        reasoning_body.addLayout(reasoning_grid)
        layout.addWidget(self.translation_reasoning_card)

        self.translation_access_card, access_body = self._build_card("接口接入")
        access_grid = self._form_grid()
        self.translation_openai_base_label = self._field_label("OpenAI 兼容 Base URL")
        self.translation_openai_base_input = QLineEdit("https://api.openai.com/v1")
        self.translation_openai_key_label = self._field_label("OpenAI 兼容 API Key")
        self.translation_openai_key_row, self.translation_openai_key_input, self.show_translation_openai_key_checkbox = self._build_secret_row(
            "OPENAI_API_KEY"
        )
        access_grid.addWidget(self.translation_openai_base_label, 0, 0)
        access_grid.addWidget(self.translation_openai_base_input, 0, 1)
        access_grid.addWidget(self.translation_openai_key_label, 1, 0)
        access_grid.addWidget(self.translation_openai_key_row, 1, 1)
        access_body.addLayout(access_grid)
        layout.addWidget(self.translation_access_card)
        layout.addStretch(1)

        self.translation_common_cards = [
            self.translation_basic_card,
            self.translation_output_card,
            self.translation_reasoning_card,
            self.translation_access_card,
        ]
        self.translation_openai_widgets = [
            self.translation_openai_base_label,
            self.translation_openai_base_input,
            self.translation_openai_key_label,
            self.translation_openai_key_row,
        ]
        return panel

    def _build_output_panel(self) -> QWidget:
        panel, layout = self._create_settings_panel(
            "输出设置",
            "把目标目录和输出格式拆分显示，常规导出选项一眼就能看明白。",
        )

        self.output_mode_combo = NoWheelComboBox(panel)
        self.output_mode_combo.addItem("输出到原文件目录", "source")
        self.output_mode_combo.addItem("输出到指定目录", "custom")
        self.output_mode_combo.currentIndexChanged.connect(self.on_output_mode_changed)

        location_card, location_body = self._build_card("输出位置")
        location_buttons, self.output_mode_buttons = self._build_choice_buttons(
            self.output_mode_combo,
            [
                ("输出到原目录", "与源文件放在一起", "source"),
                ("指定目录", "集中导出到单独文件夹", "custom"),
            ],
            compact=True,
        )
        location_body.addWidget(location_buttons)

        location_grid = self._form_grid()
        self.output_dir_label = self._field_label("指定输出目录")
        self.output_dir_input = QLineEdit(str(Path.cwd() / "subtitles"))
        self.output_btn = QPushButton("浏览")
        self.output_btn.setObjectName("secondaryBtn")
        self.output_btn.clicked.connect(self.on_choose_output_dir)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(8)
        output_layout.addWidget(self.output_dir_input)
        output_layout.addWidget(self.output_btn)
        location_grid.addWidget(self.output_dir_label, 0, 0)
        location_grid.addWidget(output_row, 0, 1)
        location_body.addLayout(location_grid)
        layout.addWidget(location_card)

        format_card, format_body = self._build_card("输出格式")
        self.save_srt_checkbox = QCheckBox("保存 .srt")
        self.save_srt_checkbox.setChecked(True)
        self.save_lrc_checkbox = QCheckBox("纯音频保存 .lrc")
        self.save_lrc_checkbox.setChecked(True)
        self.save_txt_checkbox = QCheckBox("保存 .txt")
        self.save_txt_checkbox.setChecked(True)
        self.save_json_checkbox = QCheckBox("保存 .json")
        format_grid = QGridLayout()
        format_grid.setContentsMargins(0, 0, 0, 0)
        format_grid.setHorizontalSpacing(16)
        format_grid.setVerticalSpacing(12)
        format_grid.addWidget(self.save_srt_checkbox, 0, 0)
        format_grid.addWidget(self.save_lrc_checkbox, 0, 1)
        format_grid.addWidget(self.save_txt_checkbox, 1, 0)
        format_grid.addWidget(self.save_json_checkbox, 1, 1)
        format_body.addLayout(format_grid)
        layout.addWidget(format_card)
        layout.addStretch(1)
        return panel

    def _build_preprocess_panel(self) -> QWidget:
        panel, layout = self._create_settings_panel(
            "预处理（VAD）",
            "VAD 适合长音频和停顿明显的内容，能先切分音频再交给转写引擎处理。",
        )

        intro_card, intro_body = self._build_card("功能说明")
        self.ffmpeg_hint_label = QLabel("视频任务与 VAD 预切分会自动使用软件内置 ffmpeg，无需单独配置路径。")
        self.ffmpeg_hint_label.setObjectName("mutedLabel")
        self.ffmpeg_hint_label.setWordWrap(True)
        intro_body.addWidget(self.ffmpeg_hint_label)
        layout.addWidget(intro_card)

        vad_card, vad_body = self._build_card("VAD 参数")
        self.enable_vad_checkbox = QCheckBox("启用 Silero VAD 预切分")
        self.enable_vad_checkbox.setChecked(False)
        self.enable_vad_checkbox.toggled.connect(self.on_vad_enabled_changed)
        vad_grid = self._form_grid()

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

        vad_grid.addWidget(self._field_label("最短语音"), 0, 0)
        vad_grid.addWidget(self.vad_min_speech_spin, 0, 1)
        vad_grid.addWidget(self._field_label("最短静音"), 1, 0)
        vad_grid.addWidget(self.vad_min_silence_spin, 1, 1)
        vad_grid.addWidget(self._field_label("语音补边"), 2, 0)
        vad_grid.addWidget(self.vad_speech_pad_spin, 2, 1)
        vad_grid.addWidget(self._field_label("单段最长时长"), 3, 0)
        vad_grid.addWidget(self.vad_max_segment_spin, 3, 1)
        vad_grid.addWidget(self._field_label("检测阈值"), 4, 0)
        vad_grid.addWidget(self.vad_threshold_spin, 4, 1)

        vad_body.addWidget(self.enable_vad_checkbox)
        vad_body.addLayout(vad_grid)
        layout.addWidget(vad_card)
        layout.addStretch(1)
        return panel

    def _set_widgets_visible(self, widgets: List[QWidget], visible: bool) -> None:
        for widget in widgets:
            widget.setVisible(visible)

    def _set_summary_state(self, state: str) -> None:
        self.summary_label.setProperty("summaryState", state)
        refresh_widget_style(self.summary_label)

    def refresh_settings_visibility(self) -> None:
        self.on_language_mode_changed()
        self.on_translation_mode_changed()
        self.on_transcription_provider_changed()
        self.on_translation_thinking_changed()
        self.on_segmentation_thinking_changed()
        self.on_timestamp_granularity_changed()
        self.on_intelligent_segmentation_changed()
        self.on_output_mode_changed()
        self.on_vad_enabled_changed()

    def apply_style(self, theme_name: str | None = None) -> None:
        if theme_name:
            self._palette = ThemeManager.palette(theme_name)
        app = QApplication.instance()
        if app is not None:
            self._palette = ThemeManager.apply_theme(app, self._palette.name)
        self._set_summary_state(str(self.summary_label.property("summaryState") or "idle"))

    def log(self, message: str) -> None:
        self.log_text.appendPlainText(message)
        if self.log_autoscroll_checkbox.isChecked():
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def on_clear_log(self) -> None:
        self.log_text.clear()

    def load_settings_into_ui(self) -> None:
        settings = load_settings()
        self.apply_settings_to_ui(settings)
        self.log("已加载本地设置")
        self.refresh_settings_visibility()

    def apply_settings_to_ui(self, settings: AppSettings) -> None:
        theme_index = {"light": 0, "dark": 1}.get(settings.ui_theme, 0)
        self.theme_combo.setCurrentIndex(theme_index)

        self.whisper_base_url_input.setText(settings.transcription.whisper.base_url)
        self.whisper_api_key_input.setText(settings.transcription.whisper.api_key)
        self.whisper_model_input.setText(settings.transcription.whisper.model)
        self.language_mode_combo.setCurrentIndex(1 if settings.transcription.language_mode == "manual" else 0)
        self.language_input.setText(settings.transcription.language)
        self.timestamp_combo.setCurrentText(settings.transcription.timestamp_granularity)
        self.thread_spin.setValue(settings.transcription.thread_count)
        self.max_retries_spin.setValue(settings.transcription.max_retries)
        self.retry_base_delay_spin.setValue(settings.retry_base_delay)
        self.context_bias_input.setPlainText(settings.transcription.context_bias)

        self.enable_segmentation_checkbox.setChecked(settings.segmentation.enabled)
        self.segmentation_base_url_input.setText(settings.segmentation.openai_base_url)
        self.segmentation_api_key_input.setText(settings.segmentation.openai_api_key)
        self.segmentation_model_input.setText(settings.segmentation.model)
        self.segmentation_thinking_checkbox.setChecked(settings.segmentation.thinking_enabled)
        self.segmentation_reasoning_effort_combo.setCurrentText(settings.segmentation.reasoning_effort)
        self.segmentation_temperature_spin.setValue(settings.segmentation.temperature)
        self.segmentation_window_spin.setValue(settings.segmentation.max_words_per_window)

        self.translation_mode_combo.setCurrentIndex({"none": 0, "openai": 1}.get(settings.translation.mode, 0))
        self.translation_target_input.setText(settings.translation.target_language)
        self.translation_model_input.setText(settings.translation.model)
        self.translation_bilingual_checkbox.setChecked(settings.translation.bilingual_srt)
        self.translation_keep_original_checkbox.setChecked(settings.translation.keep_original_srt)
        self.allow_subtitle_import_checkbox.setChecked(settings.translation.allow_subtitle_import)
        self.subtitle_translation_thread_spin.setValue(settings.translation.subtitle_translation_thread_count)
        self.translation_thinking_checkbox.setChecked(settings.translation.thinking_enabled)
        self.translation_reasoning_effort_combo.setCurrentText(settings.translation.reasoning_effort)
        self.translation_temperature_spin.setValue(settings.translation.temperature)
        self.translation_chunk_size_spin.setValue(settings.translation.chunk_size)
        self.translation_openai_base_input.setText(settings.translation.openai_base_url)
        self.translation_openai_key_input.setText(settings.translation.openai_api_key)

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
        settings.ui_theme = self.theme_combo.currentData() or "light"
        settings.retry_base_delay = self.retry_base_delay_spin.value()

        settings.transcription.provider = "whisper_openai_compatible"
        settings.transcription.whisper.base_url = self.whisper_base_url_input.text().strip() or "https://api.openai.com/v1"
        settings.transcription.whisper.api_key = self.whisper_api_key_input.text().strip()
        settings.transcription.whisper.model = self.whisper_model_input.text().strip() or "whisper-1"
        settings.transcription.language_mode = "manual" if self.language_mode_combo.currentIndex() == 1 else "auto"
        settings.transcription.language = normalize_language_code(self.language_input.text().strip())
        settings.transcription.timestamp_granularity = self.timestamp_combo.currentText().strip() or "none"
        settings.transcription.thread_count = self.thread_spin.value()
        settings.transcription.max_retries = self.max_retries_spin.value()
        settings.transcription.context_bias = parse_context_bias(self.context_bias_input.toPlainText())

        settings.segmentation.enabled = self.enable_segmentation_checkbox.isChecked()
        settings.segmentation.openai_base_url = self.segmentation_base_url_input.text().strip() or "https://api.openai.com/v1"
        settings.segmentation.openai_api_key = self.segmentation_api_key_input.text().strip()
        settings.segmentation.model = self.segmentation_model_input.text().strip() or "gpt-4o-mini"
        settings.segmentation.thinking_enabled = self.segmentation_thinking_checkbox.isChecked()
        settings.segmentation.reasoning_effort = self.segmentation_reasoning_effort_combo.currentData() or "high"
        settings.segmentation.temperature = self.segmentation_temperature_spin.value()
        settings.segmentation.max_words_per_window = self.segmentation_window_spin.value()

        settings.translation.mode = self.translation_mode_combo.currentData()
        settings.translation.target_language = normalize_language_code(self.translation_target_input.text().strip())
        settings.translation.model = self.translation_model_input.text().strip()
        settings.translation.bilingual_srt = self.translation_bilingual_checkbox.isChecked()
        settings.translation.keep_original_srt = self.translation_keep_original_checkbox.isChecked()
        settings.translation.allow_subtitle_import = self.allow_subtitle_import_checkbox.isChecked()
        settings.translation.subtitle_translation_thread_count = self.subtitle_translation_thread_spin.value()
        settings.translation.thinking_enabled = self.translation_thinking_checkbox.isChecked()
        settings.translation.reasoning_effort = self.translation_reasoning_effort_combo.currentData() or "high"
        settings.translation.temperature = self.translation_temperature_spin.value()
        settings.translation.chunk_size = self.translation_chunk_size_spin.value()
        settings.translation.openai_base_url = self.translation_openai_base_input.text().strip() or "https://api.openai.com/v1"
        settings.translation.openai_api_key = self.translation_openai_key_input.text().strip()

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

    def on_theme_changed(self) -> None:
        self.apply_style(self.theme_combo.currentData() or "light")

    def on_transcription_provider_changed(self) -> None:
        self.on_intelligent_segmentation_changed()

    def on_translation_mode_changed(self) -> None:
        if not hasattr(self, "translation_mode_buttons"):
            return
        mode = self.translation_mode_combo.currentData()
        enable_translation = mode != "none"
        use_openai = mode == "openai"

        self._sync_choice_buttons(self.translation_mode_combo, self.translation_mode_buttons)
        if not hasattr(self, "translation_common_cards"):
            return
        self._set_widgets_visible(self.translation_common_cards, enable_translation)
        self._set_widgets_visible(self.translation_openai_widgets, use_openai)

        self.translation_target_input.setEnabled(enable_translation)
        self.translation_model_input.setEnabled(enable_translation)
        self.translation_bilingual_checkbox.setEnabled(enable_translation)
        self.translation_keep_original_checkbox.setEnabled(enable_translation)
        self.allow_subtitle_import_checkbox.setEnabled(enable_translation)
        self.subtitle_translation_thread_spin.setEnabled(enable_translation)
        self.translation_thinking_checkbox.setEnabled(enable_translation)
        self.translation_openai_base_input.setEnabled(use_openai)
        self.translation_openai_key_input.setEnabled(use_openai)
        self.show_translation_openai_key_checkbox.setEnabled(use_openai)

        self.on_translation_thinking_changed()
        self.on_transcription_provider_changed()

    def on_translation_thinking_changed(self) -> None:
        if not hasattr(self, "translation_reasoning_effort_combo"):
            return
        enable_translation = self.translation_mode_combo.currentData() != "none"
        thinking_enabled = self.translation_thinking_checkbox.isChecked()
        self.translation_reasoning_effort_combo.setEnabled(enable_translation and thinking_enabled)
        self.translation_reasoning_effort_label.setEnabled(enable_translation and thinking_enabled)
        self.translation_temperature_spin.setEnabled(enable_translation and not thinking_enabled)
        self.translation_temperature_label.setEnabled(enable_translation and not thinking_enabled)
        self.translation_chunk_size_spin.setEnabled(enable_translation)
        self.translation_chunk_size_label.setEnabled(enable_translation)

    def on_segmentation_thinking_changed(self) -> None:
        if not hasattr(self, "segmentation_reasoning_effort_combo"):
            return
        eligible = (
            self.enable_segmentation_checkbox.isChecked()
            and self.timestamp_combo.currentText().strip() == "word"
        )
        thinking_enabled = self.segmentation_thinking_checkbox.isChecked()
        self.segmentation_reasoning_effort_combo.setEnabled(eligible and thinking_enabled)
        self.segmentation_reasoning_effort_label.setEnabled(eligible and thinking_enabled)
        self.segmentation_temperature_spin.setEnabled(eligible and not thinking_enabled)
        self.segmentation_temperature_label.setEnabled(eligible and not thinking_enabled)

    def on_language_mode_changed(self) -> None:
        if not hasattr(self, "language_input"):
            return
        manual = self.language_mode_combo.currentIndex() == 1
        self.language_input.setEnabled(manual)
        if manual:
            self.language_input.setPlaceholderText("语言代码，例如 zh / en")
        else:
            self.language_input.setPlaceholderText("自动识别时无需填写")

    def on_timestamp_granularity_changed(self) -> None:
        self.on_intelligent_segmentation_changed()

    def on_intelligent_segmentation_changed(self) -> None:
        if not hasattr(self, "segmentation_config_widgets"):
            return
        enabled = self.enable_segmentation_checkbox.isChecked()
        timestamp_granularity = self.timestamp_combo.currentText().strip()
        eligible = timestamp_granularity == "word"

        if timestamp_granularity != "word":
            hint = "智能分段仅在时间戳粒度为 word 时生效。"
        else:
            hint = "启用后将调用专用 OpenAI 兼容 API 做语义分段，并以新分段继续翻译和写出。"

        self.segmentation_hint_label.setText(hint)
        self.enable_segmentation_checkbox.setToolTip(hint)
        self._set_widgets_visible(self.segmentation_config_widgets, enabled)
        for widget in self.segmentation_config_widgets:
            widget.setEnabled(enabled and eligible)
        self.on_segmentation_thinking_changed()

    def on_output_mode_changed(self) -> None:
        if not hasattr(self, "output_mode_buttons"):
            return
        self._sync_choice_buttons(self.output_mode_combo, self.output_mode_buttons)
        custom = self.output_mode_combo.currentData() == "custom"
        self.output_dir_input.setEnabled(custom)
        self.output_btn.setEnabled(custom)
        self.output_dir_label.setEnabled(custom)

    def on_vad_enabled_changed(self) -> None:
        if not hasattr(self, "vad_controls"):
            return
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
            status_item = QTableWidgetItem(f"● {STATUS_LABELS.get('Queued', '排队中')}")
            status_item.setForeground(QColor(STATUS_COLORS.get("Queued", "#6b7280")))
            self.task_table.setItem(row, 1, status_item)
            progress_bar = QProgressBar()
            progress_bar.setObjectName("tableProgress")
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setTextVisible(False)
            self.task_table.setCellWidget(row, 2, progress_bar)
            self.task_table.setItem(row, 3, QTableWidgetItem("-"))
            self.task_table.setItem(row, 4, QTableWidgetItem("-"))
            self.task_table.setItem(row, 5, QTableWidgetItem("就绪"))
            added += 1

        self.log(f"已添加 {added} 个文件")
        if skipped_subtitle > 0 and not allow_subtitle_import:
            self.log(f"已忽略 {skipped_subtitle} 个字幕文件（导入开关已关闭）")
        self._update_summary()

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
            QMessageBox.warning(
                self,
                "缺少 ffmpeg",
                "视频任务或 VAD 预切分需要 ffmpeg。当前运行环境未检测到内置或可用的 ffmpeg。",
            )
            return
        if has_subtitle and settings.translation.mode == "none":
            QMessageBox.warning(self, "翻译未启用", "导入字幕任务需要启用翻译模式")
            return
        if has_subtitle and not settings.translation.allow_subtitle_import:
            QMessageBox.warning(self, "字幕导入已关闭", "请在设置中开启“允许导入字幕文件并翻译”")
            return

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
        if settings.segmentation.enabled:
            self.log("智能分段：已启用（专用 OpenAI 兼容 API，仅 word 时间戳生效）")
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

    def _build_context_menu(self, menu: QMenu) -> None:
        rows = {idx.row() for idx in self.task_table.selectionModel().selectedRows()}
        if not rows:
            return

        has_running = has_paused = has_failed = False
        for row in rows:
            tid = self.qm.get_task_id_by_row(row)
            if not tid:
                continue
            status = self.qm.tasks[tid].status
            if status in {"Preparing", "Extracting", "Transcribing", "Translating", "Writing", "Queued"}:
                has_running = True
            if status == "Paused":
                has_paused = True
            if status in {"Failed", "Cancelled"}:
                has_failed = True

        if has_running:
            menu.addAction("取消所选任务", self._cancel_selected)
            menu.addAction("⏸ 暂停所选任务", self._pause_selected)
        if has_paused:
            menu.addAction("▶ 恢复所选任务", self._resume_selected)
        if has_failed:
            menu.addAction("↻ 重试所选任务", self._retry_selected)

        menu.addSeparator()
        selected_tids = self.qm.get_selected_task_ids(list(rows))
        menu.addAction("上移优先级", lambda: self.qm.move_priority(selected_tids, +1))
        menu.addAction("下移优先级", lambda: self.qm.move_priority(selected_tids, -1))

        if not self.qm.is_running:
            menu.addSeparator()
            menu.addAction("删除所选", self.on_remove_selected)

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

    def _update_task_row(self, task_id: str, status: str, progress: int, message: str, outputs: str = "-") -> None:
        state = self.qm.tasks.get(task_id)
        if not state:
            return
        row = state.row
        if row >= self.task_table.rowCount():
            return

        status_item = self.task_table.item(row, 1)
        if status_item:
            status_item.setText(f"● {STATUS_LABELS.get(status, status)}")
            status_item.setForeground(QColor(STATUS_COLORS.get(status, "#6b7280")))

        progress_bar = self.task_table.cellWidget(row, 2)
        if isinstance(progress_bar, QProgressBar):
            progress_bar.setValue(progress)

        if state.start_time > 0:
            end = state.end_time if state.end_time > 0 else time.monotonic()
            elapsed_item = self.task_table.item(row, 3)
            if elapsed_item:
                elapsed_item.setText(self._format_duration(end - state.start_time))

        out_item = self.task_table.item(row, 4)
        if out_item:
            out_item.setText(outputs)

        msg_item = self.task_table.item(row, 5)
        if msg_item:
            display = message if len(message) <= 80 else message[:80] + "..."
            msg_item.setText(display)
            msg_item.setToolTip(message if len(message) > 80 else "")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds <= 0:
            return "-"
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours > 0 else f"{minutes:02d}:{secs:02d}"

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

    def _update_summary(self) -> None:
        summary = self.qm.get_summary()
        total = summary["total"]
        if total == 0:
            self.summary_label.setText("暂无任务")
            self._set_summary_state("idle")
            return

        if self.qm.is_running:
            done_batch = summary["done"] + summary["failed"] + summary["canceled"]
            self.summary_label.setText(
                f"当前批次：已完成 {done_batch}/{total} | 运行中={summary['running']} | 暂停={summary['paused']}"
            )
            self._set_summary_state("running")
            return

        parts = [f"总数={total}", f"排队={summary['queued']}", f"完成={summary['done']}"]
        if summary["failed"] > 0:
            parts.append(f"失败={summary['failed']}")
        if summary["canceled"] > 0:
            parts.append(f"取消={summary['canceled']}")
        self.summary_label.setText(" | ".join(parts))

        if summary["failed"] > 0:
            self._set_summary_state("error")
        elif summary["done"] > 0 and summary["queued"] == 0:
            self._set_summary_state("done")
        else:
            self._set_summary_state("idle")

    def _update_button_states(self) -> None:
        running = self.qm.is_running
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.pause_btn.setEnabled(running)
        self.resume_btn.setEnabled(running)
        self.remove_btn.setEnabled(not running)
        self.clear_btn.setEnabled(not running)
        self.retry_btn.setEnabled(not running and bool(self.qm.get_failed_task_ids()))

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

    def _validate_settings(self) -> AppSettings:
        settings = self.collect_settings_from_ui()

        if settings.transcription.language_mode == "manual":
            if not settings.transcription.language:
                raise RuntimeError("已选择指定语言，请填写有效语言代码，例如 zh / en")
            if not is_valid_language_code(settings.transcription.language):
                raise RuntimeError("指定语言格式无效，请使用 2-3 位语言代码，例如 zh / en / ja")

        if settings.translation.mode != "none":
            if not settings.translation.target_language:
                raise RuntimeError("请填写目标语言代码，例如 zh / en / ja")
            if not is_valid_language_code(settings.translation.target_language):
                raise RuntimeError("目标语言格式无效，请使用 2-3 位语言代码，例如 zh / en / ja")
            if not settings.translation.model:
                raise RuntimeError("请填写翻译模型名称")

        if settings.translation.mode == "openai":
            if not settings.translation.openai_api_key:
                raise RuntimeError("OpenAI 兼容翻译模式需要填写 API Key")
            if not is_valid_http_url(settings.translation.openai_base_url):
                raise RuntimeError("OpenAI 兼容翻译模式的 Base URL 无效，请填写 http/https 地址")

        if settings.transcription.provider == "whisper_openai_compatible":
            if not settings.transcription.whisper.api_key:
                raise RuntimeError("Whisper 转写需要填写第三方/OpenAI 兼容 API Key")
            if not settings.transcription.whisper.model:
                raise RuntimeError("Whisper 转写需要填写模型名称")
            if not is_valid_http_url(settings.transcription.whisper.base_url):
                raise RuntimeError("Whisper Base URL 无效，请填写 http/https 地址")

        if settings.segmentation.enabled:
            if settings.transcription.timestamp_granularity != "word":
                raise RuntimeError("启用智能分段时，时间戳粒度必须为 word")
            if not settings.segmentation.openai_base_url:
                raise RuntimeError("请填写智能分段专用 API 的 Base URL")
            if not is_valid_http_url(settings.segmentation.openai_base_url):
                raise RuntimeError("智能分段 Base URL 无效，请填写 http/https 地址")
            if not settings.segmentation.openai_api_key:
                raise RuntimeError("请填写智能分段专用 API 的 API Key")
            if not settings.segmentation.model:
                raise RuntimeError("请填写智能分段模型名称")

        if not (settings.output.save_srt or settings.output.save_lrc or settings.output.save_txt or settings.output.save_json):
            raise RuntimeError("请至少选择一种输出格式")

        if settings.output.mode == "custom":
            try:
                settings.output.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                raise RuntimeError(f"输出目录不可用：{exc}") from exc

        return settings
