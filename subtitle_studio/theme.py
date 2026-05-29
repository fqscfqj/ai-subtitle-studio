from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class ThemePalette:
    name: str
    window_bg: str
    page_bg: str
    card_bg: str
    card_bg_alt: str
    card_hover: str
    border: str
    border_strong: str
    text_primary: str
    text_secondary: str
    text_muted: str
    accent: str
    accent_soft: str
    accent_hover: str
    accent_text: str
    success: str
    warning: str
    danger: str
    info: str
    shadow: str
    selection_bg: str
    input_bg: str
    input_disabled: str
    menu_bg: str


LIGHT_THEME = ThemePalette(
    name="light",
    window_bg="#f3f6fb",
    page_bg="#edf2f8",
    card_bg="#ffffff",
    card_bg_alt="#f8fbff",
    card_hover="#f1f6ff",
    border="#d7e0ec",
    border_strong="#bfd0e4",
    text_primary="#17212b",
    text_secondary="#334155",
    text_muted="#64748b",
    accent="#2563eb",
    accent_soft="#e8f0ff",
    accent_hover="#1d4ed8",
    accent_text="#123a94",
    success="#16a34a",
    warning="#d97706",
    danger="#dc2626",
    info="#2563eb",
    shadow="rgba(15, 23, 42, 0.06)",
    selection_bg="#dbeafe",
    input_bg="#ffffff",
    input_disabled="#eef2f7",
    menu_bg="#ffffff",
)


DARK_THEME = ThemePalette(
    name="dark",
    window_bg="#11151b",
    page_bg="#161b22",
    card_bg="#1b2330",
    card_bg_alt="#202938",
    card_hover="#243144",
    border="#313d50",
    border_strong="#42516a",
    text_primary="#ecf2f9",
    text_secondary="#c7d2e3",
    text_muted="#94a3b8",
    accent="#60a5fa",
    accent_soft="#102443",
    accent_hover="#3b82f6",
    accent_text="#bfdbfe",
    success="#4ade80",
    warning="#fbbf24",
    danger="#f87171",
    info="#60a5fa",
    shadow="rgba(0, 0, 0, 0.28)",
    selection_bg="#1d4e89",
    input_bg="#111827",
    input_disabled="#1f2937",
    menu_bg="#1b2330",
)


class ThemeManager:
    _THEMES = {
        "light": LIGHT_THEME,
        "dark": DARK_THEME,
    }

    @classmethod
    def palette(cls, theme_name: str) -> ThemePalette:
        return cls._THEMES.get(theme_name, LIGHT_THEME)

    @classmethod
    def app_font(cls) -> QFont:
        return QFont("Segoe UI", 10)

    @classmethod
    def apply_theme(cls, app: QApplication, theme_name: str) -> ThemePalette:
        palette = cls.palette(theme_name)
        app.setFont(cls.app_font())
        app.setStyleSheet(cls.stylesheet(theme_name))
        return palette

    @classmethod
    def stylesheet(cls, theme_name: str) -> str:
        p = cls.palette(theme_name)
        return f"""
        QMainWindow#mainWindow, QWidget {{
            background: {p.window_bg};
            color: {p.text_primary};
        }}

        QWidget#taskPage, QWidget#settingsPage, QWidget#settingsPanel {{
            background: {p.page_bg};
        }}

        QTabWidget::pane {{
            border: 1px solid {p.border};
            border-radius: 14px;
            background: {p.page_bg};
            top: -1px;
        }}

        QTabBar::tab {{
            background: {p.card_bg_alt};
            color: {p.text_secondary};
            border: 1px solid {p.border};
            border-bottom: none;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            padding: 10px 22px;
            margin-right: 6px;
            font-weight: 600;
            min-width: 88px;
        }}

        QTabBar::tab:selected {{
            background: {p.card_bg};
            color: {p.text_primary};
        }}

        QTabBar::tab:!selected:hover {{
            background: {p.card_hover};
        }}

        QScrollArea, QScrollArea > QWidget > QWidget {{
            background: transparent;
        }}

        QFrame#settingsHeaderCard,
        QFrame#settingsCard,
        QFrame#logCard,
        QFrame#dropFrame,
        QListWidget#settingsNav,
        QFrame#taskToolbarCard {{
            background: {p.card_bg};
            border: 1px solid {p.border};
            border-radius: 16px;
        }}

        QFrame#settingsCard:hover,
        QFrame#logCard:hover,
        QFrame#taskToolbarCard:hover {{
            border: 1px solid {p.border_strong};
        }}

        QFrame#dropFrame {{
            border: 2px dashed {p.border_strong};
            background: {p.card_bg_alt};
            min-height: 124px;
        }}

        QFrame#dropFrame[dragActive="true"] {{
            border: 2px solid {p.accent};
            background: {p.accent_soft};
        }}

        QLabel#dropIconLabel {{
            font-size: 32px;
            font-weight: 700;
            color: {p.accent};
            background: transparent;
        }}

        QLabel#dropTitleLabel {{
            font-size: 16px;
            font-weight: 700;
            color: {p.text_primary};
            background: transparent;
        }}

        QLabel#dropHintLabel,
        QLabel#heroDescription,
        QLabel#pageDescription,
        QLabel#sectionDescription,
        QLabel#mutedLabel {{
            color: {p.text_muted};
            background: transparent;
        }}

        QLabel#heroTitle {{
            font-size: 20px;
            font-weight: 700;
            color: {p.text_primary};
            background: transparent;
        }}

        QLabel#pageTitle {{
            font-size: 18px;
            font-weight: 700;
            color: {p.text_primary};
            background: transparent;
        }}

        QLabel#sectionTitle {{
            font-size: 14px;
            font-weight: 700;
            color: {p.text_primary};
            background: transparent;
        }}

        QLabel#fieldLabel {{
            color: {p.text_secondary};
            font-weight: 600;
            background: transparent;
        }}

        QListWidget#settingsNav {{
            padding: 10px;
            outline: none;
        }}

        QListWidget#settingsNav::item {{
            border-radius: 12px;
            padding: 12px 14px;
            margin: 3px 0;
            color: {p.text_secondary};
            background: transparent;
            font-weight: 600;
        }}

        QListWidget#settingsNav::item:selected {{
            background: {p.accent_soft};
            color: {p.accent_text};
            border: 1px solid {p.accent};
        }}

        QListWidget#settingsNav::item:hover:!selected {{
            background: {p.card_hover};
        }}

        QPushButton {{
            background: {p.accent};
            color: white;
            border: none;
            border-radius: 10px;
            padding: 8px 14px;
            font-weight: 600;
        }}

        QPushButton:hover {{
            background: {p.accent_hover};
        }}

        QPushButton:disabled {{
            background: {p.border};
            color: {p.text_muted};
        }}

        QPushButton#primaryBtn {{
            min-height: 36px;
            padding: 8px 18px;
        }}

        QPushButton#secondaryBtn {{
            background: {p.card_bg_alt};
            color: {p.text_secondary};
            border: 1px solid {p.border};
        }}

        QPushButton#secondaryBtn:hover {{
            background: {p.card_hover};
            border: 1px solid {p.border_strong};
        }}

        QPushButton#dangerBtn {{
            background: {p.danger};
            color: white;
        }}

        QPushButton#warningBtn {{
            background: {p.warning};
            color: white;
        }}

        QPushButton#optionCardButton {{
            text-align: left;
            padding: 14px 16px;
            min-height: 76px;
            background: {p.card_bg_alt};
            color: {p.text_primary};
            border: 1px solid {p.border};
            border-radius: 14px;
        }}

        QPushButton#optionCardButton:hover {{
            background: {p.card_hover};
            border: 1px solid {p.border_strong};
        }}

        QPushButton#optionCardButton:checked {{
            background: {p.accent_soft};
            color: {p.accent_text};
            border: 1px solid {p.accent};
        }}

        QPushButton#modePillButton {{
            background: {p.card_bg_alt};
            color: {p.text_secondary};
            border: 1px solid {p.border};
            border-radius: 10px;
            padding: 8px 12px;
        }}

        QPushButton#modePillButton:checked {{
            background: {p.accent_soft};
            color: {p.accent_text};
            border: 1px solid {p.accent};
        }}

        QLabel, QCheckBox {{
            background: transparent;
        }}

        QCheckBox {{
            color: {p.text_secondary};
            spacing: 8px;
            padding: 3px 2px;
        }}

        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 1px solid {p.border_strong};
            border-radius: 6px;
            background: {p.input_bg};
        }}

        QCheckBox::indicator:checked {{
            background: {p.accent};
            border: 1px solid {p.accent};
        }}

        QCheckBox::indicator:disabled {{
            background: {p.input_disabled};
            border: 1px solid {p.border};
        }}

        QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            border: 1px solid {p.border};
            border-radius: 10px;
            padding: 8px 10px;
            background: {p.input_bg};
            color: {p.text_primary};
            selection-background-color: {p.selection_bg};
        }}

        QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {p.accent};
        }}

        QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            background: {p.input_disabled};
            color: {p.text_muted};
        }}

        QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            width: 22px;
            border-left: 1px solid {p.border};
            background: {p.card_bg_alt};
            border-top-right-radius: 10px;
            border-bottom-right-radius: 10px;
        }}

        QComboBox QAbstractItemView {{
            background: {p.menu_bg};
            border: 1px solid {p.border};
            color: {p.text_primary};
            selection-background-color: {p.selection_bg};
            selection-color: {p.text_primary};
        }}

        QTableWidget {{
            border: 1px solid {p.border};
            border-radius: 16px;
            background: {p.card_bg};
            alternate-background-color: {p.card_bg_alt};
            gridline-color: {p.border};
        }}

        QTableWidget::item {{
            padding: 8px 8px;
        }}

        QHeaderView::section {{
            background: {p.card_bg_alt};
            color: {p.text_secondary};
            font-weight: 700;
            border: none;
            border-right: 1px solid {p.border};
            padding: 10px 8px;
        }}

        QProgressBar {{
            border: 1px solid {p.border};
            border-radius: 7px;
            background: {p.card_bg_alt};
            text-align: center;
            color: {p.text_secondary};
            font-size: 11px;
        }}

        QProgressBar#totalProgress {{
            min-height: 8px;
            max-height: 8px;
            border: none;
            background: {p.border};
        }}

        QProgressBar#tableProgress {{
            min-height: 8px;
            max-height: 8px;
            border: none;
            background: {p.border};
        }}

        QProgressBar::chunk {{
            background: {p.accent};
            border-radius: 6px;
        }}

        QLabel#summaryLabel {{
            font-weight: 700;
            font-size: 13px;
        }}

        QLabel#summaryLabel[summaryState="idle"] {{
            color: {p.text_muted};
        }}

        QLabel#summaryLabel[summaryState="running"] {{
            color: {p.info};
        }}

        QLabel#summaryLabel[summaryState="error"] {{
            color: {p.danger};
        }}

        QLabel#summaryLabel[summaryState="done"] {{
            color: {p.success};
        }}

        QMenu {{
            background: {p.menu_bg};
            color: {p.text_primary};
            border: 1px solid {p.border};
            border-radius: 10px;
            padding: 4px 0;
        }}

        QMenu::item {{
            padding: 7px 24px 7px 12px;
        }}

        QMenu::item:selected {{
            background: {p.selection_bg};
        }}
        """
