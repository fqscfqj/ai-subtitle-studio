from __future__ import annotations

import unittest

from subtitle_studio.theme import ThemeManager


class ThemeManagerTests(unittest.TestCase):
    def test_stylesheet_contains_core_selectors(self) -> None:
        stylesheet = ThemeManager.stylesheet("light")
        self.assertIn("QMainWindow#mainWindow", stylesheet)
        self.assertIn("QListWidget#settingsNav", stylesheet)
        self.assertIn("QPushButton#optionCardButton", stylesheet)
        self.assertIn("QFrame#dropFrame[dragActive=\"true\"]", stylesheet)

    def test_dark_palette_differs_from_light(self) -> None:
        light = ThemeManager.palette("light")
        dark = ThemeManager.palette("dark")
        self.assertNotEqual(light.window_bg, dark.window_bg)
        self.assertNotEqual(light.card_bg, dark.card_bg)
        self.assertNotEqual(light.text_primary, dark.text_primary)


if __name__ == "__main__":
    unittest.main()