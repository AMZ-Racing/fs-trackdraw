from __future__ import annotations

from typing import Dict

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .models import RuleSettings
from .validation import RULE_DEFINITIONS


class RulesEditorDialog(QDialog):
    def __init__(self, parent, rules: Dict[str, RuleSettings]):
        super().__init__(parent)
        self.setWindowTitle("Validation Rules")
        self.resize(720, 420)
        self._row_widgets = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Adjust enabled state, threshold, and severity for each geometry rule."))

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)
        grid.addWidget(QLabel("Rule"), 0, 0)
        grid.addWidget(QLabel("Enabled"), 0, 1)
        grid.addWidget(QLabel("Threshold"), 0, 2)
        grid.addWidget(QLabel("Severity"), 0, 3)

        for row, (rule_id, definition) in enumerate(RULE_DEFINITIONS.items(), start=1):
            current = rules[rule_id]
            enabled = QCheckBox()
            enabled.setChecked(current.enabled)

            if definition.threshold_label is None:
                threshold_widget = QLabel("-")
                threshold_edit = None
            else:
                threshold_edit = QLineEdit()
                threshold_value = current.threshold if current.threshold is not None else definition.default_threshold
                threshold_edit.setText(f"{float(threshold_value):.3f}")
                threshold_widget = threshold_edit

            severity = QComboBox()
            severity.addItems(["error", "warning"])
            severity.setCurrentText(current.severity)

            grid.addWidget(QLabel(definition.label), row, 0)
            grid.addWidget(enabled, row, 1)
            grid.addWidget(threshold_widget, row, 2)
            grid.addWidget(severity, row, 3)
            self._row_widgets[rule_id] = {
                "enabled": enabled,
                "threshold": threshold_edit,
                "severity": severity,
                "default_threshold": definition.default_threshold,
            }

        scroll.setWidget(content)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_rules(self) -> Dict[str, RuleSettings]:
        rules: Dict[str, RuleSettings] = {}
        for rule_id, widgets in self._row_widgets.items():
            threshold_edit = widgets["threshold"]
            threshold = widgets["default_threshold"]
            if threshold_edit is not None:
                try:
                    threshold = float(threshold_edit.text().strip())
                except ValueError:
                    threshold = widgets["default_threshold"]
                if threshold_edit is not None and threshold is not None:
                    threshold_edit.setText(f"{float(threshold):.3f}")
            rules[rule_id] = RuleSettings(
                enabled=widgets["enabled"].isChecked(),
                threshold=threshold,
                severity=widgets["severity"].currentText(),
            )
        return rules
