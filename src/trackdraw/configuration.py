from __future__ import annotations

import os
from typing import Dict, List

import yaml

from .models import BackgroundSpec, LocationConfig, RuleSettings


DEFAULT_RULES_CONFIG = "config/validation_rules.yaml"


def load_track_defaults(config_path: str) -> Dict[str, float]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return {
        "track_width": float(config.get("track_width", 3.0)),
        "cone_distance": float(config.get("cone_distance", 3.5)),
        "min_boundary_backoff": float(config.get("min_boundary_backoff", 10.0)),
        "n_points_midline": int(config.get("n_points_midline", 300)),
        "standard_location": str(config.get("standard_location", "empty")),
    }


def discover_location_configs(base_dir: str) -> List[LocationConfig]:
    locations: List[LocationConfig] = []
    if not os.path.isdir(base_dir):
        return locations

    for entry in sorted(os.listdir(base_dir)):
        folder = os.path.join(base_dir, entry)
        if not os.path.isdir(folder):
            continue
        config_path = os.path.join(folder, f"{entry}_config.yaml")
        if not os.path.isfile(config_path):
            continue
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        sat_img_path = os.path.join(folder, str(data.get("sat_img_path", "")))
        locations.append(
            LocationConfig(
                name=entry,
                px_per_m=float(data.get("px_per_m", 10.0)),
                sat_img_path=sat_img_path,
                config_path=config_path,
            )
        )
    return locations


def default_background(default_location: str, locations: List[LocationConfig]) -> BackgroundSpec:
    location_names = {item.name for item in locations}
    if default_location in location_names:
        location = next(item for item in locations if item.name == default_location)
        return BackgroundSpec(kind="location", location_name=location.name, px_per_m=location.px_per_m)
    if locations:
        location = locations[0]
        return BackgroundSpec(kind="location", location_name=location.name, px_per_m=location.px_per_m)
    return BackgroundSpec(kind="grid", location_name="", px_per_m=10.0)


def load_rule_settings(config_path: str) -> Dict[str, RuleSettings]:
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    rules_data = data.get("rules", {})
    rules: Dict[str, RuleSettings] = {}
    for rule_id, values in rules_data.items():
        rules[rule_id] = RuleSettings(
            enabled=bool(values.get("enabled", True)),
            threshold=float(values["threshold"]) if values.get("threshold") is not None else None,
            severity=str(values.get("severity", "error")),
        )
    return rules


def dump_rule_settings(rule_settings: Dict[str, RuleSettings]) -> Dict[str, Dict[str, object]]:
    return {
        rule_id: {
            "enabled": settings.enabled,
            "threshold": settings.threshold,
            "severity": settings.severity,
        }
        for rule_id, settings in rule_settings.items()
    }
