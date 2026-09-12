"""Configuration loading and path resolution utilities."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from .utils import load_json_file


@functools.lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Finds the project root directory based on landmark files."""
    current = Path(__file__).resolve().parent
    candidates = [current] + list(current.parents)
    for parent in candidates:
        if (
            (parent / "config.json").exists()
            or (parent / ".git").exists()
            or (parent / "pyproject.toml").exists()
        ):
            return parent
    return Path.cwd()


def resolve_input_path(filename: str | Path, default_subfolder: str = "raw") -> Path:
    """Resolves an input file path across standard search locations."""
    path = Path(filename)
    if path.is_absolute() and path.exists():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()

    root = get_project_root()
    search_paths = (
        root / "data" / default_subfolder / path.name,
        root / "data" / "reference" / path.name,
        root / "data" / "dictionaries" / path.name,
        root / "data" / "raw" / path.name,
        root / "data" / "processed" / path.name,
        root / path.name,
    )

    for candidate in search_paths:
        if candidate.exists():
            return candidate

    target_dir = root / "data" / default_subfolder
    if target_dir.exists():
        return target_dir / path.name
    return root / path.name


def resolve_output_path(filename: str | Path, default_subfolder: str = "processed") -> Path:
    """Resolves an output path, directing outputs to data/<default_subfolder> by default."""
    path = Path(filename)
    if path.is_absolute():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    root = get_project_root()
    target_dir = root / "data" / default_subfolder
    if target_dir.exists():
        return target_dir / path.name

    return (root / path.name).resolve()


def load_config(
    config_file: str | Path = "config.json", section: str | None = None
) -> dict[str, Any]:
    """Loads configuration settings from a JSON file and applies defaults."""
    root = get_project_root()
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = root / config_path

    raw_config: dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = load_json_file(config_path)
            if isinstance(loaded, dict):
                raw_config = loaded
        except (OSError, ValueError, TypeError, KeyError) as err:
            print(f"Warning: Failed to read '{config_path.name}' ({err}). Using defaults.")
    else:
        print(f"Notice: Config file '{config_file}' not found. Using defaults.")

    config = {k: v for k, v in raw_config.items() if not isinstance(v, dict)}

    if section:
        section_aliases = {
            "cleanup": ("cleanup", "clean"),
            "translation": ("translation", "translate"),
            "validation": ("validation", "validate"),
        }
        matched_section = None
        for key in section_aliases.get(section, (section,)):
            val = raw_config.get(key)
            if isinstance(val, dict):
                matched_section = val
                break

        pipeline_keys = (
            "cleanup",
            "translation",
            "translate",
            "validation",
            "validate",
        )
        if matched_section:
            config.update(matched_section)
        elif not any(k in raw_config for k in pipeline_keys):
            config.update(raw_config)

    config.setdefault("api_endpoint", "http://127.0.0.1:1234/v1/chat/completions")
    config.setdefault("api_key", "lm-studio")
    config.setdefault("request_timeout", 60)

    return config
