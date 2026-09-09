import json
from pathlib import Path
from typing import Dict, Any, Optional


def get_project_root() -> Path:
    """Finds the project root directory based on landmark files (.git, config.json, pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "config.json").exists() or (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def resolve_input_path(filename: str, default_subfolder: str = "raw") -> Path:
    """
    Resolves an input file path intelligently:
    1. Exact path if absolute or relative to cwd.
    2. data/<default_subfolder>/<filename>
    3. data/raw/<filename> or data/processed/<filename>
    4. Project root / <filename>
    """
    path = Path(filename)
    if path.is_absolute() and path.exists():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()

    root = get_project_root()

    subfolder_path = root / "data" / default_subfolder / path.name
    if subfolder_path.exists():
        return subfolder_path

    for folder in ("raw", "processed"):
        candidate = root / "data" / folder / path.name
        if candidate.exists():
            return candidate

    root_candidate = root / path.name
    if root_candidate.exists():
        return root_candidate

    # If file doesn't exist yet, return preferred path under data/<default_subfolder> or root
    target_dir = root / "data" / default_subfolder
    if target_dir.exists():
        return target_dir / path.name
    return root / path.name


def resolve_output_path(filename: str, default_subfolder: str = "processed") -> Path:
    """
    Resolves an output path, directing outputs to data/<default_subfolder> by default.
    """
    path = Path(filename)
    if path.is_absolute():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    root = get_project_root()
    target_dir = root / "data" / default_subfolder
    if target_dir.exists():
        return target_dir / path.name

    return (root / path.name).resolve()


def load_config(config_file: str = "config.json", section: Optional[str] = None) -> Dict[str, Any]:
    """Loads configuration settings from a JSON file and applies defaults."""
    root = get_project_root()
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = root / config_path

    raw_config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read '{config_path.name}' ({e}). Using defaults.")
    else:
        print(f"Notice: Config file '{config_file}' not found. Using defaults.")

    config = {k: v for k, v in raw_config.items() if not isinstance(v, dict)}

    if section:
        section_aliases = {
            "cleanup": ["cleanup", "clean"],
            "translation": ["translation", "translate"],
            "validation": ["validation", "validate"]
        }
        matched_section = None
        for key in section_aliases.get(section, [section]):
            if key in raw_config and isinstance(raw_config[key], dict):
                matched_section = raw_config[key]
                break

        if matched_section:
            config.update(matched_section)
        elif not any(k in raw_config for k in ("cleanup", "translation", "translate", "validation", "validate")):
            config.update(raw_config)

    # Global defaults
    config.setdefault("api_endpoint", "http://127.0.0.1:1234/v1/chat/completions")
    config.setdefault("api_key", "lm-studio")
    config.setdefault("request_timeout", 60)

    return config
