"""Loads config.json once. Set JOB_CONFIG=/path/to/other.json to use a different file."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = Path(os.environ.get("JOB_CONFIG", ROOT / "configpoll.json"))

with open(CONFIG_FILE, encoding="utf-8") as f:
    cfg = json.load(f)


def path(name):
    """Resolve cfg['paths'][name] relative to the project root; create the parent folder."""
    p = Path(cfg["paths"][name])
    p = p if p.is_absolute() else ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def model_settings():
    """(backend, model, base_url). Env vars SLM_BACKEND / SLM_MODEL / SLM_BASE_URL win."""
    m = cfg["model"]
    return (
        os.environ.get("SLM_BACKEND", m["backend"]),
        os.environ.get("SLM_MODEL", m["model"]),
        os.environ.get("SLM_BASE_URL", m["base_url"]),
    )


def prompt(name):
    """A prompt stored as a list of lines (or a plain string) in cfg['prompts']."""
    v = cfg["prompts"][name]
    return "\n".join(v) if isinstance(v, list) else v