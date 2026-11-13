"""Loads config.json once. Every script imports this instead of hardcoding settings.

Optional environment overrides (handy for one-off experiments):
    SLM_BACKEND, SLM_MODEL, SLM_BASE_URL, APPAI_CONFIG (path to another config file)
"""
import json
import os

# Run this everytime imported to get the json files
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.environ.get("APPAI_CONFIG") or os.path.join(ROOT, "config.json")

with open(CONFIG_FILE, encoding="utf-8") as _f:
    cfg = json.load(_f)
########################

def path(name):
    """Absolute path for an entry in cfg['paths']. Creates its parent folder."""
    p = cfg["paths"][name]
    full = p if os.path.isabs(p) else os.path.join(ROOT, p)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full



def model_settings(task="default"):
    m = cfg["model"]
    backend = (
        os.environ.get("SLM_BACKEND")
        or m["backend"]
    ).lower()
    base_url = (
        os.environ.get("SLM_BASE_URL")
        or m["endpoints"][backend]
    )
    model_name = (
        os.environ.get("SLM_MODEL")
        or m["models"].get(task, m["models"]["default"])
    )
    return backend, model_name, base_url.rstrip("/")