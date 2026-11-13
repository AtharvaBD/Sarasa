"""Starts the Ollama server and pulls the model if needed. Stdlib only.

Standalone (from the project root):  python -m startup.ollama_setup
Uses the model and URL from config.json (or the SLM_* environment variables).
"""
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import config


class SetupError(RuntimeError):
    """Couldn't start the server or get the model."""


def _server_up(base_url):
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        return True
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def _installed_models(base_url):
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=10) as r:
        return [m["name"] for m in json.loads(r.read()).get("models", [])]


def ensure_ready(model, base_url, wait_seconds=30):
    """Start `ollama serve` if it isn't running, then pull `model` if it's missing."""
    # 1. server
    if not _server_up(base_url):
        if not shutil.which("ollama"):
            raise SetupError("ollama isn't installed or not on PATH")
        print("Starting ollama serve...")
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(wait_seconds):
            time.sleep(1)
            if _server_up(base_url):
                break
        else:
            raise SetupError(f"ollama didn't come up within {wait_seconds}s")

    # 2. model ("qwen3" with no tag matches "qwen3:latest")
    want = model if ":" in model else model + ":latest"
    if want not in _installed_models(base_url):
        if not shutil.which("ollama"):
            raise SetupError(f"model {model} is missing and the ollama CLI isn't on PATH to pull it")
        print(f"Pulling {model} (one-time download, may take a few minutes)...")
        if subprocess.run(["ollama", "pull", model]).returncode != 0:
            raise SetupError(f"'ollama pull {model}' failed")


if __name__ == "__main__":
    _backend, _model, _url = config.model_settings()
    ensure_ready(_model, _url)
    print(f"Ollama is ready ({_model}).")