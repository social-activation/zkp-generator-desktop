# setup.py
from __future__ import annotations
import subprocess
from pathlib import Path
import shutil
import sys
import json
import os

# ---------- Path helpers ----------

def _exe_dir() -> Path:
    # Where the frozen executable lives (inside .app/Contents/MacOS on macOS)
    return Path(getattr(sys, "executable", __file__)).resolve().parent

def _meipass_dir() -> Path:
    # Where PyInstaller extracts bundled data (--add-data); may equal _exe_dir in dev
    return Path(getattr(sys, "_MEIPASS", _exe_dir()))

def _resources_dir() -> Path:
    # On macOS `.app` structure: .../Contents/Resources
    ed = _exe_dir()
    return ed.parent.parent / "Resources"

def search_dirs() -> list[Path]:
    # Order matters: user's CWD first, then next to exe, then bundled data, then Resources
    return [Path.cwd().resolve(), _exe_dir(), _meipass_dir(), _resources_dir()]

def find_file(name: str) -> Path | None:
    for base in search_dirs():
        candidate = base / name
        if candidate.exists():
            return candidate.resolve()
    return None

def ensure_writable_copy(src: Path, dest_dir: Path) -> Path:
    """
    If src is inside a read-only bundle (_MEIPASS), copy it to dest_dir and return the copy.
    Otherwise just return src.
    """
    try:
        # crude check: treat paths under _MEIPASS as read-only
        if str(_meipass_dir()) in str(src):
            dest_dir.mkdir(parents=True, exist_ok=True)
            dst = dest_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            return dst
    except Exception:
        pass
    return src

# ---------- CLI helpers ----------

def _have_cli() -> bool:
    return shutil.which("ezkl") is not None

def _run(cmd: list[str], cwd: Path, allow_fail: bool = False) -> None:
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True)
    except subprocess.CalledProcessError as e:
        if allow_fail:
            return
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {e}") from e

# ---------- Public: run_installation ----------

def run_installation() -> Path:
    """
    One-time setup:
      - locate (or copy) network.onnx
      - generate settings.json if missing
      - fetch SRS if missing
      - compile circuit + setup prover
      - write installed.flag
    Returns the 'base_dir' where artifacts live (use this to read/write files).
    """
    # 0) locate the model
    model = find_file("network.onnx")
    if not model:
        raise FileNotFoundError(
            "network.onnx not found.\n"
            "Place it next to the app (or in the same folder you launch the app from)."
        )

    # If bundled in the app (_MEIPASS), copy it to a writable user dir
    # (You can change this to wherever you want your runtime assets to live)
    default_user_dir = Path.home() / "EZKLApp"
    base_dir = ensure_writable_copy(model, default_user_dir).parent

    # Make base dir and a marker file path
    base_dir.mkdir(parents=True, exist_ok=True)
    installed_flag = base_dir / ".installed"

    # If already installed, we’re done
    if installed_flag.exists():
        return base_dir

    # We need the CLI for these steps
    if not _have_cli():
        raise EnvironmentError(
            "The ezkl CLI was not found in PATH.\n"
            "Install it and re-run the app (or bundle the ezkl binary with the app)."
        )

    # 1) settings.json
    settings = base_dir / "settings.json"
    if not settings.exists():
        _run(["ezkl", "gen-settings"], cwd=base_dir)

    # 2) SRS
    srs_path = base_dir / "srs" / "kzg.srs"
    srs_path.parent.mkdir(parents=True, exist_ok=True)
    if not srs_path.exists():
        # Use explicit args first; fall back to defaults if ezkl version differs
        try:
            _run(["ezkl", "get-srs", "--srs-path", str(srs_path)], cwd=base_dir)
        except Exception:
            _run(["ezkl", "get-srs"], cwd=base_dir)

    # 3) compile circuit + 4) setup prover (idempotent)
    _run(["ezkl", "compile-circuit"], cwd=base_dir, allow_fail=True)
    _run(["ezkl", "setup"],   cwd=base_dir, allow_fail=True)

    # 5) write marker
    try:
        installed_flag.write_text(json.dumps({"installed": True}))
    except Exception:
        pass

    return base_dir
