"""
Shared utilities for adversarial validation pytest modules.

Design goals:
- Deterministic seeds and deterministic artifact filenames
- Artifacts written under Quantum-Eye/artifacts/
- Lightweight version context (python, key package versions, git SHA if available)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# Fixed seed list committed in code for repeatability across runs/machines.
SEEDS: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def _project_root() -> Path:
    # tests/ -> project root is one level up from tests/
    return Path(__file__).resolve().parents[1]


def artifacts_root() -> Path:
    root = _project_root() / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_module_run_id(config: Dict[str, Any]) -> str:
    """
    Deterministic, content-addressed identifier for a module run.
    """
    payload = _json_dumps_stable(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def artifact_path(module_name: str, filename: str) -> Path:
    """
    Deterministic artifact location for a module.
    """
    safe_module = module_name.replace("\\", "_").replace("/", "_")
    out_dir = artifacts_root() / safe_module
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_json_dumps_stable(data) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _try_git_sha(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        sha = result.stdout.strip()
        return sha or None
    except Exception:
        return None


def _pkg_version(dist_name: str) -> Optional[str]:
    try:
        # Python 3.10+
        from importlib import metadata as importlib_metadata  # type: ignore

        return importlib_metadata.version(dist_name)
    except Exception:
        return None


def get_version_context(extra_packages: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Version context for reproducibility.
    """
    repo_root = _project_root()
    packages = [
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "qiskit",
        "qiskit-aer",
        "pytest",
    ]
    if extra_packages:
        packages.extend(extra_packages)

    pkg_versions = {name: _pkg_version(name) for name in sorted(set(packages))}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": pkg_versions,
        "git": {"sha": _try_git_sha(repo_root)},
    }


@dataclass(frozen=True)
class ModuleRunContext:
    module_name: str
    config: Dict[str, Any]

    @property
    def run_id(self) -> str:
        return stable_module_run_id(self.config)

    def base_payload(self) -> Dict[str, Any]:
        return {
            "module": self.module_name,
            "config": self.config,
            "run_id": self.run_id,
            "seeds": SEEDS,
            "version_context": get_version_context(),
        }



