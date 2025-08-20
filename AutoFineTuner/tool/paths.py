# 반드시 파일 최상단
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 프로젝트 산출물 경로
@dataclass
class ProjPaths:
    proj_root: Path
    target: Path
    refactored: Path

def get_proj_paths(
    proj_root: str | Path,
    target: str,
    refactored: str = "refactored.py",
) -> ProjPaths:
    root = Path(proj_root)
    return ProjPaths(
        proj_root=root,
        target=root / target,
        refactored=root / refactored,
    )

# 출력(결과) 경로
@dataclass
class OutputPaths:
    save_root: Path
    metadata: Path
    results: Path

def get_output_paths(save_dir: str | Path, run_id: Optional[str] = None) -> OutputPaths:
    root = Path(save_dir)
    return OutputPaths(
        save_root=root,
        metadata=root / "metadata.json",
        results=root / "results.json",
    )
