# codeLauncher.py
from __future__ import annotations
import os, subprocess, time, shlex, pathlib, sys
from typing import Dict, List, Optional
from pathlib import Path

def _is_windows() -> bool:
    return os.name == "nt"

def _python_from_venv(venv_path: str | Path) -> str:
    """venv 루트에서 파이썬 실행 파일 경로 추론"""
    v = Path(venv_path)
    if _is_windows():
        cand = v / "Scripts" / "python.exe"
    else:
        cand = v / "bin" / "python"
    if not cand.exists():
        raise FileNotFoundError(f"가상환경의 python을 찾을 수 없습니다: {cand}")
    return str(cand)

def _auto_find_project_python(cwd: Path) -> Optional[str]:
    """
    프로젝트 루트 기준으로 흔한 위치의 venv를 자동 탐색.
    - .venv/, venv/ 우선 탐색
    - 성공 시 해당 python 경로 반환, 실패 시 None
    """
    for name in [".venv", "venv"]:
        vpath = cwd / name
        try:
            return _python_from_venv(vpath)
        except Exception:
            pass
    return None

def run_python(
    pyfile: str,
    args: Optional[List[str]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    log_dir: str = "outputs",
    *,
    python_exec: Optional[str] = None,   # ① 직접 지정
    venv_path: Optional[str] = None,     # ② venv 루트 지정
    conda_env: Optional[str] = None,     # ③ conda 환경 이름
    conda_exec: Optional[str] = None     # (옵션) conda 바이너리 경로 (기본: PATH 탐색)
):
    """
    pyfile: 실행할 파이썬 파일 경로
    args:   ["--epochs=3", "--lr=1e-3"] 형태의 인자 리스트
    env_overrides: 환경변수 오버라이드(dict)
    cwd:    작업 디렉터리(없으면 현재 디렉터리)
    log_dir: 실행 로그 저장 상위 디렉터리

    python_exec/venv_path/conda_env 중 하나로 실행 인터프리터를 선택:
      - python_exec: /path/to/python
      - venv_path:   /path/to/venv (bin/python 또는 Scripts/python.exe를 추론)
      - conda_env:   conda run -n <envname> python
      - 모두 None이면:
          1) cwd 기준 .venv/venv 자동 탐색
          2) 실패 시 sys.executable (현재 인터프리터)
    """
    pyfile_path = str(Path(pyfile))
    args = args or []

    # 작업 디렉터리 Path 객체
    workdir = Path(cwd).resolve() if cwd else Path.cwd()

    # 어떤 파이썬으로 실행할지 결정
    if python_exec:
        cmd = [python_exec, pyfile_path]
    elif venv_path:
        cmd = [_python_from_venv(venv_path), pyfile_path]
    elif conda_env:
        conda_bin = conda_exec or "conda"  # conda 경로를 모르면 PATH에서 탐색
        cmd = [conda_bin, "run", "-n", conda_env, "python", pyfile_path]
    else:
        auto = _auto_find_project_python(workdir)
        if auto:
            cmd = [auto, pyfile_path]
        else:
            # 현재 프로세스의 인터프리터 사용(가장 마지막 fallback)
            cmd = [sys.executable, pyfile_path]

    # 인자 추가
    cmd += args

    # 환경변수 구성
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    # 실행로그용 파일 생성
    ts = time.strftime("%Y%m%d%H%M%S")
    run_dir = Path(log_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    # 실행 & 로깅
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# CMD: {' '.join(map(shlex.quote, cmd))}\n")
        logf.write(f"# CWD: {str(workdir)}\n")
        logf.write(f"# PYEXEC: {cmd[0]}\n\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(workdir),
            env=env,
        )
        ret = proc.wait()

    if ret != 0:
        raise RuntimeError(f"Process failed with code {ret}. See {log_path}")
    return str(run_dir)
