# codeLauncher.py
from __future__ import annotations
import os, subprocess, time, shlex, pathlib, sys
from typing import Dict, List, Optional
from pathlib import Path

def _is_windows() -> bool:
    return os.name == "nt"

def _python_from_venv(venv_path: str | Path) -> str:
    v = Path(venv_path)
    cand = v / ("Scripts/python.exe" if _is_windows() else "bin/python")
    if not cand.exists():
        raise FileNotFoundError(f"가상환경 python을 찾을 수 없습니다: {cand}")
    return str(cand)

def _auto_find_project_python(cwd: Path) -> Optional[str]:
    for name in [".venv", "venv"]:
        try:
            return _python_from_venv(cwd / name)
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
    python_exec: Optional[str] = None,   # ① 정확한 파이썬 경로 지정
    venv_path: Optional[str] = None,     # ② venv 루트
    conda_env: Optional[str] = None,     # ③ conda 환경 이름
    conda_exec: Optional[str] = None,    #   conda 바이너리 경로 없으면 PATH/CONDA_EXE
    timeout: Optional[float] = None,
    raise_on_error: bool = False
):
    pyfile_path = str(Path(pyfile))
    args = args or []

    # ----- 환경 변수 구성 -----
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    # (선택) 간섭 방지: 외부 PYTHONPATH가 다른 env에 섞이지 않도록
    # env.pop("PYTHONPATH", None)

    # ----- 실행 로그 준비 -----
    ts = time.strftime("%Y%m%d%H%M%S")
    run_dir = Path(log_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    workdir = Path(cwd).resolve() if cwd else Path.cwd()

    # ----- 어떤 인터프리터로 실행할지 결정 -----
    cmd: List[str]
    launcher_note = ""  # 로그용

    if python_exec:  # ①
        cmd = [python_exec, pyfile_path]
        launcher_note = f"python_exec={python_exec}"
    elif venv_path:  # ②
        pe = _python_from_venv(venv_path)
        cmd = [pe, pyfile_path]
        launcher_note = f"venv_path={venv_path}"
    elif conda_env:  # ③
        # conda 실행 파일 결정
        conda_bin = (
            conda_exec
            or os.environ.get("CONDA_EXE")
            or "conda"
        )
        # --no-capture-output: TTY가 아닐 때 출력 동작 일관성
        cmd = [conda_bin, "run", "--no-capture-output", "-n", conda_env, "python", pyfile_path]
        launcher_note = f"conda_env={conda_env}, conda_exec={conda_bin}"
    else:
        auto = _auto_find_project_python(workdir)
        if auto:
            cmd = [auto, pyfile_path]
            launcher_note = f"auto_detected={auto}"
        else:
            cmd = [sys.executable, pyfile_path]
            launcher_note = "fallback=sys.executable"

    cmd += args
    print("실행 명령어 상태 : ", cmd)
    result = {
        "ok": False,
        "run_id" : str(ts),
        "returncode": None,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "cmd": cmd,
        "error": None,
    }

    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"# CMD: {' '.join(map(shlex.quote, cmd))}\n")
            logf.write(f"# CWD: {str(workdir)}\n")
            logf.write(f"# LAUNCHER: {launcher_note}\n")
            logf.write(f"# CURRENT_PYEXEC: {sys.executable}\n\n")
            logf.flush()

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    cwd=str(workdir),
                    env=env,
                )
            except Exception as e_start:
                msg = f"[launcher] Failed to start process: {e_start}\n"
                logf.write(msg)
                result["error"] = str(e_start)
                if raise_on_error:
                    raise
                return result

            try:
                ret = proc.wait(timeout=timeout) if timeout is not None else proc.wait()
            except subprocess.TimeoutExpired:
                proc.kill()
                ret = -9
                logf.write("[launcher] Timeout expired. Process killed.\n")

        result["returncode"] = ret
        result["ok"] = (ret == 0)
        if ret != 0:
            result["error"] = f"Non-zero return code: {ret}"
            if raise_on_error:
                raise RuntimeError(f"Process failed with code {ret}. See {log_path}")
        return result

    except Exception as e:
        try:
            with open(log_path, "a", encoding="utf-8") as logf:
                logf.write(f"[launcher] Exception: {e}\n")
        except Exception:
            pass
        result["error"] = str(e)
        if raise_on_error:
            raise
        return result


'''
run_dir = run_python(
    "/path/to/project/output.py",
    args=["--epochs=3", "--learning_rate=0.001"],
    venv_path="/path/to/project/.venv",  # 또는 venv
)

run_dir = run_python(
    "/path/to/project/output.py",
    args=["--epochs=3"],
    python_exec="/Users/yujin/.pyenv/versions/3.11.7/bin/python",
)
`
run_dir = run_python(
    "/path/to/project/output.py",
    args=["--epochs=3"],
    conda_env="AutoFineTuner",
    # conda_exec="/Users/yujin/miniforge3/bin/conda",  # PATH에 없으면 이렇게
)

'''