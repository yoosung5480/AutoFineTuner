# codeLauncher.py
from __future__ import annotations
import os, subprocess, time, shlex, pathlib
from typing import Dict, List, Optional
from pathlib import Path

def run_python(
        pyfile: str, 
        args=None, 
        env_overrides=None, 
        cwd: Optional[str] = None, 
        log_dir: str = "outputs"
    ):
    '''
    pyfile : 실행하고자 하는 파일의 경로.
    args : 파이썬 파일에 args 리스트
    env_overrides : 실행하고자 하는 파이썬의 환경설정. (미기입시, codeLauncher.py의 실행환경으로 파이썬 실행한다.)
    cwd : 실행할 작업 디렉터리 (없으면 현재 디렉터리)
    log_dir : 실행 로그가 저장될 디렉토리경로.

    #### 사용예시.
    output_path = "/Users/yujin/Desktop/코딩shit/python_projects/opensource/output.py"
    args = ["--epochs=3", "--learning_rate=0.001", "--batch_size=16"]
    run_dir = run_python(output_path, args=args)
    print(f"실행 완료. 로그 경로: {run_dir}")
    print("로그 파일 내용:")
    print(Path(run_dir, "run.log").read_text(encoding="utf-8"))
    '''
    pyfile_path = str(Path(pyfile))
    args = args or []
    # 현재 파이썬 가상환경 상황 설정.
    env = os.environ.copy() # 현재 프로세스(파이썬 실행 환경)의 환경변수를 dict 형태로 복사합니다.
    if env_overrides:       # env_overrides가 주어졌다면(dict), 기존 환경변수 env에 덮어씌웁니다.
        env.update(env_overrides)

    # 실행로그용 파일 생성
    ts = time.strftime("%Y%m%d%H%M%S")
    run_dir = pathlib.Path(log_dir)/ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir/"run.log"

    # 커멘드 생성.
    cmd = ["python", pyfile_path] + args
    # args = ["--epochs=3", "--learning_rate=0.001", "--batch_size=16"] 일때,
    # cmd = [python', '/Users/yujin/Desktop/코딩shit/python_projects/opensource/output.py', '--epochs=3', '--learning_rate=0.001', '--batch_size=16']


    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=cwd, env=env)
        # cmd: 실행할 명령어 리스트
        # stdout=logf: 표준 출력(stdout)을 로그 파일에 기록
        # stderr=subprocess.STDOUT: 표준 에러(stderr)도 stdout과 합쳐서 로그 파일에 기록
        # cwd: 실행할 작업 디렉터리 (없으면 현재 디렉터리)
        # env: 위에서 만든 환경변수 dict
        ret = proc.wait()   # 외부 프로세스가 끝날 때까지 대기하고, 종료 코드를 ret에 받습니다, 0이면 정상 종료, 0이 아니면 오류
    if ret != 0:
        raise RuntimeError(f"Process failed with code {ret}. See {log_path}")
    return str(run_dir)
