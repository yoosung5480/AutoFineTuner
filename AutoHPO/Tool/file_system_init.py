import shutil
import os
from pathlib import Path

def init_filesystem(base_dir: Path):
    """
    #### 기능 요약
    1. base_dir 내에 'output/' 디렉토리와 'output.py' 파일의 존재를 확인.
    2. 존재할 경우, 최상위 'temp/' 디렉토리를 생성(없으면 새로 생성) 후 해당 항목들을 그 안으로 이동.
    3. 원본 'output/' 및 'output.py' 제거.
    4. 이동 완료 후 temp 디렉토리 경로를 반환.
    """
    output_dir = base_dir / "output"
    output_file = base_dir / "output.py"
    temp_dir = base_dir / "temp"

    # ✅ temp 디렉토리 확인 및 생성
    try:
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] temp 디렉토리 생성 완료 → {temp_dir}")
        else:
            print(f"[INFO] 기존 temp 디렉토리 사용 → {temp_dir}")
    except Exception as e:
        print(f"[ERROR] temp 디렉토리 생성 실패: {e}")
        return None

    # ✅ output 디렉토리 이동
    try:
        if output_dir.exists() and output_dir.is_dir():
            dest_output = temp_dir / f"output_backup"
            # 이전 백업 있으면 제거
            if dest_output.exists():
                shutil.rmtree(dest_output)
            shutil.move(str(output_dir), str(dest_output))
            print(f"[INFO] 'output/' 디렉토리를 'temp/output_backup/' 으로 이동 완료")
    except Exception as e:
        print(f"[WARN] output 디렉토리 이동 중 예외 발생: {e}")

    # ✅ output.py 파일 이동
    try:
        if output_file.exists() and output_file.is_file():
            dest_file = temp_dir / "output_backup.py"
            if dest_file.exists():
                os.remove(dest_file)
            shutil.move(str(output_file), str(dest_file))
            print(f"[INFO] 'output.py' 파일을 'temp/output_backup.py' 로 이동 완료")
    except Exception as e:
        print(f"[WARN] output.py 이동 중 예외 발생: {e}")

    # ✅ 원본 정리 (혹시 잔여파일 있을 시)
    try:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if output_file.exists():
            os.remove(output_file)
        print("[INFO] 원본 output/ 및 output.py 제거 완료")
    except Exception as e:
        print(f"[WARN] 원본 제거 중 예외 발생: {e}")

    print("[INFO] 파일 시스템 초기화 완료 — 깨끗한 output 환경 준비됨")
    return temp_dir
