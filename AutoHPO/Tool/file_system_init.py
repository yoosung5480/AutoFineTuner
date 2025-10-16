import os
import shutil
from pathlib import Path

def init_filesystem(base_dir : Path):
    """
    #### 기능 요약
    1. 'output/' 디렉토리와 'output.py' 파일의 존재를 확인.
    2. 존재할 경우 'temp/' 디렉토리를 생성 후 그 안으로 이동.
    3. 이동 완료 후 원본 'output/' 및 'output.py' 제거.

    #### 반환값
    temp_dir (Path): 이동된 temp 디렉토리 경로

    #### 작업파잉 상황
    실행디렉토리/
    - AutoFineTuner/
        - main.py     # 실행파일 위치
    - output/       # AutoHPO부산물
    - output.py     # AutoHPO부산물

    - temp/         # 시스템 시작시, 'output/' 내부에 있는 모든 파일과, output.py파일을 해당파일로 옮긴다.
    - target.py     # 기존작업 영역
    - ...그 외 파일들  # 기존작업 영역


    #### useCase
    1. main.py를 실행하면, 'output/'디렉토리와, 'output.py'파일의 존재유무를 검사한다.
    2. 만약 존재한다면 'output/'디렉토리와 'output.py'파일 전체를 모두다 'temp/' 디렉토리로 옮긴다. 
    3. 원래있던 'output/'디렉토리와 'output.py'을 제거한다.
    """
    output_dir = base_dir / "output"
    output_file = base_dir / "output.py"
    temp_dir = base_dir / "temp"

    temp_dir.mkdir(exist_ok=True)

    # 1️⃣ output 디렉토리 이동 (존재하지 않아도 예외 없음)
    try:
        if output_dir.exists() and output_dir.is_dir():
            dest_output = temp_dir / "output"
            if dest_output.exists():
                shutil.rmtree(dest_output)
            shutil.move(str(output_dir), str(dest_output))
            print(f"[INFO] 'output/' 디렉토리를 'temp/'로 이동 완료")
    except Exception as e:
        print(f"[WARN] 'output/' 디렉토리 이동 중 예외 발생: {e}")

    # 2️⃣ output.py 파일 이동 (존재하지 않아도 예외 없음)
    try:
        if output_file.exists() and output_file.is_file():
            dest_file = temp_dir / "output.py"
            if dest_file.exists():
                os.remove(dest_file)
            shutil.move(str(output_file), str(dest_file))
            print(f"[INFO] 'output.py' 파일을 'temp/'로 이동 완료")
    except Exception as e:
        print(f"[WARN] 'output.py' 이동 중 예외 발생: {e}")

    # 3️⃣ 원본 삭제 확인 (존재하지 않아도 예외 없음)
    try:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if output_file.exists():
            os.remove(output_file)
        print("[INFO] 불필요한 원본 파일 정리 완료")
    except Exception as e:
        print(f"[WARN] 파일 정리 중 예외 발생: {e}")

    print("[INFO] 파일 시스템 초기화 완료")
    return temp_dir