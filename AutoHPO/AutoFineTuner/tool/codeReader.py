# codeReader.py
from __future__ import annotations
import pathlib

def read_text_strict(path_str: str) -> str:
    ''' 
    path_str : 읽고자하는 파일의 경로.

    #### 사용예시.
    target_abs_path = "/Users/yujin/Desktop/코딩shit/python_projects/opensource/target.py"
    target_contents = read_text_strict(target_abs_path)
    print(target_contents)  # 실전에서는 너무 큰 파일은 미출력 또는 일부만 출력
    '''
    p = pathlib.Path(path_str).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"파일이 아니라 디렉터리/특수파일입니다: {p}")
    # 필요하다면 크기 제한 등을 둘 수 있음 (예: p.stat().st_size > 50MB 시 경고)
    return p.read_text(encoding="utf-8")
