# codeMaker.py
from __future__ import annotations
import os
import time
import pathlib
from typing import Optional

def write_text_atomic(
    path_str: str,
    content: str,
    *,
    encoding: str = "utf-8",
    newline: Optional[str] = "\n",
    make_backup: bool = False
) -> pathlib.Path:
    """
    path_str : 쓰고자 하는 파일의 경로.
    content : path_str에 쓰고자 하는 내용

    #### 사용예시.
    target_abs_path = "/Users/yujin/Desktop/코딩shit/python_projects/opensource/target.py"
    output_abs_path = "/Users/yujin/Desktop/코딩shit/python_projects/opensource/output.py"
    target_contents = read_text_strict(target_abs_path)
    out_path = write_text_atomic(output_abs_path, target_contents, make_backup=False)
    print(f"생성 완료: {out_path}")   

    #### 코드설명.
    실무용 가벼운 안전 쓰기:
    1) 상위 디렉터리 생성
    2) tmp 파일에 먼저 기록 + flush + fsync
    3) 필요 시 .bak 백업 생성
    4) tmp를 최종 파일로 원자적 교체
    """
    p = pathlib.Path(path_str).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    # 선택: 기존 파일 백업
    if make_backup and p.exists() and p.is_file():
        ts = time.strftime("%Y%m%d%H%M%S")
        backup = p.with_suffix(p.suffix + f".{ts}.bak")
        try:
            p.replace(backup)  # rename으로 빠르게 백업(원자적)
        except Exception:
            # replace가 부담스럽다면 copy로 대체 가능. 여기선 간단히 무시.
            pass

    # 임시 파일에 먼저 쓰기
    tmp = p.with_suffix(p.suffix + ".tmp")
    # 텍스트 모드로 열고 newline='\n'로 고정하면 OS와 무관하게 줄바꿈 일관성 유지
    with open(tmp, "w", encoding=encoding, newline=newline) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())

    # 최종 교체 (원자적)
    tmp.replace(p)
    return p
