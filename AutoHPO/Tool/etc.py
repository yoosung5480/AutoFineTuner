import json
import os
from pathlib import Path

def make_safe_code(source_code : str) -> str:
    '''
    프롬프트에서 {}가 변수로 잘못인식되는 문제를해결하기 위해서 치환하는 과정
    '''
    return  (
        source_code
        .replace("{", "⟦")
        .replace("}", "⟧")
    )

def make_dict_safe_str(dict : dict):
    return (
            json.dumps(dict, ensure_ascii=False, indent=4)
            .replace("{", "⟦")
            .replace("}", "⟧")
        )


def get_latest_excuted_output_path(save_path : Path):
    '''
    save_path/
        {실행시간1}/result.json
        {실행시간2}/result.json
        ...
    위 디렉토리중 가장 최근에 수행된 result.json의 경로를 반환해준다.
    '''
    if not save_path.exists() or not save_path.is_dir():
        return False

    # output/{timestamp}/ 중 최신 실행폴더 탐색
    runs = sorted(save_path.glob("*/result.json"), key=os.path.getmtime, reverse=True)
    if not runs:
        return False

    result_path = runs[0]
    
    return result_path