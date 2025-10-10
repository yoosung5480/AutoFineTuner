'''
실행코드의 실행 가상한경 지정하면 알아서 코드 돌려줌.

1. refactor.py에서 코드를 이미. 지정한 api에 맞게 만들어놓고, api.json저장함
2. api.json를 읽고 코드를 수행함.
    2.1 api.json을 파싱해서 필요변수 가져오고 코드 생성하기.
    2.2 코드를 실행하기.
'''
from pathlib import Path
import read_write

def make_code(code_api_path : Path) -> str:
    api_dict = read_write.read_api_json(code_api_path)
    ### api_dict을 파싱해서, 단일 실행코드로 변환
    excute_code = "python main.py"
    return excute_code

def launch_code(code_api_path : Path) -> None:
    excute_code = make_code(code_api_path)
    ## excute_code 진짜 실행.