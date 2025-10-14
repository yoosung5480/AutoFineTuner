'''
I/O 입출력은 모두다 이 파일에 책임이 있다.
'''
import json
from pathlib import Path
# class write_json_exception
# class write_metadata_json_exception
# class write_result_json_exception
# class write_api_json_exception
# class write_file_exception
# class read_json_exception
# class read_metadata_json_exception
# class read_result_json_exception
# class read_api_json_exception
# class read_file_exception


def write_json(save_path: Path, content: dict):
    '''
    save_path에, content 딕셔너리를 JSON 파일로 저장
    디렉토리가 없으면 자동 생성하며, UTF-8로 인코딩한다.
    '''
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)


def write_metadata_json(save_path : Path, content : dict) -> bool:
    ...


def write_result_json(save_path : Path, content : dict) -> bool:
    ...


def write_api_json(save_path : Path, content : dict) -> bool:
    ...



def write_file(save_path: Path, content: str) -> bool:
    '''
    #### input
    save_path : 저장경로
    content : 작성할 파일내용

    #### output
    성공여부 (bool)

    #### 기능 
    save_path에 content를 문자열로 저장한다.
    디렉토리가 없으면 생성하고, 성공 시 True, 실패 시 False를 반환한다.
    '''
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception:
        return False

################################ READ #################################3 

import json
from pathlib import Path

def read_json(path: Path) -> dict:
    '''
    #### input
    path : Path 객체 또는 문자열 경로

    #### output
    - 성공 시: JSON 파일 내용을 dict로 반환
    - 실패 시: {}
    '''
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON 파일이 존재하지 않습니다: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    except json.JSONDecodeError as e:
        print(f"[read_json] JSONDecodeError: {e}")
    except FileNotFoundError as e:
        print(f"[read_json] FileNotFoundError: {e}")
    except Exception as e:
        print(f"[read_json] Unexpected Error: {e}")

    # 실패 시 빈 dict 반환
    return {}



def read_file(save_path: Path) -> str:
    '''
    #### input
    save_path : 저장경로

    #### output
    파일의 전체 내용을 문자열(str)로 반환

    #### 기능
    지정된 save_path의 텍스트 파일을 UTF-8로 읽어 전체 내용을 반환한다.
    파일이 존재하지 않으면 FileNotFoundError를 발생시킨다.
    '''
    if not save_path.exists():
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {save_path}")
    with open(save_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_metadata_json(save_path : Path) -> dict:
    ...


def read_result_json(save_path : Path) -> dict:
    ...


def read_api_json(save_path : Path) -> dict:
    ...