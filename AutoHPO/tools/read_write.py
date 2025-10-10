''' 
json, 실행파일 파일의 read, write
단, result.json, metadata.json, api.json 이 3개의 파일은 내부적인 쓰는 로직이 존재한다.

main useCase 시나리오.
1. api.json 이 생성된다.
2. api.json을 통해 excutor.py 모듈로 코드를 샐행한다.
3. result.json이 생성된다.
4. result.json과 *.log파일을 통해서 FineTuner가 작동한다.
5. metadata.json이 생성된다.
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

def read_json(path : Path) -> dict:
    ...


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