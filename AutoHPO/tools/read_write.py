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


def write_json(save_path : Path, content : dict):
    ...


def write_metadata_json(save_path : Path, content : dict) -> bool:
    ...


def write_result_json(save_path : Path, content : dict) -> bool:
    ...


def write_api_json(save_path : Path, content : dict) -> bool:
    ...


def write_file(save_path : Path, content : str) -> bool:
    ...

#################################################################3

def read_json(path : Path) -> dict:
    ...


def read_file(save_path : Path, content : str) -> str:
    ...


def read_metadata_json(save_path : Path) -> dict:
    ...


def read_result_json(save_path : Path) -> dict:
    ...


def read_api_json(save_path : Path) -> dict:
    ...