from typing import Annotated, List
from typing_extensions import TypedDict
from pathlib import Path
import os
import sys

class Container(TypedDict):
    sourceCodePath: Annotated[Path, "리펙토링할 소스코드 저장경로"]
    pythonEnv: Annotated[Path, "파이썬 실행 환경. 콘다 가상환경 없을때의 대안으로, 현재 가상환경을 자동으로 읽어와서 저장할거임."]
    savePath: Annotated[Path, "리펙토링된 코드의 부산물 저장경로. 작업디렉토리/output/{실행시간}/.. 으로 하드코딩될꺼임."]

    sourceCode: Annotated[str, "소스파일 경로에서 읽어들인 원본 코드"]
    refactoredCode: Annotated[str, "재실행을 위해서 인자화된 코드"]
    userRequirements: Annotated[str, "사용자의 요구사항"]

    # flags, 시작값 : 0
    isML: Annotated[bool, "현재 코드가 파인튜닝이 필요한 ML코드인지 판별 True, Flase"]

    # 시작값은 반드시 0 이여야함!!
    rewriteNodeCode: Annotated[int, "파일 실행 노드에서, 이 코드를 어떤 모드로 실행할지 지정, {0 : 기본 리펙토링 실행}, {1 : 런타임오류}, {2 : 파일 시스템 불일치}"]

    paramList: Annotated[list[str], "파인튜닝을 할 하이퍼 파라미터"]
    condaEnv: Annotated[str, "사용자가 지정하는 코드 실행을 위한 콘다 가상환경."]

    repairMaxTries : Annotated[int, "코드 리페어 최대 반복 시도횟수"]
    repairNum : Annotated[int, "현재까지 코드 리페어 반복횟수"]

    maxFineTuningTries : Annotated[int, "파인튜닝 최대 코드 수행횟수 "]
    fineTuningTrieNum : Annotated[int, "현재까지 파인튜닝 실행 횟수"]

    codeAPI : Annotated[dict, "리펙토링 된 코드를 실행하기위한 모든 필요한 인자가 들어있는 딕셔너리"]

    repairHistroy : Annotated[list[str], "현재까지 코드 고친 히스토리, 로그파일 내용과 고쳐진 소스코드로 이뤄져있다."]
    finetuningHistory : Annotated[list[str], "현재까지의 파인튜닝 히스토리, 선정된 하이퍼 파라미터와, 훈련결과, 분석내용으로 구성돼있다."]
    lastExcuteResult : Annotated[dict, "가장 마지막 코드 실행결과 딕셔너리"]
   

def get_current_python_env() -> Path:
    """
    현재 실행 중인 파이썬 인터프리터 경로를 반환.
    (예: /Users/user/anaconda3/envs/AutoFineTuner/bin/python)
    """
    python_path = Path(sys.executable).resolve()
    return python_path


def make_container() -> Container:
    return {
        "sourceCodePath": Path(""),
        "pythonEnv": get_current_python_env(),   
        "savePath": Path("./output"),
        "sourceCode": "",
        "refactoredCode": "",
        "userRequirements": "",
        "isML": False,
        "rewriteNodeCode": 0,
        "paramList": [],
        "condaEnv": "",
        "repairMaxTries": 3,
        "maxFineTuningTries": 10,
        "codeAPI": {},
        "repairNum": 1,
        "repairHistroy" : [],
        "finetuningHistory" : [],
        'lastExcuteResult' : {},
        "fineTuningTrieNum" : 0
    }
