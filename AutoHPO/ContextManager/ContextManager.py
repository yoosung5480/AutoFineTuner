''' 
공통 api : context manager에서는 str 형식만을 input으로 갖는다.
'''
from pathlib import Path
import json
from datetime import datetime

# AutoHPO/tools/prompts.py

def _make_safe_code(source_code : str) -> str:
    '''
    프롬프트에서 {}가 변수로 잘못인식되는 문제를해결하기 위해서 치환하는 과정
    '''
    return  (
        source_code
        .replace("{", "⟦")
        .replace("}", "⟧")
    )

############################################################################################
# 기본 프롬프트 반환 로직.
############################################################################################
def _get_result_json_skeleton(hyper_params: list[str], save_path: str, conda_env: str, python_env: str) -> dict:
    '''
    #### 기능
    - hyper_params 리스트를 순회하며 {"param_name": None} 형태로 구성
    - save_path, conda_env, python_env 값을 포함하는 result.json 스켈레톤을 생성
    - hyper_params의 예시: ["epochs", "batch_size", "learning_rate"]

    #### output
    result.json 구조의 기본 뼈대(dict)
    '''
    # hyper_params를 {param: None} 형태의 dict로 변환
    params_dict = {param: None for param in hyper_params}

    # 고정 필드 추가
    params_dict.update({
        "save_path": save_path,
        "healthcheack": 0
    })

    # 최종 result.json 스켈레톤 반환
    return {
        "실행시간1": {
            "params": params_dict,
            "system_env": {
                "conda_env": conda_env,
                "python_venv": python_env
            }
        }
    }



def get_basic_code_refactoring_prompt(
    source_code: str, 
    hyper_params: list[str], 
    user_requirements: str,
    save_path: str,
    conda_env: str,
    python_env: str
) -> str:
    """
    #### input
    - source_code: 원본 코드 문자열
    - hyper_params: 적용할 하이퍼파라미터 리스트 (예: ["epochs", "batch_size"])
    - user_requirements: 사용자의 추가 요구사항 설명
    - save_path : 해당 데이터가 저장돼야할 장소 (예: "./output")
    - conda_env : 이 코드의 콘다 실행 가상환경 이름
    - python_env : 이 코드의 실행을 위한 가상환경 경로

    #### output
    - LLM에 바로 전달 가능한 완성형 문자열 프롬프트

    #### 기능
    - 기존 리팩토링 프롬프트 구조를 유지하면서,
      result.json과 log 저장 로직을 강제로 포함시킨다.
    """ 
    source_code = _make_safe_code(source_code)
    # 기본 스켈레톤 구조
    params_dict = {param: None for param in hyper_params}
    params_dict.update({
        "save_path": save_path,
        "healthcheack": 0
    })

    json_skeleton = {
        "실행시간1": {          
            "validation_score": None,
            "train_score": None,
            "params": params_dict,
            "runtime_info": {       
                "start_time": "<자동 기록>",
                "end_time": "<자동 기록>",
                "elapsed_time_sec": "<자동 계산>"
            },
            "execution_status": {   
                "success": True,
                "error_type": None,
                "error_message": None,
                "exception_trace": None
            },
            "system_env": {        
                "conda_env": conda_env,
                "cuda_visible_devices": "auto",
                "device": "cuda",
                "python_venv": python_env
            }
        }
    }

    # 하이퍼파라미터 목록을 보기 좋게 표현
    hyper_param_str = ", ".join(hyper_params) if hyper_params else "(없음)"

    # ------------------------- PROMPT -------------------------
    prompt = f"""
    예시 출력(예시에서는 중괄호 대신 ⟦ ⟧ 표기를 사용. 실제 출력에서는 표준 JSON 중괄호 사용):
    아래 source_code를 기반으로 인자화된 실행 스크립트로 리팩토링하라.
    **오직 실행 가능한 파이썬 코드만** 출력할 것(설명/마크다운/코드펜스 금지).

    ## 유지 원칙
    - 데이터 경로/입출력 포맷/주요 알고리즘 로직은 반드시 유지
    - 하드코딩된 하이퍼파라미터만 argparse 인자로 치환
    - 원본의 함수 구조나 클래스 설계는 손상시키지 말 것

    ## CLI 인자(고정)
    --epochs(int, default=1), --batch_size(int, default=1),
    --save_dir(str, default="⟦save_path⟧"),
    --train_path(str, default=원본에서 추출),
    --test_path(str, default=원본에서 추출 or 필요 없으면 공백),
    --healthcheck(flag)

    ## 추가 저장 로직 (필수)
    학습 및 평가 완료 후 반드시 아래의 경로 구조로 결과를 저장할 것:
    └── ⟦save_path⟧/⟦실행시간⟧/
         ├── result.json
         └── ⟦실행시간⟧.log

    ### result.json 구조 (강제)
    아래의 스켈레톤 구조를 그대로 따르고, None 또는 placeholder는 실행 중 실제 값으로 채운다:
    ⟦json.dumps(json_skeleton, ensure_ascii=False, indent=4)⟧

    ### log 파일 규칙
    - 모든 터미널 출력(stdout)을 ⟦실행시간⟧.log 파일에 기록한다.
    - 로그는 학습 중간/완료 메시지를 모두 포함해야 한다.
    - 파일은 UTF-8로 저장한다.

    ## 결과 API 계약(기존 유지)
    아래 함수를 반드시 포함해야 한다:
    def autofinetuner_result() -> dict:
        '''
        Returns:
        ⟦
            "model_pt_path": str,
            "validation": float,
            "params": dict
        ⟧
        '''

    이 함수는 다음 단계를 수행한다:
    (1) 데이터 로드 및 전처리
    (2) 모델 구성 및 학습
    (3) 검증 및 평가지표 계산
    (4) 모델 저장 (<save_dir>/model.pt)
    (5) result.json 파일 작성 (위 스켈레톤 기반)
    (6) stdout에 JSON 한 줄 출력
    (7) log 파일 기록

    ## pseudo-code 예시 (유지 + 확장)
    아래 예시는 기존 프롬프트의 흐름을 유지하며, result.json 및 log 저장을 추가한 형태다.

    ```python
    import sys, json, time, pickle, os
    from pathlib import Path
    import datetime

    def _save_model_generic(model, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            torch.save(getattr(model, "state_dict", lambda: model)(), path)
        except Exception:
            with open(path, "wb") as f:
                pickle.dump(model, f)
        return str(path)

    def _emit_json_line(payload: dict) -> None:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\\n")
        sys.stdout.flush()

    def autofinetuner_result() -> dict:
        import argparse
        t0 = time.time()
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--save_dir", type=str, default="⟦save_path⟧")
        parser.add_argument("--train_path", type=str, default="./datas/train.csv")
        parser.add_argument("--test_path", type=str, default="")
        parser.add_argument("--healthcheck", action="store_true")
        args, _ = parser.parse_known_args()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")   # timestamp = 202510102010 일경우
        run_dir = Path(args.save_dir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        log_path = run_dir / timestamp.log"                 # 202510102010.log로 저장
        sys.stdout = open(log_path, "w", encoding="utf-8")

        # --- (1) 데이터 로드 ---
        # TODO: 원본 코드 로직 복사
        # --- (2) 모델 학습 ---
        # TODO: 학습 루프
        # --- (3) 검증 ---
        val_metric, train_metric = 0.0, 0.0  # TODO: 실제 값으로 대체
        # --- (4) 모델 저장 ---
        model_path = run_dir / "model.pt"
        _save_model_generic(model, model_path)

        # --- (5) result.json 저장 ---
        result = ⟦json.dumps(json_skeleton, ensure_ascii=False, indent=4)⟧
        result["실행시간1"]["validation_score"] = val_metric
        result["실행시간1"]["train_score"] = train_metric
        result["실행시간1"]["runtime_info"]["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        result["실행시간1"]["runtime_info"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        result["실행시간1"]["runtime_info"]["elapsed_time_sec"] = time.time() - t0
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        # --- (6) 종료 ---
        _emit_json_line(result)
        return result
    ```

    ## 입력 내용
    # 원본 코드
    {source_code}

    # 하이퍼파라미터
    {hyper_param_str}

    # 사용자 요구사항
    {user_requirements}

    # 출력 제한
    오직 실행 가능한 파이썬 코드만 출력하라.
    """

    return prompt


def get_basic_param_list_prompt(source_code: str, user_requirements: str) -> str:
    """
    #### input
    - source_code: 원본 코드 문자열
    - user_requirements: 사용자의 추가 요구사항 설명

    #### output
    - LLM에 바로 전달 가능한 완성형 문자열 프롬프트

    #### 기능
    - 주어진 ML 코드(source_code)에서 외부 인자화가 필요한 하이퍼파라미터 후보 리스트를 추출하는 프롬프트를 생성한다.
    - 모델, 데이터 로더, 학습 루프 등에서 '수동 설정된 값(epochs, lr, batch_size 등)'을 찾아내어 리스트 형태로 반환해야 한다.
    """
    source_code = _make_safe_code(source_code)
    prompt = f"""
        # 역할
        너는 머신러닝/딥러닝 코드를 분석하여, 하드코딩된 설정값 중에서 외부에서 인자화(argparse 등)하기 적합한 하이퍼파라미터 목록을 제안하는 전문 분석가이다.
        예시 출력(예시에서는 중괄호 대신 ⟦ ⟧ 표기를 사용. 실제 출력에서는 표준 JSON 중괄호 사용):

        # 지시사항
        아래 제공된 sourceCode를 분석하여 **외부 CLI 인자화 또는 하이퍼파라미터 튜닝 대상으로 적합한 변수 이름 리스트**를 도출하라.

        ## 후보 기준
        - 학습 제어 관련 변수: epochs, batch_size, lr, weight_decay 등
        - 모델 구조 관련 변수: hidden_size, dropout, num_layers 등
        - 데이터/전처리 관련 변수: max_len, image_size, augmentations 등
        - 경로/장치/플래그: save_dir, device, use_gpu 등은 제외
        - 주석이나 문자열 안의 값은 무시

        ## 출력 형식
        오직 파이썬 리스트 형태의 문자열로 출력하라. (예: ["epochs", "lr", "batch_size"])
        설명, 마크다운, 코드펜스, JSON 객체 등은 금지.

        ## 사용자 요구사항(선택적 참고)
        {user_requirements}

        # 원본 코드
        {source_code}

        # 출력 규칙
        - 오직 리스트 형태로만 반환
        - 후보가 전혀 없을 경우 빈 리스트 []를 출력
        """
    return prompt


def get_basic_code_anaylsis_prompt(source_code: str, user_requirements: str) -> str:
    """
    #### input
    - source_code: 원본 코드 문자열
    - user_requirements: 사용자의 추가 요구사항 설명

    #### output
    - LLM에 바로 전달 가능한 완성형 문자열 프롬프트

    #### 기능
    - 제공된 소스코드가 '내용상' 머신러닝 또는 딥러닝 학습 코드인지 판별하기 위한 프롬프트를 생성한다.
    - 코드의 실행 가능성은 고려하지 않는다. (실행 여부는 후속 프로세스에서 검증됨)
    - 모델, 학습 루프, 손실 계산, 데이터 로드 등 ML 관련 구성요소가 포함되어 있으면 True로 판단한다.
    """
    source_code = _make_safe_code(source_code)
    prompt = f"""
        # 역할
        너는 머신러닝/딥러닝 코드를 판별하는 전문 개발자이다.
        아래의 코드가 **내용적으로 머신러닝 또는 딥러닝 학습을 수행하는 코드인지** 판단하라.
        예시 출력(예시에서는 중괄호 대신 ⟦ ⟧ 표기를 사용. 실제 출력에서는 표준 JSON 중괄호 사용):

        # 판단 기준
        다음의 요소 중 일부라도 포함되면 머신러닝 코드(True)로 판정한다.
        - 모델, 신경망, 또는 ML 알고리즘 객체 정의 (예: model =, nn.Module, LogisticRegression, Trainer 등)
        - 데이터 로드 및 전처리 (pandas.read_csv, Dataset, DataLoader 등)
        - 학습 루프 또는 손실 계산 (for epoch in range, criterion, loss.backward(), optimizer.step() 등)
        - 모델 평가 또는 검증 (accuracy_score, f1_score, evaluate, validation loss 등)
        - 모델 저장, 가중치 저장 (torch.save, joblib.dump, pickle.dump 등)
        - 파라미터 튜닝 또는 파인튜닝 코드 (fit, compile, predict, train 등)

        다음의 경우는 머신러닝 코드(False)가 아니다.
        - 단순 데이터 전처리나 시각화만 수행하는 코드
        - 웹 서버, API, 파일 입출력, OS 유틸리티 스크립트 등 일반 목적 코드
        - 함수나 클래스 정의만 있고, 학습 로직이 없는 코드
        - 파라미터 설정 없이 계산기나 수학 연산만 수행하는 코드

        # 판단 목적
        - 코드의 실행 가능 여부는 고려하지 않는다.
        - 오직 '내용상 머신러닝 관련 코드인가'만 판단하라.

        # 사용자 요구사항 (참고용)
        {user_requirements}

        # 분석 대상 코드
        {source_code}

        # 출력 규칙
        - 오직 JSON 스키마에 맞게 결과를 출력한다.
        - 코드 내용이 머신러닝 관련이면 "isML": true
        - 그렇지 않으면 "isML": false
        - 불확실하거나 애매하면 false로 처리한다.
        """
    return prompt





def get_basic_finetuner_prompt(train_result : str, train_history :str, user_requirements: str) -> str:
    """
    #### input
    - train_result: 가장 최신의 훈련 코드 결과. json파일 내용을 기반으로 생성한 문자열을 통해 각 파라미터 조합별 성능 점수와 피드백을 볼수있다.
    - train_history: 지금껏 훈련 결과. json파일 내용을 기반으로 생성한 문자열을 통해 각 파라미터 조합별 성능 점수와 피드백을 볼수있다.
    - user_requirements: 사용자의 추가 요구사항 

    #### output
    - LLM에 바로 전달 가능한 완성형 문자열 프롬프트

    #### 기능
    - 기존 refactoring_basic_prompt의 구조를 그대로 따르되,
      input 변수를 직접 주입하여 최종 문자열을 반환한다.
    
      
    """
    return ""



############################################################################################
# 워크플로우
############################################################################################
def get_code_refactoring_prompt(
    source_code: str, 
    hyper_params: list[str], 
    user_requirements: str,
    save_path: str,
    conda_env: str,
    python_env: str
    ) -> str:
    return get_basic_code_refactoring_prompt(
        source_code=source_code, 
        hyper_params=hyper_params, 
        user_requirements=user_requirements,
        save_path=save_path,
        conda_env=conda_env,
        python_env=python_env
        )

def get_param_list_prompt(source_code: str, user_requirements: str) -> str:
    return get_basic_param_list_prompt(source_code=source_code, user_requirements=user_requirements)


def get_code_anaylsis_prompt(source_code: str, user_requirements: str) -> str: 
    return get_basic_code_anaylsis_prompt(source_code=source_code, user_requirements=user_requirements)


def get_code_excute_api_prompt(refactored_code: str) -> str:
    """
    #### input
    - refactored_code: 리팩토링 완료된 코드 전문 (argparse 포함)

    #### output
    - LLM에 전달할 완성형 프롬프트 문자열

    #### 기능
    - 코드 내부에서 argparse, add_argument 등을 탐색하여
      실행 시 외부에서 지정 가능한 모든 매개변수(arguments)를 추출하도록 지시한다.
    - 각 인자의 이름, 타입, 기본값(default)을 포함하여 딕셔너리로 반환한다.
    - 코드 실행 가능성은 고려하지 않고, 코드 내용상 정의된 인자들을 기준으로 판단한다.
    """ 
    refactored_code = _make_safe_code(refactored_code)
    prompt = f"""
     # 역할
    너는 Python 코드를 분석하여 코드 실행 시 외부에서 전달 가능한 모든 argparse 인자(argument)를 추출하는 분석가다.

    # 지시사항
    아래의 리팩토링된 파이썬 코드(refactored_code)를 분석하고,
    코드에 정의된 모든 argparse 인자(add_argument)를 찾아 최종적으로 하나의 JSON 딕셔너리로 요약하라.
    예시 출력(예시에서는 중괄호 대신 ⟦ ⟧ 표기를 사용. 실제 출력에서는 표준 JSON 중괄호 사용):

    ## 판단 기준
    - 대상: parser.add_argument("--name", type=..., default=..., action=...) 형태로 선언된 인자
    - 이름: 선행 '--'는 제거하여 저장 (예: --epochs → "epochs")
    - 타입: type 힌트가 있으면 반영. 없으면 default로 추론. 모두 직렬화 가능한 값으로 표현
    - flag: action="store_true"/"store_false"는 1 또는 0으로 표현
    - 경로형 인자(save_dir/save_path/train_path/test_path 등)가 있으면 포함
    - 코드에 존재하지 않는 임의 인자 추가 금지

    ## 출력 형식(중요)
    - 오직 하나의 JSON 딕셔너리만 출력
    - 최상위 키는 반드시 "API"
    - API 값은 ⟦ 인자명": 기본값 ⟧ 형태의 딕셔너리
    - 설명 문장, 마크다운, 코드펜스 출력 금지

    예시 입력(발췌):
      parser = argparse.ArgumentParser()
      parser.add_argument("--epochs", type=int, default=10)
      parser.add_argument("--batch_size", type=int, default=32)
      parser.add_argument("--save_path", type=str, default="./outputs")
      parser.add_argument("--healthcheck", action="store_true")

    
      ⟦
        "API": ⟦
          "epochs": 10,
          "batch_size": 32,
          "save_path": "./outputs",
          "healthcheck": 0
        ⟧
      ⟧

    # 분석 대상 코드
    {refactored_code}


    # 출력 규칙
    - 실제 출력 시에는 표준 JSON 중괄호를 사용하고, 오직 하나의 딕셔너리만 출력한다.
    - 최상위 키는 "API" 여야 하며, 그 값으로 모든 argparse 인자를 포함한다.
    - JSON 문법 오류가 없도록 완전한 구조로 출력한다.
    """
    return prompt