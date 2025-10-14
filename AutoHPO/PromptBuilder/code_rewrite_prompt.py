from AutoHPO.Tool.etc import make_safe_code
import json

def get_result_skeleton(hyper_params, save_path, conda_env, python_env): 
    # 기본 스켈레톤 구조
    params_dict = {param: None for param in hyper_params}
    params_dict.update({
        "save_path": save_path,
        "healthcheack": 0
    })
    return {
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
            },
            "system_env": {        
                "conda_env": conda_env,
                "cuda_visible_devices": "auto",
                "device": "cuda",
                "python_venv": python_env
            }
        }
    }



def get_basic_code_rewrite_prompt(
    source_code: str, 
    hyper_params: list[str], 
    user_requirements: str,
    save_path: str,
    json_skeleton : str,
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
    # print("--------------------json_skeleton----------------")
    # print(json_skeleton)

    # 하이퍼파라미터 목록을 보기 좋게 표현
    hyper_param_str = ", ".join(hyper_params) if hyper_params else "(없음)"

    # ------------------------- PROMPT -------------------------
    # 교정된 프롬프트
    prompt = f"""
    너는 AutoFineTuner 시스템의 코드 리팩토링 에이전트이다.  
    지금부터 너는 제공된 원본 머신러닝 코드를 **자동화된 파인튜닝 실행 코드**로 리팩토링해야 한다.  
    이 코드는 사용자의 가상환경(conda, python path) 안에서 실행 가능해야 하며,  
    모든 훈련 결과는 아래와 같은 **표준 디렉토리 구조**를 따라야 한다

    예시 출력(예시에서는 중괄호 대신 ⟦ ⟧ 표기를 사용. 실제 출력에서는 표준 JSON 중괄호 사용):
    결과 파일 저장 구조는 반드시 아래 규칙을 따라야 한다.
    # ===============================================
    # (1) 결과 파일 구조 (필수)
    # ===============================================
    output/
        ⟦실행시간⟧/                  ← 실행 시각 기반 하위 폴더 (예: 20251014024632)
            result.json              ← 학습 결과 JSON
            model.pt                 ← 학습된 모델 가중치
            ⟦실행시간⟧.log           ← 모든 stdout 로그 저장

    주의: result.json, model.pt, 로그파일은 반드시 동일한 하위 폴더에 저장되어야 한다.  (추가 부산물 가능)
    즉, output/⟦실행시간⟧/ 내부에 ⟦실행시간⟧.log,result.json, model.pt등의 추가 부산물 파일이 모두 존재해야 한다.
    
    ## 유지 원칙
    - 데이터 경로/입출력 포맷/주요 알고리즘 로직은 반드시 유지
    - 하드코딩된 하이퍼파라미터만 argparse 인자로 치환
    - 원본의 함수 구조나 클래스 설계는 손상시키지 말 것

    ## CLI 인자(고정)
    --epochs(int, default=1), --batch_size(int, default=1),
    --save_dir(str, default="./output"),
    --train_path(str, default=원본에서 추출),
    --test_path(str, default=원본에서 추출 or 필요 없으면 공백),⟦⟧
    --healthcheck(int, choices=⟦0,1⟧, default=0)   # 정수 플래그(0/1)로 강제. store_true 금지.
    # 호출 예: --healthcheck=0 또는 --healthcheck=1

    ### healthcheck 모드
    - 0이면 전체 학습 실행
    - 1이면 데이터/모델 초기화만 확인 후 "READY" 출력 후 종료 (학습 및 저장 스킵)

    ## 결과 파일 생성 규칙
    1. 실행 시각(timestamp)을 기반으로 `run_dir = Path(args.save_dir) / timestamp` 디렉토리 생성  
    2. 모든 출력(log, result.json, model.pt)을 **이 run_dir 안에 저장**
    3. stdout을 run_dir/⟦timestamp⟧.log 로 리다이렉션
    4. 학습 완료 후 result.json을 다음 구조로 저장

    ### result.json 구조 (강제)
    아래의 스켈레톤 구조를 그대로 따르고, None 또는 placeholder는 실행 중 실제 값으로 채운다:
    {json_skeleton}

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

        
    # pseudo-code 예시 
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
        parser.add_argument("--save_dir", type=str, default="./output")
        parser.add_argument("--train_path", type=str, default="./datas/train.csv")
        parser.add_argument("--test_path", type=str, default="")
        parser.add_argument("--healthcheck", type=int, choices=[0,1], default=0)
        args, _ = parser.parse_known_args()
        hc = bool(args.healthcheck)

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        run_dir = Path(args.save_dir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # ✅ 동일한 timestamp로 로그 파일명 고정
        log_path = run_dir / f⟦timestamp⟧.log"
        sys.stdout = open(log_path, "w", encoding="utf-8")

        # (1) 데이터 로드
        # TODO: 원본 코드 로직 복사/유지

        if hc:
            # 헬스체크 모드
            print("READY")
            return ⟦"healthcheck": 1⟧

        # (2) 학습
        # TODO

        # (3) 검증
        val_metric, train_metric = 0.0, 0.0  # TODO

        # (4) 모델 저장
        model_path = run_dir / "model.pt"
        _save_model_generic(model, model_path)

        # (5) result.json 저장
        result = ⟦
            "실행시간1": ⟦
                "validation_score": val_metric,
                "train_score": train_metric,
                "params": vars(args),
                "runtime_info": ⟦
                    "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_time_sec": time.time() - t0
                ⟧,
                "execution_status": ⟦
                    "success": True,
                    "error_type": None,
                    "error_message": None,
                    "exception_trace": None
                ⟧
            ⟧
        ⟧
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        _emit_json_line(result)
        return result

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


def get_runtime_error_rewrite_prompt(
    source_code: str, 
    user_requirements: str, 
    repair_history: list
) -> str:
    ''' 
    ### 런타임 에러일때 (rewriteNodeCode : 1)
    #### 프롬프트 구조
    전체 프롬프트 = ( 배경 설명 및 코드 생성 규칙 ) + ( 이전 코드 수정 히스토리 )
    #### useCase
    1. 런타임 에러 원인을 해결하기 위한 코드 수정 프롬프트 생성
    2. LLM에게 전달 가능한 완전한 문자열 반환
    '''
    # 수정 히스토리 문자열 정리
    history = ""
    if repair_history:
        for i, hist in enumerate(repair_history, start=1):
            history += f"\n--- {i}번째 코드 수정 히스토리 ---\n{hist}\n"
    else:
        history = "(이전 수정 기록 없음)\n"

    runtime_error_repair_prompt = f"""
    당신은 뛰어난 머신러닝 개발자이자 디버깅 전문가입니다.
    아래 코드는 실행 중 **런타임 오류(Runtime Error)** 가 발생하여 수정이 필요한 코드입니다.
    원래 코드의 알고리즘과 데이터 처리 로직은 그대로 유지하되,
    런타임 오류의 원인(변수 참조, 타입 오류, import 오류 등)을 해결하도록 코드를 수정하십시오.

    ## 지시사항
    - 모델 학습/검증 로직의 흐름은 변경하지 마십시오.
    - 파일 저장, result.json, 로그 기록 로직은 변경하지 마십시오.
    - 단순히 런타임 오류를 일으킨 부분만 수정하십시오.
    - 추가적인 import, 예외처리, 경로 생성 등은 필요하다면 허용됩니다.
    - 출력은 반드시 실행 가능한 파이썬 코드만 포함하십시오.

    ## 수정할 소스코드
    {source_code}

    ## 사용자 요구사항
    {user_requirements}

    ## 이전 코드 수정 히스토리
    {history}

    ## 출력 형식
    오직 수정된 실행 가능한 파이썬 코드만 출력하시오. (마크다운/설명 금지)
    """
    return runtime_error_repair_prompt
   


def get_filesystem_inconsistency_rewrite_prompt(
    source_code: str, 
    user_requirements: str, 
    repair_history: list,
    json_skeleton : str
) -> str:
    '''
    ### 파일시스템 불일치일때 (rewriteNodeCode : 2)
    #### 프롬프트 구조
    전체 프롬프트 = ( 기본 배경 설명(파일시스템 불일치 수정) 및 코드 생성 규칙 ) + ( 이전 코드 수정 히스토리 )
    #### useCase
    1. 코드의 학습/모델 로직은 그대로 유지
    2. 파일 저장(result.json, log) 형식 및 경로 불일치 문제 수정
    '''
    history = ""
    if repair_history:
        for i, hist in enumerate(repair_history, start=1):
            history += f"\n--- {i}번째 코드 수정 히스토리 ---\n{hist}\n"
    else:
        history = "(이전 수정 기록 없음)\n"

    print(" ======================= [DEBUG] : history =======================")
    print(history)

    filesystem_fix_prompt = f"""
    당신은 뛰어난 머신러닝 엔지니어입니다.
    아래 코드는 학습은 정상적으로 실행되었으나,
    **파일 시스템 불일치(FileSystem Inconsistency)** 문제가 발생했습니다.
    즉, result.json 또는 log 파일이 지정된 규격에 맞게 저장되지 않았습니다.

    ## 지시사항
    - 데이터 로드 및 학습/검증 로직은 절대 수정하지 마십시오.
    - 반드시 result.json과 log 파일이 아래 경로에 올바른 이름으로 저장되도록 수정하십시오.
      예: ./output/⟦실행시간⟧/result.json, ./output/⟦실행시간⟧/⟦실행시간⟧.log
    - result.json에는 반드시 다음 키가 존재해야 합니다:
        ["validation_score", "train_score", "runtime_info", "execution_status", "system_env"]
    ### result.json의 필수인자가 들어간 skeleton 딕셔너리 형태
    {json_skeleton}
    - 로그 파일에는 학습 진행 및 완료 메시지가 포함되어야 합니다.
    - 저장 경로나 파일명 하드코딩 대신, 반드시 Path 객체를 사용하십시오.
    - 출력은 오직 실행 가능한 파이썬 코드여야 하며, 마크다운/설명 금지.

    ## 수정할 소스코드
    {source_code}

    ## 사용자 요구사항
    {user_requirements}

    ## 이전 코드 수정 히스토리
    {history}

    ## 출력 형식
    오직 수정된 실행 가능한 파이썬 코드만 출력하시오. (마크다운/설명 금지)
    """
    return filesystem_fix_prompt

def get_code_rewrite_prompt(
        source_code: str, 
        hyper_params: list[str], 
        user_requirements: str,
        save_path: str,
        conda_env: str,
        python_env: str,
        repair_history : list,
        flag : int
    ) -> str:
    source_code = make_safe_code(source_code)


    json_skeleton = get_result_skeleton(hyper_params, save_path, conda_env, python_env)
    json_skeleton_str = (
        json.dumps(json_skeleton, ensure_ascii=False, indent=4)
        .replace("{", "⟦")
        .replace("}", "⟧")
    )

    if flag == 0 :
        return get_basic_code_rewrite_prompt(
            source_code=source_code,
            hyper_params=hyper_params, 
            user_requirements=user_requirements,
            save_path=save_path,
            json_skeleton=json_skeleton_str
        )
    elif flag == 1 :
        return get_runtime_error_rewrite_prompt(source_code=source_code, user_requirements=user_requirements, repair_history=repair_history)
    elif flag == 2 :
        return get_filesystem_inconsistency_rewrite_prompt(source_code=source_code, 
                                                           user_requirements=user_requirements, 
                                                           repair_history=repair_history,
                                                           json_skeleton=json_skeleton_str)
    else :
        print("invalid rewrite_propmt flag number, it must be interger which is in [0, 1, 2] 0:refactor, 1:runtime error, 2:file system inconsistency")
        return ""