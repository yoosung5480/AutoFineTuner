'''
get_basic_code_rewrite_prompt를 구현해야한다.

def get_basic_code_rewrite_prompt(
    source_code: str,
    hyper_params: List[str],
    user_requirements: str,
    save_path: str,
    json_skeleton: str,
) -> str:
'''

from AutoHPO.Tool.etc import make_safe_code
from AutoHPO.Tool.llms import llm_list

import asyncio
import os
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
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






def get_generic_pseudo_code (save_path : str):
    return f"""
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
        parser.add_argument("--save_dir", type=str, default={save_path})
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
"""


def get_prompt_base_header(json_skeleton : str, save_path:str):
    return f"""
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
    --save_dir(str, default={save_path}),
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
"""


def get_prompt_basic_tail(source_code : str, hyper_param_str : str, user_requirements : str):
    return f"""
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

def get_compare_promt(source_code : str, user_requirements : str, prompt : str):
    return f"""
        ## 상황설명
        너는 프롬프트 엔지니어링을 하는 개발자다. 아래는 주어진 소스코드와 사용자의 요청이다.
        해당코드는 하이퍼 파라미터 튜닝을 위한 실행가능한 머신러닝 코드이다. 해당 코드를 외부에서 하이퍼 파라미터를
        인자화해서 재활용 가능한 코드로 리펙토링 하기위해 프롬프트들을 선별 해내야한다.
        원본 코드와 사용자 요구사항과, 제공된 프롬프트를 비교해서 적절하면 "yes", 적절하지 않으면 "no"를 호출한다.
        
        판단기준은 보수적으로 한다. 적절하지 않은것을 적절하지 않다고 판단하는것이 더 중요하다.
        예를들어, 같은 머신러닝 코드라도, 프롬프트는 keras_tf상황을 가정하고있는데, 현재 코드는 그렇지 않다면 적절하지 않는다고 판단한다.

        # 원본 코드
        {source_code}

        # 하이퍼파라미터
        {user_requirements}

        # 사용자 요구사항
        {prompt}

        # 대답 
        부가설명이나 텍스트 없이 반드시 "yes" 또는 "no"로 대답해야한다.
    """

def get_prompt_whole(header, generic_pseudo_code, body, tail):
    return f"""
        {header}
        {generic_pseudo_code}

        # ===============================
        # [선택된 세부 프롬프트 가이드라인]
        # ===============================
        {body}

        {tail}
    """

# -------------------------------------------------------------
# (1) 관련 유틸
# -------------------------------------------------------------
class RelavantCodeCheck(BaseModel):
    isRelavant: str = Field(description="단일 프롬프트에 대해 해당 프롬프트가 현재 코드/요구사항과 관련 있는지 LLM에게 검사ㄴ, 'yes' or 'no'")

# -------------------------------------------------------------
# (2) 단일 프롬프트 적합성 검사 함수
# -------------------------------------------------------------
async def check_prompt_relavant(prompt_str: str, source_code: str, user_requirements: str):
    """
    단일 프롬프트가 현재 코드/요구사항과 관련 있는지 검사
    """
    llm = llm_list["gpt-5-nano"].with_structured_output(RelavantCodeCheck)

    compare_prompt = get_compare_promt(
        source_code=source_code,
        user_requirements=user_requirements,
        prompt=prompt_str,
    )

    chat = ChatPromptTemplate.from_messages([
        ("system", "You are a strict ML code reviewer. Respond only with 'yes' or 'no'."),
        ("human", compare_prompt),
    ])

    # ❌ StrOutputParser 제거
    # ✅ structured_output을 직접 받는다
    chain = chat | llm
    result = await chain.ainvoke({})

    print("is relavant? :", result.isRelavant)
    return result.isRelavant.strip().lower() == "yes"


# -------------------------------------------------------------
# (3) 프롬프트 통합 함수
# -------------------------------------------------------------
async def build_dynamic_rewrite_prompt(
    source_code: str,
    hyper_params: List[str],
    user_requirements: str,
    save_path: str,
    json_skeleton: str,
) -> str:
    """
    - ./pseudo_code_prompts 디렉토리의 모든 .txt 파일 중
      현재 코드 및 요구사항과 '적합한' 프롬프트만 자동 선택하여
      최종 리팩토링용 프롬프트를 동적으로 생성한다.
    """

    prompts_dir = Path("AutoHPO/PromptBuilder/pseudo_code_prompts")
    if not prompts_dir.exists():
        raise FileNotFoundError(f"프롬프트 디렉토리가 존재하지 않습니다: {prompts_dir}")

    # 1️⃣ 텍스트 파일들 읽기
    prompt_files = list(prompts_dir.glob("*.txt"))
    print(f"[DEBUG] 프롬프트 경로: {prompts_dir.resolve()}")
    print(f"[DEBUG] 발견된 프롬프트 파일 수: {len(prompt_files)}")
    if not prompt_files:
        raise ValueError("*.txt 프롬프트 파일이 없습니다.")

    open_files = [p.read_text(encoding="utf-8") for p in prompt_files]
    prompts_list = [(p.name, make_safe_code(content)) for p, content in zip(prompt_files, open_files)]

    # 2️⃣ 병렬 관련성 평가
    tasks = [
        check_prompt_relavant(content, source_code , user_requirements)
        for _, content in prompts_list
    ]
    results = await asyncio.gather(*tasks)

    # 3️⃣ 관련 프롬프트만 필터링
    selected_prompts = [content for (name, content), ok in zip(prompts_list, results) if ok]
    print(f"[INFO] 관련성 통과 프롬프트 {len(selected_prompts)}/{len(prompts_list)}개 선택됨")


    header = get_prompt_base_header(json_skeleton, save_path)
    tail = get_prompt_basic_tail(source_code, ", ".join(hyper_params), user_requirements)
    body = "\n\n".join(selected_prompts)
    generic_pseudo_code = get_generic_pseudo_code(save_path=save_path)
    prompt_whole = get_prompt_whole(
        header=header,
        generic_pseudo_code=generic_pseudo_code,
        body=body,
        tail=tail
    )
    print(prompt_whole)
    return prompt_whole

# -------------------------------------------------------------
# (4) 외부에서 호출할 수 있는 동기 wrapper
# -------------------------------------------------------------
def get_basic_code_rewrite_prompt(
    source_code: str,
    hyper_params: List[str],
    user_requirements: str,
    save_path: str,
    json_skeleton: str,
) -> str:
    """
    내부적으로 build_dynamic_rewrite_prompt()를 비동기로 실행하고,
    관련 프롬프트들을 자동 선택 후 통합하여 완성형 LLM 프롬프트를 반환한다.
    """
    return asyncio.run(
        build_dynamic_rewrite_prompt(
            source_code=source_code,
            hyper_params=hyper_params,
            user_requirements=user_requirements,
            save_path=save_path,
            json_skeleton=json_skeleton,
        )
    )