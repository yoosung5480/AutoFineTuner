'''
소스코드를 분석하고, 리펙토링해주는 워크플로우 이다. 
input : 사용자 프롬포트, 소스코드
output : 파라미터 분석, 리펙토링된 소스코드파일

## 코드 사용예시

# 1. 소스파일,유저 프롬프트 받기 내용준비
source_file_contents = codeReader.read_text_strict(target_file)
# 2. state 정의
test_analyzer_state = analyzer.CodeRefactorState()
test_analyzer_state['sourceCode'] = source_file_contents
test_analyzer_state['userPrompt'] = "이틀 내에 작업을 완료하고싶어. 데이터는 참고로 대략 10~20만개의 문장이 준비돼있어."
# 3. 워크플로우 invoke
refactoring_workflow = analyzer.getAnalyerWorkflow()
refactoring_workflow.invoke(test_analyzer_state)
'''

from AutoFineTuner.tool import codeLauncher
from AutoFineTuner.tool import codeMaker
from AutoFineTuner.tool import codeReader
from AutoFineTuner.tool import llms

# codeAnalyzer 코드 실험
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from typing import Annotated, List
from typing_extensions import TypedDict
from openai import OpenAI
import json

############################################################################################################################
# 코드 분석용 state
############################################################################################################################
class CodeRefactorState(TypedDict):
    sourceCode: Annotated[str, "sourceCode"]
    modelInfo : Annotated[str, "modelInfo"]
    hyperParams: Annotated[str, "hyperParams"]
    userPrompt: Annotated[str, "userPrompt"]
    refactoredCode: Annotated[str, "refactoredCode"]




############################################################################################################################
# 코드 분석 -> 코드 리펙토링시, RAG로 사용할 context로 생성.
############################################################################################################################
prompt_to_reasoning_code = PromptTemplate(
    template="""
# 지시사항
소스코드에서 AI 라이브러리/모델 목적/훈련 흐름을 분석하고,
튜닝 후보 하이퍼파라미터를 제안하라.

# 소스코드
{{ sourceCode }}

# Answer
간결하게 요약하고, 근거가 되는 코드 위치(함수/클래스/라인 키워드)를 함께 제시하라.
""",
    input_variables=["sourceCode"],
    template_format="jinja2",
)

# 모델 이해하기
# basic 워크플로우로 사용.
def modelReasoning(state : CodeRefactorState):
    ''' 
    input :  state["sourceCode"] 원본 소스코드 파일내용 
    output : state["modelInfo"] llm 기반 소스코드 분석내용 답변.
    '''
    print("===================== modelReasoning 노드 실행 ======================")
    sourceCode = state["sourceCode"]
    chain = (
        prompt_to_reasoning_code
        | llms.llm_list["solar-pro2"]
        | StrOutputParser()
    )
    modelInfo = chain.invoke({"sourceCode" : sourceCode})
    print(modelInfo)
    state["modelInfo"] = modelInfo
    return state



# ############################################################################################################################
# # 하이퍼 파라미터 정보찾기.
# ############################################################################################################################
prompt_to_search_param = PromptTemplate(
    template="""
# 지시사항
modelInfo에서 제안하는 내용과 userPrompt의 내용을 바탕으로 파인튜닝때 적용할 하이퍼파라미터 변수들을 구체화해야한다.
예를들어, 
### modelInfo 내용의 일부
learning_rate`**: 0.01~0.3 범위에서 감소시키며 과적합 감소 테스트 (기본값 0.1)

### userPrompt의 일부
훈련시간의 여유는 충분히있어. 정확도를 우선으로 해서 모델의 성능을 반드시 개선하고싶어.
데이터셋의 크기는 이미지 대략 3600장이야. 이를 고려해서 에포크개수도 실험해줘.

### 예시 답변. 위 텍스트를 기반으로 아래와같이 구체적인 값들로 제안.
learning_rate = [0.01, 0.05, 0.10, 0.2, 0.3] # 시간이 충분함으로, 여러 학습률변수값을 제안.
epochs = [50, 100, 150, 200]    # 에포크값 실험. 데이터셋이 많지 않음으로 에포크를 크게설정.

# 소스코드
{{ sourceCode }}

# 훈련코드정보
{{ modelInfo }}

# 유저프롬프트
{{ userPrompt }}

# Answer
간결하게 요약하고, 근거가 되는 코드 위치(함수/클래스/라인 키워드)를 함께 제시하라.
""",
    input_variables=["modelInfo", "userPrompt"],
    template_format="jinja2",
)

# 하이퍼파라미터 찾기, 선정하기
def searchHyperParam(state : CodeRefactorState):
    ''' 
    input :  state["sourceCode"], state["modelInfo"], state["userPrompt"]
    output : state["hyperParams"] 
    '''
    sourceCode = state["sourceCode"]
    modelInfo = state["modelInfo"]
    userPrompt = state["userPrompt"]
    print("===================== searchHyperParam 노드 실행 ======================")
    chain = (
        prompt_to_reasoning_code
        | llms.llm_list["solar-pro2"]
        | StrOutputParser()
    )
    hyperParams = chain.invoke({"sourceCode" : sourceCode, "modelInfo":modelInfo,  "userPrompt" : userPrompt})
    state["hyperParams"] = hyperParams
    print(hyperParams)
    
    return state


############################################################################################################################
# 코드 리펙토링용 함수
############################################################################################################################

prompt_to_refactor = PromptTemplate(
    template="""
# 지시사항(필수)
아래 sourceCode를 기반으로 인자화된 실행 스크립트로 리팩토링하라.
**오직 실행 가능한 파이썬 코드만** 출력할 것(설명/마크다운/코드펜스 금지).

## 유지 원칙
- 데이터 경로/입출력 포맷/주요 알고리즘 로직은 유지
- 하드코딩 하이퍼파라미터만 argparse 인자로 치환

## CLI 인자(고정)
--epochs(int, default=1), --batch_size(int, default=1),
--save_dir(str, default="./outputs"),
--train_path(str, default=원본에서 추출), --test_path(str, default=원본에서 추출 or 필요 없으면 공백),
--healthcheck(flag)

## 결과 API 계약(필수)
리팩토링된 코드 안에 반드시 다음 함수를 정의하라:
def autofinetuner_result() -> dict:
    '''
    Returns:
      {
        "model_pt_path": str,
        "validation": float,
        "params": dict
      }
    '''
이 함수는
  (1) 데이터 로드/전처리/모델 구성
  (2) 학습 및 검증 지표 계산(원 코드의 핵심 지표 유지; 없으면 합리적 기본)
  (3) 모델을 <save_dir>/model.pt 로 저장(토치가 없으면 pickle fallback)
  (4) 위 3개 필드를 담은 dict를 반환
또한 main 블록에서 이 함수를 호출하고, 아래 의사코드를 포함해 표준출력에 한 줄 JSON을 찍어라:


# ---- Result utils (keep this block) ----
import sys, json, time, pickle
from pathlib import Path

def _save_model_generic(model, path: Path) -> str:
    \"""Try torch save, else pickle; always write to *.pt\"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch  # type: ignore
        try:
            torch.save(getattr(model, "state_dict", lambda: model)(), path)
        except Exception:
            torch.save(model, path)
    except Exception:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    return str(path)

def _emit_json_line(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()

# ---- CONTRACT: MUST exist & return dict ----
def autofinetuner_result() -> dict:
    \"""
    Returns:
      {
        "model_pt_path": str,
        "validation": float,
        "params": dict
      }
    \"""
    import argparse
    from pathlib import Path
    import time

    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--train_path", type=str, default="./datas/train.csv")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--healthcheck", action="store_true")
    args, _ = parser.parse_known_args()

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.pt"

    # --- (1) 데이터 로드/전처리/모델 구성 ---
    # TODO: 원본 코드 로직을 여기로 옮기거나 함수화해서 호출
    # ex) X_train, y_train, X_val, y_val = ...
    #     model = ...
    #     vectorizer = ...
    #     if args.healthcheck:
    #         # 1 샘플 전처리/forward만 수행하고 바로 종료
    #         print("READY"); raise SystemExit(0)

    # --- (2) 학습 ---
    # TODO: for epoch in range(args.epochs): model.fit(...)
    #       검증 점수 계산: val_metric = ...

    # --- (3) 모델 저장 ---
    # model 객체를 model_path에 저장(토치 없으면 pickle)
    saved = _save_model_generic(model, model_path)

    # --- (4) 결과 dict 구성(필수 3개 필드) ---
    used_params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        # 필요시 추가: "lr": args.lr, ...
    }
    result = {
        "model_pt_path": saved,
        "validation": float(val_metric),
        "params": used_params
    }


    # save_dir/save_dir.json 에 JSON 생성
    # 또한 한줄 JSON 생성.
    _save_json_line({"autofinetuner_result": result, "elapsed_sec": time.time() - t0})
    with open (save_dir/save_dir.json, "w", encoding="utf-8") as f:   
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result

if __name__ == "__main__":
    # 스크립트로 실행될 때도 API를 따라가도록
    try:
        _ = autofinetuner_result()
    except SystemExit as _e:
        # healthcheck 등 정상 종료 케이스 pass
        if _e.code != 0:
            raise


## Healthcheck 동작
--healthcheck가 참이면 데이터/전처리/모델 초기화 및 1샘플 forward까지만 수행하고
stdout에 READY 한 줄을 출력 후 0으로 종료. 학습/저장 금지.

# 소스코드
{{ sourceCode }}

# 하이퍼파라미터 힌트(참고용)
{{ hyperParams }}

# 사용자 프롬프트(맥락)
{{ userPrompt }}

# 출력 제한
실행 가능한 파이썬 코드만 출력하라.
""",
    input_variables=["sourceCode", "hyperParams", "userPrompt"],
    template_format="jinja2",     
)

# 찾은 하이퍼파라미터가 인자화된, 리펙토링된 코드를 반환받기.
def codeRefactoring(state : CodeRefactorState):
    ''' 
    input :  state["sourceCode"], state["hyperParams"], state["userPrompt"]
    output : state["refactoredCode"] 
    '''
    print("===================== codeRefactoring 노드 실행 ======================")
    sourceCode = state["sourceCode"]
    hyperParams = state["hyperParams"]
    userPrompt = state["userPrompt"]
    chain = (
        prompt_to_refactor
        | llms.llm_list["gpt-5"]
        | StrOutputParser()
    )
    refactoredCode = chain.invoke({"sourceCode" : sourceCode, "hyperParams":hyperParams,  "userPrompt" : userPrompt})
    state["refactoredCode"] = refactoredCode
    print(refactoredCode)
    return state

# result = codeRefactoring(result)
# print(result["refactoredCode"])



############################################################################################################################
# 리펙토링 결과 형태를 소스코드로 변환 
############################################################################################################################
def parseCode(state : CodeRefactorState):
    ''' 
    input :  state["refactoredCode"]
    output : state["refactoredCode"] 
    '''
    print("===================== parseCode 노드 실행 ======================")
    source_code = state["refactoredCode"]

    client = OpenAI()
    tools = [{
    "type": "function",
    "function": {
        "name": "emit_refactor",
        "description": "Return ONLY the refactored code. check if those args are cluded and if then not add those args 1.result_save_path, model_savepath 2.batch_size, 3.epochs, 4.healthcheck, ",
        "parameters": {
        "type": "object",
        "properties": {"code": {"type":"string"}},
        "required": ["code"],
        "additionalProperties": False
        }
    }
    }]

    resp = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role":"system","content":"Return ONLY via the function call. No prose."},
        {"role":"user","content": f"Refactor for CLI hyperparams:\n{source_code}"}
    ],
    tools=tools,
    tool_choice={"type":"function","function":{"name":"emit_refactor"}}
    )
    args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
    state["refactoredCode"] = args["code"]
    print(args["code"])
    return state



############################################################################################################################
# 전체 워크플로우 연결.
############################################################################################################################
def getAnalyerWorkflow():
    workflow = StateGraph(CodeRefactorState)
    workflow.add_node("modelReasoning", modelReasoning)
    workflow.add_node("codeRefactoring", codeRefactoring)
    workflow.add_node("searchHyperParam", searchHyperParam)
    workflow.add_node("parseCode", parseCode)

    workflow.add_edge(START, "modelReasoning")
    workflow.add_edge("modelReasoning", "searchHyperParam")
    workflow.add_edge("searchHyperParam", "codeRefactoring")
    workflow.add_edge("codeRefactoring", "parseCode")
    workflow.add_edge("parseCode", END)
    app = workflow.compile()
    return app


def start_refactor(
        source_code_contents:str, 
        user_prompt:str ):
    ''' 
    source_code_contents : 리펙토링 하고자하는 원본파일의 정보
    user_prompt : 사용자의 주문이 들어간 프롬프트
    '''
    refactor_state : CodeRefactorState = {
        "sourceCode": source_code_contents,
        "userPrompt": user_prompt,
    }
    refactor_workflow = getAnalyerWorkflow()
    final_state = refactor_workflow.invoke(refactor_state)
    return final_state["refactoredCode"]