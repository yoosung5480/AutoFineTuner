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
from typing import TypedDict, Annotated, Optional, Dict, List
from pathlib import Path
import json


def get_log(content : str, max_char :int = 5000):
    if len(content) > max_char:
        return content[-max_char:]
    else :
        return content
    


############################################################################################################################
# CodeState 상태클래쓰 정의.
# 리펙토링 결과 형태를 소스코드로 변환
# **recur_limit**, **OUTPUT_PATH_DEFAULT** 정의!!
############################################################################################################################
class CodeState(TypedDict):
    sourceCode: Annotated[str, "원본 소스코드"]
    refactoredCode: Annotated[str, "앱에 의해서 생성된 리펙토링된 소스코드"]
    log_content: Annotated[str, "코드실행결과 로그내용"]
    diagnosis: Annotated[dict, "코드 에러 원인 분석 및 해결방안"]
    result: Annotated[dict, "코드 실행 결과(run_python dict)"]
    exc_args: Annotated[dict, "실행변수 (run_python kwargs)"]
    count: Annotated[int, "현재까지 반복횟수"]

recur_limit = 3                                 # 자가 실행 수복 재귀방지 최대 반복횟수 (추후 실험적으로 재적용해보자.)
OUTPUT_PATH_DEFAULT = "../../../../output.py"   # 기본값 (exc_args에 pyfile 없을 때 사용)


############################################################################################################################
# excuteCode
# 코드 실행 및, 그 실행 로그를 반환한다.
############################################################################################################################
def excuteCode(state: CodeState):
    # 1) 실행 대상 코드 저장 (최신 refactoredCode를 실행)
    pyfile = state["exc_args"].get("pyfile", OUTPUT_PATH_DEFAULT)
    codeMaker.write_text_atomic(path_str=pyfile, content=state["refactoredCode"])

    # 2) 실행
    result = codeLauncher.run_python(**state["exc_args"])
    state["result"] = result

    # 3) 로그 적재
    try:
        state["log_content"] = get_log(codeReader.read_text_strict(path_str=result["log_path"]))
        print(state["log_content"])
    except Exception:
        state["log_content"] = ""

    # 4) 노드는 상태만 반환 (분기는 별도 함수가 결정)
    print("===============================excuteCode result===============================")
    print(result)
    return state


############################################################################################################################
# route_after_exec
# 코드 실행 후, 그 코드 실행 성공여부에 따라서 flow 조절 
# 성공 -> passthrough, 실패 -> codeRepairing
############################################################################################################################
def route_after_exec(state: CodeState) -> str:
    print("===============================route_after_exec===============================")
    if state.get("count", 0) >= recur_limit:
        print("반복횟수 초과.")
        return "passthrough"
    ok = bool(state.get("result", {}).get("ok"))
    if ok:
        print("코드 실행 성공!")
        return "passthrough"
    else:
        print("코드 실행 실패..")
        return "codeDiagnosis"

############################################################################################################################
# codeDiagnosis
# 원인 분석 노드이다. 에러원인과 해결방안을 제시해주는 고수준 추론노드다.
# 가장 높은 추론모델을 쓰면 좋다.
############################################################################################################################
prompt_to_diagnosis = PromptTemplate(
    template="""
# 지시사항
현재 refactoredCode 코드 실행상태는 런타임 에러가 떴다. 그 내용은 log_content에서 확인할 수 있다.
refactoredCode는 sourceCode를 리팩토링해 인자화한 코드다. sourceCode는 작동을 보장하는 코드다.

## 원인분석
log_content의 오류를 sourceCode를 참고해서 분석해라.

## 해결방안제시
원인분석내용을 토대로 오류 해결방안을 제시하라. 
 
# 소스코드
{{ sourceCode }}

# 리팩토링 코드
{{ refactoredCode }}

# 실행 로그 (발췌)
{{ log_content }}

# Answer.
답변은 간결하게 할것. 마크다운/이모티콘/기호 금지.
""",
    input_variables=["sourceCode", "refactoredCode", "log_content"],
    template_format="jinja2",
)

# -------- 원인 분석 노드: 에러 원인 평가 및 해결법제안 ----------
def codeDiagnosis(state: CodeState):
    print("===============에러 원인 분석 및 해결방안제시===============")
    chain = prompt_to_diagnosis | llms.llm_list["gpt-4o"] | StrOutputParser()
    draft = chain.invoke({
        "sourceCode": state["sourceCode"],
        "refactoredCode": state["refactoredCode"],
        "log_content": state.get("log_content", "")[:6000],  # 로그 길이 제한
    })
    # 설명 섞임 방지: 툴콜로 코드만 추출
    state["diagnosis"] = draft
    print(draft)
    return state


############################################################################################################################
# codeRepairing
# 실질적으로 코드를 고치는 노드이다. context : 로그내용, 원본소스코드 -> 코드를 재실행가능하게 고친다.
# 가장 높은 추론모델을 쓰면 좋다.
prompt_to_refactor = PromptTemplate(
    template="""
# 지시사항
현재 refactoredCode 코드 실행상태는 런타임 에러가 떴다. 그 내용은 log_content에서 확인할 수 있다.
refactoredCode는 sourceCode를 리팩토링해 인자화한 코드다. sourceCode는 작동을 보장하는 코드다.
log_content의 오류를 sourceCode와 diagnosis 참고해 보완하여 동작을 보장하는 코드로 바꿔라.
기존의 리팩토링된 코드의 인자들은 유지하고, 누락 시 추가하라.
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

# 리팩토링 코드
{{ refactoredCode }}

# 리팩토링 코드
{{ diagnosis }}

# 실행 로그
{{ log_content }}

# 출력 형식 (중요)
- 완전한 파이썬 스크립트만 출력. 설명/마크다운/코드펜스 금지.
""",
    input_variables=["sourceCode", "diagnosis" ,"refactoredCode", "log_content"],
    template_format="jinja2",
)

# -------- 리페어 노드: count 증가 + 코드 재생성 ----------
def codeRepairing(state: CodeState):
    print("현재 반복 횟수:", state.get("count", 0))
    print("===============codeRepairing===============")
    state["count"] = state.get("count", 0) + 1

    chain = prompt_to_refactor | llms.llm_list["gpt-5"] | StrOutputParser()
    draft = chain.invoke({
        "sourceCode": state["sourceCode"],
        "diagnosis" : state["diagnosis"],
        "refactoredCode": state["refactoredCode"],
        "log_content": state.get("log_content", "")[:6000],  # 로그 길이 제한
    })
    # 설명 섞임 방지: 툴콜로 코드만 추출
    state["refactoredCode"] = _extract_code_only_with_toolcall(draft)
    print(draft)
    return state


############################################################################################################################
# tool call을 이용해서 llm 답변을 소스코드 형식으로 파싱해준다.
# 상대적으로 낮은 추론모델을 사용해도 괜찮을듯하다.
############################################################################################################################
def _extract_code_only_with_toolcall(text: str) -> str:
    client = OpenAI()
    tools = [{
      "type": "function",
      "function": {
        "name": "emit_refactor",
        "description": "Return ONLY the refactored code. check if those args are cluded and if then not add those args 1.result_save_path, model_savepath 2.batch_size, 3.epochs",
        "parameters": {
          "type": "object",
          "properties": {"code": {"type":"string"}},
          "required": ["code"], "additionalProperties": False
        }
      }
    }]
    resp = client.chat.completions.create(
      model="gpt-4.1-nano",
      messages=[
        {"role":"system","content":"Return ONLY via the function call. No prose."},
        {"role":"user","content": f"Clean this to code-only (no prose):\n{text}"}
      ],
      tools=tools,
      tool_choice={"type":"function","function":{"name":"emit_refactor"}}
    )
    args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
    return args["code"]


############################################################################################################################
# 리펙토링 결과 형태를 소스코드로 변환
# 
############################################################################################################################
def repairedCodeParser(state: CodeState):
    print("===============================repairedCodeParser===============================")
    code = state["refactoredCode"]
    # 혹시 설명 섞였으면 한 번 더 정제 (옵션)
    code = _extract_code_only_with_toolcall(code)
    state["refactoredCode"] = code

    # 실행 대상 파일 경로: exc_args의 pyfile 기준
    pyfile = state["exc_args"].get("pyfile", OUTPUT_PATH_DEFAULT)
    codeMaker.write_text_atomic(path_str=pyfile, content=code)
    print(code)
    return state


############################################################################################################################
# 리펙토링 결과 형태를 소스코드로 변환
# END 노드로 보내기 위한 단순 전달노드.
############################################################################################################################
def passthrough(state: CodeState):
    print("워크플로우 종료.")
    return state


############################################################################################################################
# 코드 자가수복 워크플로우 정의.
############################################################################################################################
def getCodeRepairWorkflow():
    graph = StateGraph(CodeState)
    graph.add_node("excuteCode", excuteCode)
    graph.add_node("codeRepairing", codeRepairing)
    graph.add_node("codeDiagnosis", codeDiagnosis)
    graph.add_node("repairedCodeParser", repairedCodeParser)
    graph.add_node("passthrough", passthrough)

    graph.add_edge(START, "excuteCode")

    # 분기는 "노드 함수"가 아니라 "라우팅 함수"를 넣는다
    graph.add_conditional_edges(
        "excuteCode",
        route_after_exec,
        {"passthrough": "passthrough", "codeDiagnosis": "codeDiagnosis"},
    )
    graph.add_edge("codeDiagnosis", "codeRepairing")
    graph.add_edge("codeRepairing", "repairedCodeParser")
    graph.add_edge("repairedCodeParser", "excuteCode")
    graph.add_edge("passthrough", END)

    code_reconstruct_chain = graph.compile()
    return code_reconstruct_chain


def repair_code(
        source_code_content: str, 
        refactored_code_str: str, 
        pyfile:str, 
        conda_env_name:str) -> str:
    '''
    source_code_content : 리펙토링되기전 원본파일 (참고용)
    refactored_code_str : 현재 고치고자 하는 소스파일
    pyfile : 해당 소스파일의 경로 (실행인자로 사용)
    conda_env_name : 현재 콘다 가상환경 이름
    '''
    
    args=["--healthcheck"]     # 1회 반복이 감지되면 곧바로 종료
    code_args = {           
        "pyfile": pyfile,
        "conda_env": conda_env_name,
        "args" : args,
        "timeout": 300,         # 좀 늘려도 될듯하지만 일단 냅둘것.
        "raise_on_error": False,
    }
    codeState: CodeState = {
            "sourceCode": source_code_content,
            "refactoredCode": refactored_code_str,
            "log_content": "",
            "result": {},
            "exc_args": code_args,
            "count": 0,
        }
    repair_workflow = getCodeRepairWorkflow()
    codeState = repair_workflow.invoke(codeState)
    return codeState