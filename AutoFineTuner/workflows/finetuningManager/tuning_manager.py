from __future__ import annotations
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from typing import Annotated, List
from typing_extensions import TypedDict
from openai import OpenAI
import json

from AutoFineTuner.tool import codeLauncher
from AutoFineTuner.tool import codeMaker
from AutoFineTuner.tool import codeReader
from AutoFineTuner.tool import llms
from AutoFineTuner.tool.json_tools import load_json, atomic_write_json
from AutoFineTuner.tool.save_remove import safe_remove

from AutoFineTuner.workflows.codeRepair import repair
from AutoFineTuner.workflows.codeRefactor import refactor

from AutoFineTuner.tool.paths import ProjPaths, OutputPaths, get_proj_paths, get_output_paths

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json, os, tempfile, time, datetime, hashlib, random, string, re




######################################################################################################################################################################
# 헬퍼함수 모음
######################################################################################################################################################################
def now_id() -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rnd = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}_{rnd}"


# --- evaluatate 헬퍼 -----------------------------------------------------------------
def _as_bool(x):
    # JSON에서 'true'/'false' 문자열로 들어오는 경우까지 안전 처리
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() == "true"
    return bool(x)

def _as_int(x, default=0):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def _as_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default
    
# --- paramSelect 헬퍼 -----------------------------------------------------------------
def _extract_defined_cli_args(code: str) -> List[str]:
    """refactoredCode의 argparse에서 정의된 인자 이름 추출."""
    pat = re.compile(r'add_argument\(\s*["\']--([a-zA-Z0-9_\-]+)["\']')
    return pat.findall(code or "")

def _gather_param_history(results: Dict[str, Any], target: str):
    """results에서 target_param==target 인 히스토리 [(값 문자열, validation)] 수집."""
    hist = []
    for _, rec in (results or {}).items():
        if rec.get("target_param") != target:
            continue
        params = rec.get("params", {})
        val = rec.get("validation", None)
        if val is None:
            continue
        if target in params:
            hist.append((str(params[target]), float(val)))
    return hist

def _dedup_args(args: List[str]) -> List[str]:
    """--key=value 중복은 마지막 값으로 덮어쓰기."""
    seen = {}
    for a in args:
        if a.startswith("--") and "=" in a:
            k = a.split("=", 1)[0]
            seen[k] = a
        else:
            seen[a] = a
    return list(seen.values())



# --- evaluatate 헬퍼 -----------------------------------------------------------------
def _as_bool(x):
    # JSON에서 'true'/'false' 문자열로 들어오는 경우까지 안전 처리
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() == "true"
    return bool(x)

def _as_int(x, default=0):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def _as_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _should_end_by_params(param_info: dict) -> bool:
    """
    모든 파라미터가 (is_tuned==True) 또는 (cur_count >= limit_count)이면 True
    limit_count가 0/누락이면 튜닝 제한 없음으로 간주하여 그 파라미터는 종료 조건을 만족하지 않은 것으로 처리
    """
    if not isinstance(param_info, dict) or not param_info:
        return False
    for _, info in param_info.items():
        is_tuned = _as_bool(info.get("is_tuned", False))
        cur = _as_int(info.get("cur_count", 0), 0)
        lim = _as_int(info.get("limit_count", 0), 0)

        # limit_count가 0이면 "무제한"으로 간주 → 종료 조건 미충족
        reached_limit = (lim > 0 and cur >= lim)

        if not (is_tuned or reached_limit):
            return False
    return True

def _should_end_by_time(metadata: dict) -> bool:
    """
    time_limit < cur_exec_time 이면 종료 (요구사항 그대로 반영)
    """
    if not isinstance(metadata, dict):
        return False
    tl = _as_int(metadata.get("time_limit", 0), 0)
    cet = _as_int(metadata.get("cur_exec_time", 0), 0)
    return cet > tl  # 요구사항: time_limit < cur_exec_time

######################################################################################################################################################################
# 경로 정의
######################################################################################################################################################################
@dataclass
class ProjPaths:
    # 리펙토링 생성물 경로관리
    proj_root : Path            # /home/jeongyuseong/바탕화면/오픈소스경진대회/AutoFineTuner/
    target : Path               # main.py
    refactored : Path           # "test_output.py"    
    param_txt_name : Path       # "param.txt"

def get_proj_paths(proj_root: str | Path, target: str, refactored: str = "refactored.py", param_txt_name: str = "param.txt") -> ProjPaths:
    root = Path(proj_root)
    return ProjPaths(
        proj_root=root,
        target=root / target,
        refactored=root / refactored,
        param_txt_name =root  / param_txt_name
    )

@dataclass
class OutputPaths:
    # 훈련결과 생성물 경로관리.
    save_root: Path            # /home/jeongyuseong/바탕화면/오픈소스경진대회/AutoFineTuner/outputs/
    metadata: Path             # /home/jeongyuseong/바탕화면/오픈소스경진대회/AutoFineTuner/outputs//metadata.json
    results: Path              # /home/jeongyuseong/바탕화면/오픈소스경진대회/AutoFineTuner/outputs//results.json
    runs : list[Path]

def get_output_paths(save_dir: str | Path, run_id: Optional[str] = None) -> OutputPaths:
    root = Path(save_dir)
    return OutputPaths(
        save_root=root,
        metadata=root / "metadata.json",
        results=root / "results.json",
        runs = []
    )


######################################################################################################################################################################
# workState정의
######################################################################################################################################################################
# 추후 코드리페어를 위해서, WorkState는 CodeState와 호환되게 했다. repair함수는 CodeState를 인자로 사용하기때문이다.
class CodeState(TypedDict, total=False):
    sourceCode: Annotated[str, "원본 소스코드"]
    refactoredCode: Annotated[str, "앱에 의해서 생성된 리펙토링된 소스코드"]
    log_content: Annotated[str, "코드실행결과 로그내용"]
    diagnosis: Annotated[Dict[str, Any], "코드 에러 원인 분석 및 해결방안"]
    result: Annotated[Dict[str, Any], "코드 실행 결과(run_python dict)"]
    exc_args: Annotated[Dict[str, Any], "실행변수 (run_python kwargs)"]
    count: Annotated[int, "현재까지 반복횟수"]

class WorkState(CodeState, total=False):
    metadata: Annotated[Dict[str, Any], "파인튜닝 전체 메타데이터"]
    # dict 구조로 사용하므로 타입을 Dict로 고정
    results: Annotated[Dict[str, Any], "각 훈련결과 (run_id -> record)"]
    cur_conda_env: Annotated[str, "현재 가상환경"]
    cur_exec_code: Annotated[str, "이번 훈련 실행코드"]
    cur_exec_result: Annotated[Dict[str, Any], "이번 훈련 실행코드 실행결과"]
    cur_workId: Annotated[str, "이번 훈련의 id"]
    userPrompt: Annotated[str, "유저 프롬프트"]
    paramInfo: Annotated[str, "LLM이 제시한 파인튜닝 방향"]
    # 실제로 쓰는 키를 스키마에 추가
    selectedParamPlan: Annotated[Dict[str, Any], "이번 실험의 타겟/CLI/근거"]
    lastResult: Annotated[Dict[str, Any], "가장 마지막 실행 결과"]





######################################################################################################################################################################
# workflow의 노드를 담당할 함수들을 정의
######################################################################################################################################################################

def makeParamList(workState: Dict[str, Any], output_paths : OutputPaths):
    print("==========================================makeParamList============================================")
    """
    AutoFineTuner의 파라미터 메타데이터 JSON을 생성하고 저장한다.
    """
    
    # PromptTemplate 정의
    prompt_to_make_metadata = PromptTemplate(
        template="""
        [역할]
        너는 "argparse 코드 파라미터 추출기"이다.
        아래 Python argparse 코드에서 정의된 hyperparameter들을 읽고,
        반드시 **유효한 JSON 객체**로 변환한다.

        [규칙]
        - 제외: "test_path", "healthcheck", "save_dir"
        - 각 파라미터는 "param_info"에 포함
        - JSON 구조:
        {
            "param_info": {
            "<param_name>": {
                "is_tuned": false,
                "best_param": <default_value>,
                "cur_count": 0,
                "limit_count": 5
            }
            },
            "time_limit": 3600,
            "cur_exec_time": 0
        }
        - 설명/주석/문자열 따옴표 외 텍스트 절대 금지
        - 반드시 double quote만 사용
        - "param_info", "time_limit", "cur_exec_time" 상위구조를 엄격하게 지킬것.

        [입력]
        {{ sourceCode }}
        """,
        input_variables=["sourceCode"],
        template_format="jinja2",
    )


    # chain 실행
    code = workState["refactoredCode"]

    chain = (
        prompt_to_make_metadata 
        | llms.llm_list["gpt-4o"]
        | StrOutputParser()
    )
    metadata_content = chain.invoke(
        {"sourceCode": code}
    )

    print(metadata_content)

    # JSON Parsing Tool (wrapper 제거)
    client = OpenAI()
    json_parsing_tools = [{
    "type": "function",
    "function": {
        "name": "parse2json",
        "description": "Return ONLY a valid JSON object with hyperparam metadata. Do not include text outside JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "param_info": {"type": "object"},
                "time_limit": {"type": "integer"},
                "cur_exec_time": {"type": "integer"}
            },
            "required": ["param_info"],
            "additionalProperties": True
            }
        }
    }]

    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON generator. Output must be valid JSON via the function call. No prose, no code fences."
            },
            {
                "role": "user",
                "content": f"Extract argparse hyperparams and generate JSON metadata:\n{metadata_content}"
            }
        ],
        tools=json_parsing_tools,
        tool_choice={"type": "function", "function": {"name": "parse2json"}},
    )

    
    # 결과 파싱
    metadata = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
    workState["metadata"] = metadata
    print("메타데이터 저장경로 : ", output_paths.metadata)
    print(metadata)
    # 안전 저장
    atomic_write_json(path=output_paths.metadata, data=metadata)
    return workState


def check_metadata(workState: dict):
    """
    ## input:
    workState["metadata"]
    workState["count"]

    ## output:
    "paramSelect" or "makeParamList" or END
    """
    print("==========================================check_metadata============================================")
    metadata = workState.get("metadata", {})
    count = workState.get("count", 0)

    # ------------------------
    # Step 1. count 기반 종료
    # ------------------------
    MAX_COUNT = 5  # 외부 정의. 현재는 5로 설정.
    if count >= MAX_COUNT:
        print(f"종료 조건: count {count} >= {MAX_COUNT}")
        return END

    # ------------------------
    # Step 2. metadata 구조 확인
    # ------------------------
    required_top_keys = {"param_info", "time_limit", "cur_exec_time"}
    if not all(k in metadata for k in required_top_keys):
        print("metadata 상위 구조 불일치 → makeParamList")
        return "makeParamList"

    # param_info 내부 구조 확인
    param_info = metadata.get("param_info", {})
    if not isinstance(param_info, dict) or not param_info:
        print("param_info 구조 오류 → makeParamList")
        return "makeParamList"

    required_param_keys = {"is_tuned", "best_param", "cur_count", "limit_count"}
    for pname, pinfo in param_info.items():
        if not all(k in pinfo for k in required_param_keys):
            print(f"param_info[{pname}] 구조 오류 → makeParamList")
            return "makeParamList"

    # ------------------------
    # Step 3. 모든 조건 만족
    # ------------------------
    print("모든 조건 만족 -> paramSelect")
    return "paramSelect"


def paramSelect(workState: WorkState, output_paths : OutputPaths, proj_paths : ProjPaths):
    """
    1) metadata에서 '실행 가능 후보'(is_tuned==false & cur_count<limit_count)만 추출
    2) LLM이 후보 중 '다음 타겟'과 구체 cli_args를 결정(우선순위/동의어 모두 LLM 판단)
    3) 메타데이터의 best_param을 기본 args로 깔고, LLM이 제안한 인자로 덮어쓰기
    4) 코드 실행(run_python) 및 상태 갱신
    """
    print("==========================================paramSelect============================================")
    userPrompt: str = workState["userPrompt"]
    code: str = workState["refactoredCode"]
    results: Dict[str, Any] = workState.get("results", {}) or {}
    metadata: Dict[str, Any] = workState["metadata"]

    param_info: Dict[str, Any] = metadata.get("param_info", {}) or {}
    if not param_info:
        print("[paramSelect] metadata.param_info 없음")
        workState["selectedParamPlan"] = None
        return workState

    # 1) 실행 가능 후보(튜닝 미완료 & 제한 미도달)
    candidates = []
    for k, info in param_info.items():
        # 'false' 같은 문자열도 처리
        is_tuned = info.get("is_tuned", False)
        if isinstance(is_tuned, str):
            is_tuned = (is_tuned.lower() == "true")
        cur = int(info.get("cur_count", 0) or 0)
        lim = int(info.get("limit_count", 0) or 0)
        if (not is_tuned) and (cur < lim if lim > 0 else True):
            candidates.append(k)
            
    rationale_histories = {}
    for k in candidates:
        rh = []
        for rid, rec in (results or {}).items():
            if rec.get("target_param") == k and rec.get("rationale"):
                rh.append({
                    "run_id": rid,
                    "rationale": rec.get("rationale"),
                    "validation": rec.get("validation")
                })
        rationale_histories[k] = rh[-5:]

    if not candidates:
        print("[paramSelect] 더 이상 튜닝할 파라미터가 없습니다.")
        workState["selectedParamPlan"] = None
        return workState

    defined_args = _extract_defined_cli_args(code)

    plan_prompt = PromptTemplate(
    template="""
        # 역할
        아래 조건을 바탕으로 **다음 실험**에 사용할 파라미터 1개를 선택하고, 실행용 CLI 인자들을 제시하라.
        오직 JSON만 출력.

        # 출력(JSON)
        {
        "target_param": "<candidates 중 하나>",
        "cli_args": ["--<arg>=<value>", ...],  // 최소 1개, 반드시 아래 allowed_flags만 사용
        "rationale": "<200자 이내 근거>"
        }

        # 제약
        - target_param는 반드시 candidates 중 하나.
        - cli_args에 사용 가능한 인자 이름은 allowed_flags 안에서만 선택.
        - 동의어/우선순위/값 변화 폭 등은 너(LLM)가 판단.
        - history가 없으면 metadata.best_param을 시작점으로, 있으면 추세를 보고 보수적으로 증감.
        - **이전 실험의 rationale과 validation을 참고하여 동일 근거로 반복만 하지 말고, 성능 하락 근거가 있으면 반대 방향/대안 탐색을 시도할 것.**
        - 설명/코드펜스 금지. JSON만.

        # 입력
        [candidates]
        {{ candidates }}

        [allowed_flags]   // argparse에 실제로 정의된 인자들
        {{ allowed_flags }}

        [metadata(param_info)]
        {{ meta_param_info }}

        [recent_histories] // 각 후보별 최근 값-성능 기록
        {{ recent_histories }}

        [recent_rationales] // 각 후보별 최근 rationale + validation
        {{ recent_rationales }}

        [user_prompt]
        {{ user_prompt }}

        [source_code_excerpt]
        {{ source_excerpt }}
        """,
            input_variables=[
                "candidates","allowed_flags","meta_param_info",
                "recent_histories","recent_rationales","user_prompt","source_excerpt"
            ],
            template_format="jinja2",
        )


    # 후보별 최근 히스토리(최대 3개씩)
    histories = {k: _gather_param_history(results, k)[-5:] for k in candidates}

    draft = (plan_prompt | llms.llm_list["gpt-5"] | StrOutputParser()).invoke({
    "candidates": json.dumps(candidates, ensure_ascii=False),
    "allowed_flags": json.dumps(defined_args, ensure_ascii=False),
    "meta_param_info": json.dumps(param_info, ensure_ascii=False),
    "recent_histories": json.dumps(histories, ensure_ascii=False),
    "recent_rationales": json.dumps(rationale_histories, ensure_ascii=False),
    "user_prompt": userPrompt,
    "source_excerpt": code[:3000]
    })


    # 함수콜로 JSON 강제
    client = OpenAI()
    tools = [{
        "type": "function",
        "function": {
            "name": "emit_next_plan",
            "description": "다음 실험을 위한 단일 파라미터 선택과 실행 인자 제공.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_param": {"type": "string"},
                    "cli_args": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "rationale": {"type": "string"}
                },
                "required": ["target_param", "cli_args", "rationale"],
                "additionalProperties": False
            }
        }
    }]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return ONLY via the function call. No prose."},
            {"role": "user", "content": draft},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "emit_next_plan"}}
    )
    plan = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)

    target = plan["target_param"]
    llm_args: List[str] = plan["cli_args"]
    print("[planner]", plan.get("rationale", ""))

    # 3) 기본 인자: 메타데이터 best_param → 실제 argparse에 있는 키만 포함
    base_args: List[str] = []
    defined_set = set(defined_args)
    for k, info in (param_info or {}).items():
        if k in defined_set and "best_param" in (info or {}):
            base_args.append(f"--{k}={str(info['best_param'])}")

    # 최종 인자: 기본 + LLM 제안 (후자가 덮어씀)
    arg_save_dir = '--save_dir=' + str(output_paths.save_root)
    args = _dedup_args(base_args + llm_args)
    args.append(arg_save_dir)
    print("args : ", args)

    # 4) 실행 (중요: pyfile는 실제 파일 경로, args는 리스트로 전달)
    pyfile = str(proj_paths.refactored)  # 예: ".../refactored.py"
    run_kwargs = {
        "pyfile": pyfile,
        "args": args,
        "timeout": 3600,
        "log_dir" : output_paths.save_root,
        "raise_on_error": False,
    }
    if "cur_conda_env" in workState:
        run_kwargs["conda_env"] = workState["cur_conda_env"]

    print("exec_result 함수 실행 시도.")
    print("run_kwargs : ", run_kwargs)
    exec_result = codeLauncher.run_python(**run_kwargs)
    print(exec_result)
    # 상태 업데이트
    workState["cur_exec_result"] = exec_result
    workState["selectedParamPlan"] = {
        "target_param": target,
        "cli_args": args,
        "rationale": plan.get("rationale", "")
    }
    # 실행 로그 읽기
    workState["log_content"] = codeReader.read_text_strict(
        path_str=workState["cur_exec_result"]["log_path"]
    )
    print(workState["log_content"])
    # 카운트 증가(문자/불리언 혼재 방지)
    try:
        info = param_info.get(target, {})
        info["cur_count"] = int(info.get("cur_count", 0) or 0) + 1
        workState["metadata"]["param_info"][target] = info
    except Exception:
        pass
    

    return workState


def saveResult(workState: dict, output_paths : OutputPaths) -> dict:
    """현재 run의 result.json을 읽어 results.json에 run_id로 append/갱신하고,
    selectedParamPlan의 target_param, rationale을 같이 저장한다."""
    print("==========================================saveResult============================================")

    # 1) 현재 실행 결과(result.json) 로드
    result_path = output_paths.save_root.joinpath("result.json")
    result = load_json(result_path) or {}  # {'model_pt_path', 'validation', 'params'}

    # 2) run_id (= 실행 디렉토리 이름)
    run_id = Path(workState["cur_exec_result"]["run_dir"]).name

    # 3) 기존 results 로드(없거나 비었거나 깨지면 {})
    results_path = output_paths.results
    try:
        results = load_json(results_path) if results_path.exists() else {}
        if not isinstance(results, dict):
            results = {}
    except Exception:
        results = {}

    # 4) selectedParamPlan에서 보조 정보 추출
    plan = workState.get("selectedParamPlan") or {}
    target_param = plan.get("target_param")
    rationale = plan.get("rationale")

    # 5) 저장할 레코드 구성 (기존 예시 포맷 유지 + model_pt_path 포함)
    record = {
        "target_param": target_param,                  # 예: "epochs" / "lr"
        "rationale": rationale,                        # LLM 선택 이유 (선택)
        "validation": result.get("validation"),        # float
        "params": result.get("params", {}),            # dict
        "model_pt_path": result.get("model_pt_path"),  # str (선택)
    }

    # 6) results에 추가/갱신
    results[run_id] = record

    # 7) workState 업데이트
    workState["lastResult"] = record
    workState["results"] = results

    # 8) 파일로 저장 (append 효과를 위해 전체 덮어쓰기)
    #    가능하면 atomic write 유틸을 사용
    try:
        atomic_write_json(path=results_path, data=results)
    except NameError:
        # atomic_write_json이 없다면 일반 저장
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return workState


def evaluatate(workState: dict):
    """
    ## input
    workState["results"]
    workState["log_content"]
    workState["metadata"]

    ## output
    "updateBest" or "paramSelect" or END
    """
    print("==========================================evaluatate============================================")
    results = workState.get("results", {}) or {}
    log_content = workState.get("log_content", "") or ""
    metadata = workState.get("metadata", {}) or {}
    param_info = metadata.get("param_info", {}) or {}

    # -----------------------
    # Step0. 종료 조건 검사
    # -----------------------
    if _should_end_by_time(metadata) or _should_end_by_params(param_info):
        print("[evaluatate] 종료 조건 충족 → END")
        return END

    # -----------------------
    # Step1. 에러 여부 판단
    # -----------------------
    error_keywords = ["RuntimeError", "CUDA out of memory", "OOM", "NaN loss", "KeyboardInterrupt"]
    if any(err in log_content for err in error_keywords):
        print("[evaluatate] 에러 감지 → paramSelect")
        return "paramSelect"

    # -----------------------
    # Step2. 성능 평가
    # -----------------------
    if not results:
        print("[evaluatate] results 없음 → paramSelect")
        return "paramSelect"

    try:
        last_run_id = list(results.keys())[-1]
        last_run = results[last_run_id]
    except Exception:
        print("[evaluatate] 최근 실행 결과 접근 실패 → paramSelect")
        return "paramSelect"

    target_param = last_run.get("target_param")
    val_score = last_run.get("validation")

    if not target_param or val_score is None:
        print("[evaluatate] target_param/validation 누락 → paramSelect")
        return "paramSelect"

    best_param_info = param_info.get(target_param, {})
    best_score = best_param_info.get("best_score")

    if best_score is None or val_score > best_score:
        print(f"[evaluatate] best 갱신 필요: {target_param} → score={val_score}")
        return "updateBest"

    print("[evaluatate] 성능 향상 없음 → paramSelect")
    return "paramSelect"


def updateBest(workState: dict, output_paths : OutputPaths) -> dict:
    """
    workState["results"] 와 workState["metadata"]를 기반으로
    각 target_param의 best_param을 업데이트한다.
    """
    print("==========================================updateBest============================================")
    results = workState.get("results", {})
    metadata = workState.get("metadata", {})
    param_info = metadata.get("param_info", {})

    # target_param 별로 최고 validation 점수와 best_param을 저장
    best_by_param = {}

    for run_id, run_data in results.items():
        val_score = run_data.get("validation")
        target_param = run_data.get("target_param")
        params = run_data.get("params", {})

        if val_score is None or target_param is None:
            continue

        if target_param not in best_by_param:
            # 처음 본 경우 → 바로 저장
            print(f"[NEW] {target_param} -> {params.get(target_param)} (score={val_score})")
            best_by_param[target_param] = {
                "score": val_score,
                "value": params.get(target_param)
            }
        else:
            # 기존과 비교 후 더 좋으면 갱신
            if val_score > best_by_param[target_param]["score"]:
                print(f"[UPDATE] {target_param}: {best_by_param[target_param]['value']} (score={best_by_param[target_param]['score']}) "
                      f"-> {params.get(target_param)} (score={val_score})")
                best_by_param[target_param] = {
                    "score": val_score,
                    "value": params.get(target_param)
                }

    # metadata 갱신
    for param, best_data in best_by_param.items():
        if param in param_info:
            param_info[param]["best_param"] = best_data["value"]
            cur_count = int(param_info[param].get("cur_count", 0))
            param_info[param]["cur_count"] = cur_count + 1

    # 결과 반영
    workState["metadata"]["param_info"] = param_info
    atomic_write_json(path=output_paths.metadata, data=workState["metadata"])
    return workState




def run_finetuning(workState: WorkState, output_paths:OutputPaths, proj_paths : ProjPaths):
    # 0) output 루트 초기화: 기존 산출물 삭제
    #    - metadata.json, results.json, result.json 만 정리 (요구사항 그대로)
    #    - save_root는 존재하도록 보장
    targets = [
        output_paths.metadata,                  # .../outputs/metadata.json
        output_paths.results,                   # .../outputs/results.json
        output_paths.save_root / "result.json", # .../outputs/result.json (실행 결과 1회분)
    ]
    for p in targets:
        safe_remove(Path(p))

    MAX_STEP = 50  # 무한 루프 방지용
    step = 0

     # Step1: metadata 생성
    if "metadata" not in workState:
        workState = makeParamList(workState, output_paths=output_paths)

    # Step2: metadata 검사
        route = check_metadata(workState)
        if route == "makeParamList":
            workState = makeParamList(workState, output_paths=output_paths)
            
        elif route == END:
            print("종료 조건 충족 (check_metadata)")
            return workState


    while step < MAX_STEP:
        step += 1
        print(f"\n===== Step {step} =====")
        # Step3: 파라미터 선택 및 실행
        workState = paramSelect(workState, output_paths=output_paths, proj_paths=proj_paths)

        # Step4: 실행 결과 저장
        workState = saveResult(workState, output_paths=output_paths)

        # Step5: 평가
        route = evaluatate(workState)
        if route == "updateBest":
            workState = updateBest(workState, output_paths=output_paths)
            continue
        elif route == "paramSelect":
            continue
        elif route == END:
            print("종료 조건 충족 (evaluatate)")
            break

    return workState


# ######################################################################################################################################################################
# proj_path = "/home/jeongyuseong/바탕화면/오픈소스경진대회/AutoFineTuner/"    # params.txt, refactored.py 생성장소
# save_dir = "/home/jeongyuseong/바탕화면/오픈소스경진대회/AutoFineTuner/outputs/"     # 훈련결과 저장장소
# target_name = "main.py"
# refactored_name = "test_output.py"                                                  # 리펙토링될 코드 생성이름
# param_txt_name = "param.txt"                                                        # llm이 분석한 파라미터 내용
# cur_conda_env = "AItxt"
# output_paths = get_output_paths(save_dir=save_dir)
# proj_paths = get_proj_paths(proj_root=proj_path, target=target_name, refactored=refactored_name, param_txt_name=param_txt_name)
# ######################################################################################################################################################################
# paramInfo = codeReader.read_text_strict(str(proj_paths.param_txt_name))
# code = codeReader.read_text_strict(str(proj_paths.refactored))
# user_prompt = "두시간이내에 작업을 완료했으면해. 최적의 epochs, lr 정도만 찾으면돼."
# workState = WorkState()
# workState["userPrompt"] = user_prompt
# workState["refactoredCode"] = code
# workState["paramInfo"] = paramInfo
# workState["count"]=0
# workState["cur_conda_env"] = "AItxt"
# ######################################################################################################################################################################


from pathlib import Path

def start_finetuning(
    proj_path: str,
    save_dir: str,
    target_name: str,
    refactored_name: str,
    cur_conda_env: str,
    user_prompt: str
):
    """
    Fine-tuning 전체 워크플로우를 실행하는 단일 API.

    Parameters
    ----------
    proj_path : str
        프로젝트 루트 경로 (params.txt, refactored.py 생성 장소)
    save_dir : str
        훈련 결과 저장 경로
    target_name : str
        원본 실행 대상 파일명 (ex. main.py)
    refactored_name : str
        리팩토링된 코드 저장 파일명
    param_txt_name : str
        LLM 분석 파라미터 파일명
    cur_conda_env : str
        실행 환경 conda 환경명
    user_prompt : str
        사용자가 원하는 목표/조건 프롬프트
    """

    # 경로 정의
    proj_paths = get_proj_paths(
        proj_root=proj_path,
        target=target_name,
        refactored=refactored_name
    )
    output_paths = get_output_paths(save_dir=save_dir)

    # 코드 로드
    code = codeReader.read_text_strict(str(proj_paths.refactored))

    # workState 초기화
    workState = WorkState()
    workState["userPrompt"] = user_prompt
    workState["refactoredCode"] = code
    workState["count"] = 0
    workState["cur_conda_env"] = cur_conda_env

    # 실행
    workState = run_finetuning(
        workState=workState,
        output_paths=output_paths,
        proj_paths=proj_paths
    )
    return workState
