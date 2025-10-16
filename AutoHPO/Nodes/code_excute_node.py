import os
import subprocess
import json
from pathlib import Path
from AutoHPO.Tool.read_write import read_json
from AutoHPO.Instance.container import Container
from AutoHPO.Tool.codeLauncher import run_python
from AutoHPO.Tool.etc import get_latest_excuted_output_path

# def get_latest_excuted_output_path(save_path : Path):
#     '''
#     save_path/
#         {실행시간1}/result.json
#         {실행시간2}/result.json
#         ...
#     위 디렉토리중 가장 최근에 수행된 result.json의 경로를 반환해준다.
#     '''
#     if not save_path.exists() or not save_path.is_dir():
#         return False

#     # output/{timestamp}/ 중 최신 실행폴더 탐색
#     runs = sorted(save_path.glob("*/result.json"), key=os.path.getmtime, reverse=True)
#     if not runs:
#         return False

#     result_path = runs[0]
    
#     return result_path


# ============================================================
# 1️⃣ 파일 시스템 일관성 검사 함수
# ============================================================
def check_filesystem_consistency(save_path: Path) -> bool:
    '''
    save_path/output/{실행시간}/result.json 과 {실행시간}.log가 존재하고,
    result.json 내부 필드가 규격대로 존재하는지 검사.
    '''
    
    result_path = get_latest_excuted_output_path(save_path=save_path)
    run_dir = result_path.parent
    log_path = run_dir / f"{run_dir.name}.log"

    # 1️ log/result 파일 존재여부
    if not result_path.exists() or not log_path.exists():
        return False

    # 2️ result.json 내부 키 검사
    required_top_keys = [
        "validation_score", "train_score",
        "runtime_info", "execution_status", "system_env"
    ]
    try:
        result_data =  read_json(result_path) # result_path
        # 가장 상위 (실행시간1 등) 키 자동 탐색
        top_key = next(iter(result_data.keys()))
        result_body = result_data[top_key]

        for key in required_top_keys:
            if key not in result_body:
                return False

        # 필드 구조만 보장되면 True
        return True
    except Exception:
        return False


# ============================================================
# 2️⃣ 코드 실행 함수
# ============================================================
def make_code_and_excute(api_json: dict) -> int:
    '''
    #### input
    api_json : 실행정보(json) 구조
    #### output
    0: 정상 실행 및 파일시스템 일치
    1: 런타임 오류
    2: 파일시스템 불일치
    '''
    try:
        env_info = api_json.get("env", {})
        script_path = Path(env_info.get("script"))
        arguments = env_info.get("arguments", {})
        conda_env = env_info.get("CONDA_ENV")
        python_env = env_info.get("PYTHON_ENV_PATH")
        cuda_devices = env_info.get("CUDA_VISIBLE_DEVICES")

        # 실행 인자 리스트 구성 (--key value)
        args = []
        for k, v in arguments.items():
            if v is None or v == "":
                continue
            # flag형 (0/1) 처리
            if isinstance(v, bool) and v:
                args.append(f"--{k}")
            else:
                args.append(f"--{k}={v}")

        # 환경 변수 구성
        env_overrides = {}
        if cuda_devices:
            env_overrides["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

        print(f"[EXECUTE-API] 실행 스크립트: {script_path}")
        print(f"[EXECUTE-API] 인자 목록: {args}")
        print(f"[EXECUTE-API] conda_env={conda_env}, python_env={python_env}")

        # run_python을 통한 실제 실행
        run_result = run_python(
            pyfile=str(script_path),
            args=args,
            env_overrides=env_overrides,
            log_dir="output",           # log는 output/{timestamp}/run.log 형태로 자동 저장
            conda_env=conda_env if env_info.get("USE_CONDA") else None,
            python_exec=python_env if not env_info.get("USE_CONDA") else None,
            timeout=None,                # 10분 제한
            raise_on_error=False
        )
        print("============동작결과===============")
        print(run_result)

        # --- 실행 결과 해석 ---
        if not run_result["ok"]:
            print(f"[EXECUTE FAIL] {run_result['error']}")
            return 1  # 런타임 오류

        
        # save_dir 키 이름은 코드에 따라 달라질 수 있으므로 fallback
        save_dir = Path(arguments.get("save_dir", arguments.get("save_path", "./outputs")))
        last_excute_result_path = get_latest_excuted_output_path(save_path=save_dir)
        last_excute_result = read_json(last_excute_result_path)
        run_id = list(last_excute_result.keys())[0]
        print("latest_result_path : ", last_excute_result_path)
        print("latest_result_json : ", last_excute_result)

        # llm이 리펙토링 해준 코드가, try-except으로 감싸져있어서, 가끔 런타임 오류가 사실을 맞을때도, 런타임 오류가 뜨지 않을때가 있다.
        # 런타임 오류 없으면 True
        if last_excute_result[run_id]["execution_status"]["success"]:
            # --- 파일 시스템 일관성 확인 ---:   
            if check_filesystem_consistency(save_dir):
                # 실행 런타임 오류도 없고, 파일시스템도 제대로 돼있으면 0을 리턴하기.
                print("[FILESYSTEM] Consistency check passed ✅")
                return 0
            else:
                print("[FILESYSTEM] Inconsistent result structure ⚠️")
                return 2
        # 
        else:
            print(f"[Runtime Error Occur]")
            return 1  # 런타임 오류
        

    except Exception as e:
        print(f"[EXECUTE ERROR] {e}")
        return 1

def make_history(error_message :str, refactored_code : str) -> str:
    history =  f''' 
    # 재작성된 소스코드
    {refactored_code}

    # 해당 코드의 에러메세지
    {error_message}
    '''
    return history


# ============================================================
# 3️⃣ 실행 및 상태 업데이트 노드
# ============================================================

def code_excute(container: Container) -> Container:
    ''' 
    #### input (container)
    codeAPI
    repairHistroy
    refactored_code
    save_path

    #### output (container 업데이트)
    - container["repairFlag"] : bool
    - container["rewriteFlag"] : int  (0 정상 / 1 런타임 / 2 파일시스템불일치)
    - container["repairHistroy"] : list
    - container["lastExcuteResult"] : dict (result.json 내용)
    '''
    codeAPI = container.get("codeAPI")
    repairHistroy = container.get("repairHistroy", [])
    refactored_code = container.get("refactoredCode")
    save_path = Path(container.get("savePath"))

    # --- 실행코드 수행 ---
    excuteState = make_code_and_excute(codeAPI)

    # --- 상태 갱신 ---
    repairFlag = False
    result_data = {}

    if excuteState == 0:
        # 정상 실행
        try:
            result_path = get_latest_excuted_output_path(save_path)
            result_data = read_json(result_path)
            if not result_data:
                raise ValueError("result.json is empty or unreadable.")
        except Exception as e:
            print(f"[WARN] 정상 실행 후 result.json 읽기 실패: {e}")
            result_data = {}
        repairHistroy = []  # 히스토리 초기화

    else:
        # 런타임 에러 또는 파일시스템 불일치
        error_message = ""  # 초기화
        try:
            result_path = get_latest_excuted_output_path(save_path)
            result_data = read_json(result_path)
            error_message = result_data.get("execution_status", {}).get("error_message", "")
        except (FileNotFoundError, AttributeError, KeyError, TypeError, json.JSONDecodeError):
            # 파일이 없거나 구조가 깨졌을 경우 → 파일시스템 불일치 가능성이 큼
            error_message = "[FileSystemError] result.json could not be read or parsed."
            result_data = {}

        # 히스토리 갱신
        history = make_history(error_message, refactored_code)
        repairHistroy.append(history)
        repairFlag = True

    # --- 컨테이너 업데이트 ---
    container.update({
        "repairFlag": repairFlag,
        "rewriteNodeCode": excuteState,
        "repairHistroy": repairHistroy,
        "lastExcuteResult": result_data
    })

    print(f"[EXECUTE RESULT] rewriteNodeCode={excuteState}, repairFlag={repairFlag}")
    return container