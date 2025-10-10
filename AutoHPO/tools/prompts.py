from langchain.prompts import PromptTemplate

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