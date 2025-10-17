import sys
import os
import json
import time
import datetime
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from tqdm import tqdm

# ==========================
# 유틸리티
# ==========================

def _save_model_generic(model: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Prefer pickle for generic object
        with open(path, "wb") as f:
            pickle.dump(model, f)
    except Exception:
        # Fallback: try CatBoost's own saver if available
        try:
            model.save_model(str(path))
        except Exception:
            # Last resort: save repr
            with open(path, "w", encoding="utf-8") as f:
                f.write(repr(model))
    return str(path)


def _emit_json_line(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ==========================
# 2️⃣ 파생 피처 추가 (원본 유지)
# ==========================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["TotalBath"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    return df


# ==========================
# 메인 결과 API (필수)
# ==========================

def autofinetuner_result() -> dict:
    """
    Returns:
        {
            "model_pt_path": str,
            "validation": float,
            "params": dict
        }
    """
    import argparse

    t0 = time.time()
    start_dt = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    # 고정 CLI 인자
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--train_path", type=str, default="house_data/train.csv")
    parser.add_argument("--test_path", type=str, default="house_data/test.csv")
    parser.add_argument("--healthcheck", type=int, choices=[0, 1], default=0)
    # 하이퍼파라미터 (원본 하드코딩 → argparse)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--early_stopping_rounds", type=int, default=200)
    parser.add_argument("--l2_leaf_reg", type=float, default=3.0)
    parser.add_argument("--K", type=int, default=40)

    args, _ = parser.parse_known_args()

    # 실행 디렉토리 설정
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = Path(args.save_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # 로그 파일로 stdout 리다이렉션
    log_path = run_dir / f"{timestamp}.log"
    sys.stdout = open(log_path, "w", encoding="utf-8")

    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Logging to: {log_path}")

    # 시스템 환경 정보 수집
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "ML")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")
    try:
        import torch  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    python_venv = sys.executable or "/home/jeongyuseong/anaconda3/envs/AutoFineTuner/bin/python3.10"

    # 데이터 로드 (원본 유지)
    print("\n==========================")
    print("1️⃣ 데이터 로드")
    print("==========================")
    train_path = args.train_path
    test_path = args.test_path
    if not train_path:
        train_path = "house_data/train.csv"
    if not test_path:
        test_path = "house_data/test.csv"

    print(f"[INFO] train_path = {train_path}")
    print(f"[INFO] test_path  = {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Healthcheck: 데이터/모델 초기화 확인 후 종료
    if args.healthcheck == 1:
        print("[HEALTHCHECK] Data loaded successfully.")
        # 간단한 전처리 체킹
        assert "SalePrice" in train.columns, "SalePrice column missing in train"
        print("READY")
        return {"healthcheck": 1}

    # 타깃 및 기본 전처리 (원본 유지)
    y = np.log1p(train["SalePrice"])  # target in log1p
    train = train.drop(columns=["SalePrice"]).copy()

    # 범주형 컬럼 처리 (NaN -> 문자열)
    cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        train[c] = train[c].fillna("NA").astype(str)
        test[c] = test[c].fillna("NA").astype(str)

    # 파생 피처 추가 (원본 유지)
    print("\n==========================")
    print("2️⃣ 파생 피처 추가")
    print("==========================")
    train = add_features(train)
    test = add_features(test)

    # 이상치 clip (원본 유지)
    print("\n==========================")
    print("3️⃣ 이상치 clip")
    print("==========================")
    for col in ["GrLivArea", "LotArea", "TotalSF"]:
        hi = train[col].quantile(0.995)
        train[col] = train[col].clip(upper=hi)
        test[col] = test[col].clip(upper=hi)
        print(f"[CLIP] {col} upper -> {hi:.4f}")

    # 기본 설정 (원본 유지)
    print("\n==========================")
    print("4️⃣ 기본 설정")
    print("==========================")
    if "Id" in train.columns:
        X = train.drop(columns=["Id"]).copy()
    else:
        X = train.copy()
    if "Id" in test.columns:
        X_test = test.drop(columns=["Id"]).copy()
    else:
        X_test = test.copy()

    # 범주형 인덱스
    cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    print(f"[INFO] #features={X.shape[1]}, #categorical={len(cat_idx)}")

    # 1차 CatBoost 훈련 (Feature Importance 계산용)
    print("\n==========================")
    print("5️⃣ 1차 CatBoost 훈련 (Feature Importance 계산)")
    print("==========================")
    base_model = CatBoostRegressor(
        loss_function="RMSE",
        learning_rate=args.learning_rate,
        depth=args.depth,
        iterations=max(1, int(args.iterations)),
        early_stopping_rounds=max(1, int(args.early_stopping_rounds)),
        verbose=False,
        random_seed=42,
        l2_leaf_reg=args.l2_leaf_reg,
    )

    train_pool_full = Pool(X, y, cat_features=cat_idx)
    base_model.fit(train_pool_full)

    feature_importance = base_model.get_feature_importance(train_pool_full)
    feat_names = np.array(X.columns)
    imp_df = pd.DataFrame({
        'Feature': feat_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    # 상위 K개 피처 선택
    print("\n==========================")
    print("6️⃣ 상위 K개 피처 선택")
    print("==========================")
    K = int(args.K)
    K = max(1, min(K, len(imp_df)))
    top_k_features = imp_df['Feature'].head(K).tolist()
    print(f"✅ 상위 {K}개 피처만 선택되었습니다.")
    print(top_k_features)

    # SelectFromModel 형태로 축소
    print("\n==========================")
    print("7️⃣ 피처 축소")
    print("==========================")
    X_sel = X[top_k_features].copy()
    X_test_sel = X_test[top_k_features].copy()
    cat_idx_sel = [X_sel.columns.get_loc(c) for c in cat_cols if c in X_sel.columns]

    # 최종 모델 학습 (KFold)
    print("\n==========================")
    print("8️⃣ 최종 모델 학습 (KFold)")
    print("==========================")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    params = dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        learning_rate=float(args.learning_rate),
        depth=int(args.depth),
        l2_leaf_reg=float(args.l2_leaf_reg),
        iterations=int(args.iterations),
        random_seed=42,
        early_stopping_rounds=int(args.early_stopping_rounds),
        verbose=200,
    )

    oof_pred = np.zeros(len(X_sel), dtype=float)
    train_pred_full = np.zeros(len(X_sel), dtype=float)
    test_pred = np.zeros(len(X_test_sel), dtype=float)

    last_model = None

    for fold, (tr_idx, va_idx) in enumerate(tqdm(kf.split(X_sel), total=kf.get_n_splits(), desc="K-Fold Progress")):
        print(f"\n🟩 Fold {fold+1}/{kf.get_n_splits()} 시작 ======================")
        X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx_sel)
        valid_pool = Pool(X_va, y_va, cat_features=cat_idx_sel)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        last_model = model

        # Validation prediction
        oof_pred[va_idx] = model.predict(valid_pool)
        # Train subset prediction (for train_score)
        train_pred_full[tr_idx] = model.predict(train_pool)
        # Test prediction aggregation
        test_pool = Pool(X_test_sel, cat_features=cat_idx_sel)
        test_pred += model.predict(test_pool) / kf.n_splits

        fold_rmse = float(np.sqrt(np.mean((oof_pred[va_idx] - y_va.values) ** 2)))
        best_iter = int(model.get_best_iteration()) if hasattr(model, "get_best_iteration") else params["iterations"]
        print(f"✅ Fold {fold+1} 완료, 최적 반복수: {best_iter}, RMSE(log): {fold_rmse:.5f}")

    # 최종 RMSE 및 제출 유사 산출물
    print("\n==========================")
    print("9️⃣ 최종 RMSE 및 제출 파일 저장")
    print("==========================")
    rmse_log = float(np.sqrt(np.mean((oof_pred - y.values) ** 2)))
    train_rmse_log = float(np.sqrt(np.mean((train_pred_full - y.values) ** 2)))
    print(f"\n📊 [Final CV] RMSE(log1p): {rmse_log:.5f}")

    submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": np.expm1(test_pred)
    })
    sub_path = run_dir / "submission_topk.csv"
    submission.to_csv(sub_path, index=False)
    print(f"💾 Saved: {sub_path}")

    # 모델 저장 (마지막 fold 모델)
    model_path = run_dir / "model.pt"
    if last_model is not None:
        saved_model_path = _save_model_generic(last_model, model_path)
    else:
        # 비정상 케이스: 모델 없음
        with open(model_path, "w", encoding="utf-8") as f:
            f.write("No model trained")
        saved_model_path = str(model_path)

    end_dt = datetime.datetime.now()

    # result.json 구성 (스켈레톤 구조 준수)
    result_payload = {
        "실행시간1": {
            "validation_score": rmse_log,
            "train_score": train_rmse_log,
            "params": {
                "learning_rate": float(args.learning_rate),
                "depth": int(args.depth),
                "iterations": int(args.iterations),
                "early_stopping_rounds": int(args.early_stopping_rounds),
                "l2_leaf_reg": float(args.l2_leaf_reg),
                "K": int(K),
                "save_path": str(args.save_dir),
                "healthcheack": int(args.healthcheck),
            },
            "runtime_info": {
                "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_time_sec": float(time.time() - t0),
            },
            "execution_status": {
                "success": True,
                "error_type": None,
                "error_message": None,
            },
            "system_env": {
                "conda_env": conda_env,
                "cuda_visible_devices": cuda_visible,
                "device": device,
                "python_venv": python_venv,
            },
        }
    }

    # 저장
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=4)

    # 표준 출력(JSON 한 줄)
    _emit_json_line(result_payload)

    # 반환 (계약 유지)
    return {
        "model_pt_path": saved_model_path,
        "validation": rmse_log,
        "params": {
            "learning_rate": float(args.learning_rate),
            "depth": int(args.depth),
            "iterations": int(args.iterations),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "l2_leaf_reg": float(args.l2_leaf_reg),
            "K": int(K),
            "save_path": str(args.save_dir),
            "healthcheack": int(args.healthcheck),
        },
    }


if __name__ == "__main__":
    try:
        autofinetuner_result()
    except Exception as e:
        # In case of fatal error before logging setup or after, ensure minimal stderr
        try:
            # Attempt to produce a minimal result.json in default output dir
            timestamp_fallback = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            run_dir_fallback = Path("output") / timestamp_fallback
            run_dir_fallback.mkdir(parents=True, exist_ok=True)
            result_payload = {
                "실행시간1": {
                    "validation_score": None,
                    "train_score": None,
                    "params": {
                        "learning_rate": None,
                        "depth": None,
                        "iterations": None,
                        "early_stopping_rounds": None,
                        "l2_leaf_reg": None,
                        "K": None,
                        "save_path": "output",
                        "healthcheack": 0,
                    },
                    "runtime_info": {
                        "start_time": "<자동 기록>",
                        "end_time": "<자동 기록>",
                        "elapsed_time_sec": None,
                    },
                    "execution_status": {
                        "success": False,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                    "system_env": {
                        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "ML"),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
                        "device": "cpu",
                        "python_venv": sys.executable,
                    },
                }
            }
            with open(run_dir_fallback / "result.json", "w", encoding="utf-8") as f:
                json.dump(result_payload, f, ensure_ascii=False, indent=4)
            print(json.dumps(result_payload, ensure_ascii=False))
        except Exception:
            # give up
            print(json.dumps({"error": str(e)}))
