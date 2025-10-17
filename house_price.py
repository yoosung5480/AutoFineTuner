import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================
# 1️⃣ 데이터 로드
# ==========================
train = pd.read_csv("house_data/train.csv")
test  = pd.read_csv("house_data/test.csv")

y = np.log1p(train["SalePrice"])
train.drop(columns=["SalePrice"], inplace=True)

# 범주형 컬럼 NaN → 문자열 처리
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    train[c] = train[c].fillna("NA").astype(str)
    test[c]  = test[c].fillna("NA").astype(str)

# ==========================
# 2️⃣ 파생 피처 추가
# ==========================
def add_features(df):
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["TotalBath"] = (
        df["FullBath"].fillna(0)
        + 0.5*df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5*df["BsmtHalfBath"].fillna(0)
    )
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    return df

train = add_features(train)
test  = add_features(test)

# ==========================
# 3️⃣ 이상치 clip
# ==========================
for col in ["GrLivArea", "LotArea", "TotalSF"]:
    hi = train[col].quantile(0.995)
    train[col] = train[col].clip(upper=hi)
    test[col]  = test[col].clip(upper=hi)

# ==========================
# 4️⃣ 기본 설정
# ==========================
X = train.drop(columns=["Id"])
X_test = test.drop(columns=["Id"])
cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

# ==========================
# 5️⃣ 1차 CatBoost 훈련 (Feature Importance 계산용)
# ==========================
base_model = CatBoostRegressor(
    loss_function="RMSE",
    learning_rate=0.05,
    depth=8,
    iterations=1500,
    early_stopping_rounds=200,
    verbose=False,
    random_seed=42
)

train_pool = Pool(X, y, cat_features=cat_idx)
base_model.fit(train_pool)

# Feature Importance 추출
feature_importance = base_model.get_feature_importance(train_pool)
feat_names = np.array(X.columns)

imp_df = pd.DataFrame({
    'Feature': feat_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)


# ==========================
# 6️⃣ 상위 K개 피처 선택
# ==========================
K = 40  # 👈 원하는 개수로 조정 가능
top_k_features = imp_df['Feature'].head(K).tolist()

print(f"\n✅ 상위 {K}개 피처만 선택되었습니다.")
print(top_k_features)

# ==========================
# 7️⃣ SelectFromModel 형태로 축소
# ==========================
X_sel = X[top_k_features]
X_test_sel = X_test[top_k_features]

# ==========================
# 8️⃣ 최종 모델 학습 (KFold + tqdm)
# ==========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
params = dict(
    loss_function="RMSE",
    eval_metric="RMSE",
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    iterations=3000,
    random_seed=42,
    early_stopping_rounds=200,
    verbose=200
)

oof_pred = np.zeros(len(train))
test_pred = np.zeros(len(test))
cat_idx_sel = [X_sel.columns.get_loc(c) for c in cat_cols if c in X_sel.columns]

for fold, (tr_idx, va_idx) in enumerate(tqdm(kf.split(X_sel), total=kf.get_n_splits(), desc="K-Fold Progress")):
    print(f"\n🟩 Fold {fold+1}/{kf.get_n_splits()} 시작 ======================")
    X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx_sel)
    valid_pool = Pool(X_va, y_va, cat_features=cat_idx_sel)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    oof_pred[va_idx] = model.predict(valid_pool)
    test_pool = Pool(X_test_sel, cat_features=cat_idx_sel)
    test_pred += model.predict(test_pool) / kf.n_splits

    print(f"✅ Fold {fold+1} 완료, 최적 반복수: {model.get_best_iteration()}, RMSE(log): {np.sqrt(np.mean((oof_pred[va_idx]-y_va)**2)):.5f}")

# ==========================
# 9️⃣ 최종 RMSE 및 제출
# ==========================
rmse_log = np.sqrt(np.mean((oof_pred - y.values)**2))
print(f"\n📊 [Final CV] RMSE(log1p): {rmse_log:.5f}")

sub = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": np.expm1(test_pred)
})
sub.to_csv("submission_topk.csv", index=False)
print("💾 Saved: submission_topk.csv")
