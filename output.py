import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import os
import torch

def main(args):
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    
    # 데이터 로드
    train = pd.read_csv('./datas/train.csv', encoding='utf-8-sig')
    test = pd.read_csv('./datas/test.csv', encoding='utf-8-sig')
    X = train[['title', 'full_text']]
    y = train['generated']
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # TF-IDF 벡터화
    get_title = FunctionTransformer(lambda x: x['title'], validate=False)
    get_text = FunctionTransformer(lambda x: x['full_text'], validate=False)

    vectorizer = FeatureUnion([
        ('title', Pipeline([('selector', get_title),
                            ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=3000))])),
        ('full_text', Pipeline([('selector', get_text), 
                                ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000))])),
    ])

    # 피처 변환
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # 모델 정의
    xgb = XGBClassifier(
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=42,
        tree_method='gpu_hist',  # GPU 가속 사용
        device='cuda'
    )
    
    # 모델 학습
    xgb.fit(X_train_vec, y_train, 
            eval_set=[(X_val_vec, y_val)], 
            early_stopping_rounds=10,
            verbose=False)

    # 평가
    val_probs = xgb.predict_proba(X_val_vec)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    print(f"Validation AUC: {auc:.4f}")

    # 모델 저장
    os.makedirs(args.save_path, exist_ok=True)
    xgb.save_model(f"{args.save_path}/xgboost_model.json")
    
    # 테스트 데이터 처리
    test = test.rename(columns={'paragraph_text': 'full_text'})
    X_test = test[['title', 'full_text']]
    X_test_vec = vectorizer.transform(X_test)
    
    # 예측
    probs = xgb.predict_proba(X_test_vec)[:, 1]
    sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = probs
    
    # 결과 저장
    sample_submission.to_csv(f"{args.save_path}/baseline_submission.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XGBoost Text Classifier')
    
    # 필수 파라미터
    parser.add_argument('--save_path', type=str, default="./outputs",
                        help='모델 및 결과 저장 경로')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='배치 크기')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에포크 수')
    
    # XGBoost 하이퍼파라미터
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='학습률')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='트리 최대 깊이')
    parser.add_argument('--n_estimators', type=int, default=200,
                        help='부스팅 라운드 수')
    parser.add_argument('--subsample', type=float, default=0.8,
                        help='서브샘플 비율')
    parser.add_argument('--colsample_bytree', type=float, default=0.8,
                        help='컬럼 샘플 비율')
    
    args = parser.parse_args()
    
    main(args)