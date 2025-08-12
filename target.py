import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier


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
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train_vec, y_train)

val_probs = xgb.predict_proba(X_val_vec)[:, 1]
auc = roc_auc_score(y_val, val_probs)
print(f"Validation AUC: {auc:.4f}")


# test용으로 'paragraph_text'를 'full_text'에 맞게 재명명
test = test.rename(columns={'paragraph_text': 'full_text'})
X_test = test[['title', 'full_text']]

X_test_vec = vectorizer.transform(X_test)

probs = xgb.predict_proba(X_test_vec)[:, 1]

sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
sample_submission['generated'] = probs

sample_submission.to_csv(f'./baseline_submission.csv', index=False)