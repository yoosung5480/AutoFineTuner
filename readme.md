
---

# AutoFineTuner 오픈소스 라이브러리

## 프로젝트 개요

AutoFineTuner는 **AI 모델 파인튜닝 과정을 자동화**하기 위한 오픈소스 라이브러리입니다.
개발자가 모델 아이디어를 떠올렸을 때, 번거로운 반복 작업(코드 리팩토링, 파라미터 탐색, 학습 실행, 결과 분석 등)을 자동으로 처리하여 **아이디어 → 실행**까지의 간극을 최소화하는 것을 목표로 합니다.

---

## 개발 배경

인공지능의 강점은 방대한 데이터를 학습하고 문제 해결 방안을 제시하는 데 있지만, **새로운 문제 정의와 창의적 해결법 제시**는 인간 고유의 영역입니다.
하지만 창의적 아이디어를 실제 모델로 구현하기 위해서는 다음과 같은 반복적인 과정이 필요합니다.

1. 학습 코드 리팩토링
2. 훈련 파라미터(에포크, 학습률 등) 탐색
3. 모델 훈련 실행
4. 결과 분석 및 반복

이 과정은 높은 지능보다는 **자동화 가능한 절차적 작업**에 가깝습니다.
AutoFineTuner는 이러한 과정을 자동화하여, 연구자와 개발자가 창의적인 문제 정의와 해결 전략에 집중할 수 있도록 돕습니다.

---

## 개발 목표

* AutoFineTuner는 **아이디어 자체를 창출하지 않습니다.**
* 사용자가 “이 데이터셋을 Ko-RoBERTa 모델에 적용해보자”와 같은 아이디어를 떠올리면, AutoFineTuner는 해당 아이디어를 **즉시 실행 가능한 파인튜닝 워크플로우**로 전환합니다.
* 초기 버전은 실행이 보장된 `target.py` 파일을 기반으로 **코드 리팩토링 + 파라미터 튜닝 자동화**를 제공합니다.
* 향후에는 AI가 데이터셋 상황을 분석하여 **target.py까지 자동 생성**할 수 있는 수준으로 확장할 예정입니다.

---

## 프로젝트 구조

```
ROOT_proj/
    └── AutoFineTuner/ 
        ├── workflows/
        │   ├── codeRepair/       
        │   │   └── repair.py          # Refactor 후 코드 오류 발생 시 자동 수정을 담당
        │   ├── finetuningManager/
        │   │   └── tuning_manager.py  # 전체 파인튜닝 워크플로우 관리 (리팩토링 → 수리 → 파라미터 탐색 → 실행)
        │   ├── codeRefactor/
        │   │   └── refactor.py        # 원본 코드를 리팩토링, 결과 저장 형식 강제화
        │   └── codeAnalyzer/
        │       └── analyzer.py        # 실행 결과 및 파라미터 분석, 최적 설정 탐색
        │
        ├── tool/
        │   ├── codeMaker.py           # LLM 응답을 실행 가능한 코드로 변환
        │   ├── codeReader.py          # 코드 파일 파싱
        │   ├── codeLauncher.py        # Conda 환경에서 코드 실행
        │   ├── file_tools.py          # 기본 파일 I/O 및 유틸리티
        │   ├── json_tools.py          # result/metadata JSON 입출력 지원
        │   ├── llms.py                # LLM 클라이언트 및 API 관리
        │   ├── paths.py               # 프로젝트 전역 경로 관리
        │   └── save_remove.py         # 안전한 파일 제거 유틸리티
        │
        ├── engine/
        │   └── engine.py              # API 및 챗 인터페이스 (초기 버전)
        │
        ├── interface/                 # 추후 확장 예정
        └── __init__.py
```

### 생성물

* `output.py` : LLM이 리팩토링한 실행 가능 학습 코드
* `outputs/` : 실행 결과 저장 디렉토리

  * `model.pt` : 최고 성능 모델
  * `result.json` : 최근 실행 결과
  * `results.json` : 전체 실행 결과 기록
  * `metadata.json` : 실험 메타데이터 (최적 파라미터 포함)
  * `[run_id]/.log` : 실행 로그

### 사용자 편의 기능

* `cookbook.py` : API 활용 예제 코드 모음
* `scripts/` : 배시 스크립트 기반 실행 지원

  * `refactor.sh` : 코드 리팩토링 실행
  * `finetune.sh` : 훈련 자동화 실행
  * `pipeline.sh` : 리팩토링부터 파이프라인 실행까지 일괄 수행

---

## 실행 예시


# 의존성 설치
pip install -r requirements.txt

# 프로젝트 설치
pip install -e .

# 가상환경 생성
conda init
conda create -n AutoFineTuner python=3.10
conda activate AutoFineTuner

# 실행
1. cookbook.ipynb에서 api익히기
2. script/bash 파일실행시키기
    - bash script/pipline.sh
```
---
