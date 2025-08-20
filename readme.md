
---

# 🚀 AutoFineTuner

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)]()

> **자동화된 AI 모델 파인튜닝 라이브러리**
> Repetitive tasks out, creativity in.

---

## 📌 소개

**AutoFineTuner**는 **AI 모델 파인튜닝 과정을 자동화**하는 오픈소스 라이브러리입니다.
코드 리팩토링 → 오류 수정 → 파라미터 탐색 → 학습 실행 → 결과 분석 과정을 자동화하여
개발자가 **창의적인 문제 정의와 모델 아이디어**에 집중할 수 있도록 돕습니다.

---

## ✨ 주요 기능

* 🔄 **자동 코드 리팩토링** – LLM 기반 코드 분석 및 인자화 처리
* 🛠️ **코드 오류 자동 수정** – 실행 로그를 기반으로 반복적 코드 보정
* ⚡ **파라미터 탐색 및 학습 자동화** – 최적의 하이퍼파라미터 조합 탐색
* 📊 **결과 관리** – JSON 기반 로그 및 메타데이터 자동 기록
* 🧩 **확장성** – 다양한 모델 및 데이터셋 워크플로우에 적용 가능

---

## 🏗️ 프로젝트 구조

```
AutoFineTuner/
 ├── workflows/          # 주요 워크플로우
 │   ├── codeRefactor/   # 코드 리팩토링
 │   ├── codeRepair/     # 코드 오류 수정
 │   ├── codeAnalyzer/   # 결과/파라미터 분석
 │   └── finetuningManager/ # 파인튜닝 전체 관리
 │
 ├── tool/               # 코드 생성, 실행, I/O 유틸리티
 ├── engine/             # API 및 챗 인터페이스
 ├── interface/          # (추후 업데이트 예정)
 └── outputs/            # 결과물 저장
```
---

## 📌 라이브러리 흐름도

<img width="910" height="542" alt="스크린샷 2025-08-20 오후 9 08 48" src="https://github.com/user-attachments/assets/a270ee63-42b6-40d4-afaf-3add8e137f8a" />


---

## ⚙️ 개발환경

* **언어** : Python 3.10+
* **LLM & Frameworks** : OpenAI GPT, Upstage Solar, HuggingFace Transformers, LangChain
* **Core Libraries** : PyTorch, NumPy, Pandas, Matplotlib
* **실험 관리** : Conda, Bash Scripts, JSON 기반 로그
* **버전 관리** : Git/GitHub

---

## 🚀 설치 및 실행

### 1. 환경 준비

```bash
conda create -n AutoFineTuner python=3.10
conda activate AutoFineTuner
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 라이브러리 설치

```bash
pip install -e .
```

### 4. 실행 예시

* **Cookbook Notebook 실행**

  ```bash
  jupyter notebook cookbook/cookbook.ipynb
  ```
* **파이프라인 전체 실행**

  ```bash
  ./scripts/pipeline.sh
  ```

---

## 📊 결과물

* `output.py` : LLM이 리팩토링한 실행 가능 코드
* `outputs/` : 실행 결과 디렉토리

  * `model.pt` : 최고 성능 모델
  * `result.json` : 최근 실행 결과
  * `results.json` : 전체 결과 기록
  * `metadata.json` : 메타데이터 (최적 파라미터 포함)
  * `[run_id]/.log` : 실행 로그

---

## 🤝 기여 방법

AutoFineTuner는 오픈소스로 공개되어 있으며, 누구나 기여할 수 있습니다.

1. 저장소 Fork
2. 새로운 브랜치 생성 (`git checkout -b feature/새기능`)
3. 변경사항 커밋 (`git commit -m '설명 추가'`)
4. Push 후 Pull Request 생성

---

## 📄 라이선스

MIT License 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

💡 AutoFineTuner는 연구자, 학생, 기업 개발자 모두에게 \*\*“AI 개발의 반복 작업을 줄여주는 최고의 동반자”\*\*가 되는 것을 목표로 합니다.

---
