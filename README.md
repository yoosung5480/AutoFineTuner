# AutoFineTuner

기존 ML·딥러닝 학습 코드를 반복 실험용으로 재작성하고, 실행 결과를 분석해 다음 하이퍼파라미터를 선택하는 연구 보조 도구다. 여기서 파인튜닝은 기존 ML 모델의 학습 설정·특징·모델 구조를 조정하는 반복 실험을 뜻한다.

## 입력과 처리 흐름

입력은 Python 학습 코드, 실험 요구사항, 실행 환경과 반복 횟수 설정이다. [LangGraph 워크플로우](AutoHPO/Engine/autoHPO.py)는 다음 순서로 구성돼 있다.

1. 코드 분석
2. 조정할 파라미터 탐색
3. 반복 실행을 위한 코드 재작성
4. 실행 인자와 환경을 담은 실행 API 생성
5. 코드 실행 및 결과 파일 검사
6. 결과 분석과 다음 실험 인자 선택

실행 단계에서 런타임 오류나 파일 규약 불일치가 감지되면 코드 재작성으로 분기한다. 복구 및 실험 횟수 설정에 따라 반복을 종료한다.

재작성 코드는 입력 소스와 같은 디렉토리의 `output.py`에 저장된다. 실행별 `result.json`과 로그를 처리하고, 실험 결과를 `experiment.csv`에 누적한다. [결과 분석 코드](AutoHPO/Nodes/result_analysis_node.py)는 `validation_score`가 가장 큰 실험을 선택해 `best/`에 결과를 복사하도록 구현돼 있다.

## 저장소 구조

| 경로 | 내용 |
| --- | --- |
| [AutoFineTuner.py](AutoFineTuner.py) | 실행 진입점 |
| [AutoHPO/Interface/](AutoHPO/Interface/) | 입력 설정과 실험 결과 표시 |
| [AutoHPO/Engine/](AutoHPO/Engine/) | 그래프와 반복·종료 분기 |
| [AutoHPO/Nodes/](AutoHPO/Nodes/) | 분석·재작성·실행·결과 분석 노드 |
| [AutoHPO/PromptBuilder/](AutoHPO/PromptBuilder/) | 단계별 프롬프트와 예시 조립 |
| [AutoHPO/Instance/container.py](AutoHPO/Instance/container.py) | 공통 상태 정의 |
| [AutoHPO/Tool/](AutoHPO/Tool/) | LLM 설정, 파일 처리, subprocess 실행 |
| [target_titanic.py](target_titanic.py), [target_VGGnet.py](target_VGGnet.py), [house_price.py](house_price.py) | 학습 코드 예제 |
| [output.py](output.py) | 저장소에 포함된 재작성 코드 |
| [notebook.ipynb](notebook.ipynb) | 워크플로우 점검 코드와 저장된 실행 출력 |

중첩된 [AutoFineTuner/AutoHPO/Nodes/code_rewrite_node.py](AutoFineTuner/AutoHPO/Nodes/code_rewrite_node.py)도 포함돼 있다. 루트 진입점이 불러오는 재작성 노드는 [AutoHPO/Nodes/code_rewrite_node.py](AutoHPO/Nodes/code_rewrite_node.py)다.

## 현재 사용 방식과 준비 사항

실행 진입점은 `AutoFineTuner.py`이며 파일을 불러오면 `main()`이 호출된다. 상대 경로는 저장소 루트 기준이다. [입력 설정](AutoHPO/Interface/interface.py)의 대화형 입력 호출은 주석 처리돼 있고, 현재 설정은 다음과 같다.

- 대상 코드: `./target_titanic.py`
- 실행 Conda 환경: `ML`
- 복구 상한: 3회
- 실험 상한: 30회
- 결과 저장 경로: `./output`

이 값은 실행 설정이며 실제 완료 횟수를 뜻하지 않는다. 대상 코드·환경·요구사항을 바꾸려면 해당 설정을 확인해야 한다.

Titanic 예제는 `data/train.csv`와 `data/test.csv`를 읽는다. 현재 저장소에는 이 데이터 파일이 포함돼 있지 않다.

[환경 로더](AutoHPO/__init__.py)는 저장소 루트의 `.env`를 읽는다. [LLM 설정](AutoHPO/Tool/llms.py)은 `OPENAI_API_KEY`와 `UPSTAGE_API_KEY`를 참조하고 OpenAI·Upstage 클라이언트를 구성한다. API 키는 공개 문서나 저장소에 기록하지 않는다.

코드는 LangGraph, LangChain 관련 패키지, Pydantic, python-dotenv와 데이터 처리·시각화 패키지를 사용하며, Titanic 예제는 PyTorch와 scikit-learn을 사용한다. 의존성 명세와 검증된 버전 조합은 저장소에 없다.

실행은 외부 API 호출과 학습, 파일 변경을 동반한다. [초기화 코드](AutoHPO/Tool/file_system_init.py)는 기존 `output/`과 `output.py`를 `temp/`의 백업 위치로 옮기며, 같은 이름의 이전 백업을 제거한다. 이후 재작성 단계가 새 `output.py`를 기록한다.

## 현재 결과와 제약

코드 분석부터 다음 실험 선택까지의 워크플로우와 실패 유형별 재작성 분기가 구현돼 있다. 노트북에는 결과 파일 검사 통과 출력과 오류 출력이 함께 저장돼 있어 전체 흐름의 성공을 입증하지는 않는다.

현재 저장소에는 실험별 `result.json`, `experiment.csv`, 모델 결과 디렉토리와 자동화 테스트가 없다. 실행 성공과 성능 재현은 검증되지 않았다. 최고 실험 선정은 `validation_score`의 최댓값을 기준으로 하므로 낮을수록 좋은 지표에 대한 범용 지원은 확인되지 않았다.
