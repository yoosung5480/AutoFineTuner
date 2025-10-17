from AutoHPO.Tool.etc import make_safe_code

def get_anaylsis_prompt(source_code: str, user_requirements: str) -> str:
    """
    #### input
    - source_code: 원본 코드 문자열
    - user_requirements: 사용자의 추가 요구사항 설명

    #### output
    - LLM에 바로 전달 가능한 완성형 문자열 프롬프트

    #### 기능
    - 제공된 소스코드가 '내용상' 머신러닝 또는 딥러닝 학습 코드인지 판별하기 위한 프롬프트를 생성한다.
    - 코드의 실행 가능성은 고려하지 않는다. (실행 여부는 후속 프로세스에서 검증됨)
    - 모델, 학습 루프, 손실 계산, 데이터 로드 등 ML 관련 구성요소가 포함되어 있으면 True로 판단한다.
    """
    source_code = make_safe_code(source_code)
    user_requirements = make_safe_code(user_requirements)
    prompt = f"""
        # 역할
        너는 머신러닝/딥러닝 코드를 판별하는 전문 개발자이다.
        아래의 코드가 **내용적으로 머신러닝 또는 딥러닝 학습을 수행하는 코드인지** 판단하라.
        예시 출력(예시에서는 중괄호 대신 ⟦ ⟧ 표기를 사용. 실제 출력에서는 표준 JSON 중괄호 사용):

        # 판단 기준
        다음의 요소 중 일부라도 포함되면 머신러닝 코드(True)로 판정한다.
        - 모델, 신경망, 또는 ML 알고리즘 객체 정의 (예: model =, nn.Module, LogisticRegression, Trainer 등)
        - 데이터 로드 및 전처리 (pandas.read_csv, Dataset, DataLoader 등)
        - 학습 루프 또는 손실 계산 (for epoch in range, criterion, loss.backward(), optimizer.step() 등)
        - 모델 평가 또는 검증 (accuracy_score, f1_score, evaluate, validation loss 등)
        - 모델 저장, 가중치 저장 (torch.save, joblib.dump, pickle.dump 등)
        - 파라미터 튜닝 또는 파인튜닝 코드 (fit, compile, predict, train 등)

        다음의 경우는 머신러닝 코드(False)가 아니다.
        - 단순 데이터 전처리나 시각화만 수행하는 코드
        - 웹 서버, API, 파일 입출력, OS 유틸리티 스크립트 등 일반 목적 코드
        - 함수나 클래스 정의만 있고, 학습 로직이 없는 코드
        - 파라미터 설정 없이 계산기나 수학 연산만 수행하는 코드

        # 판단 목적
        - 코드의 실행 가능 여부는 고려하지 않는다.
        - 오직 '내용상 머신러닝 관련 코드인가'만 판단하라.

        # 사용자 요구사항 (참고용)
        {user_requirements}

        # 분석 대상 코드
        {source_code}

        # 출력 규칙
        - 오직 JSON 스키마에 맞게 결과를 출력한다.
        - 코드 내용이 머신러닝 관련이면 "isML": true
        - 그렇지 않으면 "isML": false
        - 불확실하거나 애매하면 false로 처리한다.
        """
    return prompt
