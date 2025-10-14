from AutoHPO.Tool.etc import make_safe_code




def get_evaluate_result_promt(last_excute_result : str, user_requirements : str, finetuning_history_str : str):
    ''' 
    현재 훈련결과에 대한 분석과 평가를 수행하도록 하는 프롬프트.
    '''
    evaluate_result_promt = f'''
    # 역할
    당신은 머신러닝 파인튜닝 과정의 연구 책임자입니다.
    아래의 정보들을 기반으로 현재 훈련결과를 분석하고, 훈련 방향성을 평가해야 합니다.
    당신의 목표는 "모델의 성능 향상"이며, 과적합/과소적합, 학습률, 배치사이즈, 에포크 수 등의 변화 가능성을 고려해야 합니다.

    # 지시사항
    1. 훈련 및 검증 점수를 보고 과적합/과소적합 여부를 진단하세요.
    2. 사용자의 요구사항을 참고하여, 이번 훈련에서 무엇이 잘 되었고, 무엇을 개선해야 하는지 간단히 기술하세요.
    3. 다음 훈련에서 집중해야 할 주요 파라미터와 조정 방향을 간략히 제시하세요.
    4. 반드시 한 문단으로 요약된 자연어 설명으로만 작성하세요.
    5. JSON, 코드, 마크다운, 인용부호 등은 사용하지 마세요.

    # 입력 정보
    ## (1) 마지막 실행 결과
    ⟦
    {last_excute_result}
    ⟧

    ## (2) 사용자 요구사항
    ⟦
    {user_requirements}
    ⟧

    ## (3) 지금까지의 파인튜닝 히스토리
    ⟦
    {finetuning_history_str}
    ⟧

    # 출력형식
    한 문단 요약문으로 결과를 평가하고, 다음 훈련에서 어떤 방향(예: 에포크 증가, 러닝레이트 조정 등)이 필요할지 기술하라.
    '''
    return evaluate_result_promt

    

def get_search_next_train_params_prompt(args : str, user_requirements : str, finetuning_history_str : str):
    ''' 
    다음 훈련에서 사용할 최적의 파라미터를 결정하기 위한 LLM 프롬프트.
    '''
    search_next_train_params_prompt = f'''
    # 역할
    당신은 머신러닝 파인튜닝을 자동으로 조정하는 AI 연구자입니다.
    당신의 임무는 아래의 정보를 참고하여 "다음 훈련에서 사용할 하이퍼파라미터 값"을 제안하는 것입니다.
    오직 성능 향상을 최우선으로 고려하며, 합리적이고 근거 있는 수치를 제시해야 합니다.

    # 지시사항
    1. 주어진 기존 파라미터(args)를 기반으로, 수정이 필요한 값만 조정하세요.
    2. 과적합/과소적합 판단, 히스토리 트렌드, 사용자 요구사항을 모두 반영하세요.
    3. 모든 출력은 아래 예시 구조를 따라야 하며, 추가 설명은 절대 포함하지 마세요.
    4. 각 key는 기존 args와 동일해야 하며, 값만 수정 가능합니다.
    5. JSON 문법 오류를 일으키지 않도록 하세요.
    6. 오직 하나의 ⟦"hyperParams"⟧ 딕셔너리만 출력하세요.

    # 입력 정보
    ## (1) 현재 코드 실행 파라미터(args)
    ⟦
    {args}
    ⟧

    ## (2) 사용자 요구사항
    ⟦
    {user_requirements}
    ⟧

    ## (3) 지금까지의 파인튜닝 히스토리
    ⟦
    {finetuning_history_str}
    ⟧

    # 출력 예시
    ⟦
    "hyperParams": ⟦
        "epochs": 10,
        "batch_size": 4,
        "lr": 0.0005,
        "test_size": 0.2,
        "save_dir": "./output",
        "healthcheck": 0
    ⟧
    ⟧

    # 출력형식
    오직 위 구조에 맞는 JSON 딕셔너리 형태로만 반환하라.
    설명문, 마크다운, 텍스트 주석, 따옴표 없는 키 등은 절대 포함하지 말라.
    '''
    return search_next_train_params_prompt

