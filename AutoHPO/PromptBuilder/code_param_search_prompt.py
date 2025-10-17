from AutoHPO.Tool.etc import make_safe_code




def get_param_list_prompt(source_code: str, user_requirements: str) -> str:
    """
    #### input
    - source_code: 원본 코드 문자열
    - user_requirements: 사용자의 추가 요구사항 설명

    #### output
    - LLM에 바로 전달 가능한 완성형 문자열 프롬프트

    #### 기능
    - 주어진 ML 코드(source_code)에서 외부 인자화가 필요한 하이퍼파라미터 후보 리스트를 추출하는 프롬프트를 생성한다.
    - 모델, 데이터 로더, 학습 루프 등에서 '수동 설정된 값(epochs, lr, batch_size 등)'을 찾아내어 리스트 형태로 반환해야 한다.
    """
    source_code = make_safe_code(source_code)
    user_requirements = make_safe_code(user_requirements)
    prompt = f"""
        # 역할
        너는 머신러닝/딥러닝 코드를 분석하여, 하드코딩된 설정값 중에서 외부에서 인자화(argparse 등)하기 적합한 하이퍼파라미터 목록을 제안하는 전문 분석가이다.
        예시 출력(예시에서는 중괄호 대신 ⟦ ⟧ 표기를 사용. 실제 출력에서는 표준 JSON 중괄호 사용):

        # 지시사항
        아래 제공된 sourceCode를 분석하여 **외부 CLI 인자화 또는 하이퍼파라미터 튜닝 대상으로 적합한 변수 이름 리스트**를 도출하라.

        ## 후보 기준
        - 학습 제어 관련 변수: epochs, batch_size, lr, weight_decay 등
        - 모델 구조 관련 변수: hidden_size, dropout, num_layers 등
        - 데이터/전처리 관련 변수: max_len, image_size, augmentations 등
        - 경로/장치/플래그: save_dir, device, use_gpu 등은 제외
        - 주석이나 문자열 안의 값은 무시

        ## 출력 형식
        오직 파이썬 리스트 형태의 문자열로 출력하라. (예: ["epochs", "lr", "batch_size"])
        설명, 마크다운, 코드펜스, JSON 객체 등은 금지.

        ## 사용자 요구사항(선택적 참고)
        {user_requirements}

        # 원본 코드
        {source_code}

        # 출력 규칙
        - 오직 리스트 형태로만 반환
        - 후보가 전혀 없을 경우 빈 리스트 []를 출력
        """
    return prompt