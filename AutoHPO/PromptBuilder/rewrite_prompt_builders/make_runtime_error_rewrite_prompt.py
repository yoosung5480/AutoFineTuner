'''
get_runtime_error_rewrite_prompt을 구현해야한다.

def get_runtime_error_rewrite_prompt(
    source_code: str, 
    user_requirements: str, 
    repair_history: list
) -> str:
'''


def get_runtime_error_rewrite_prompt(
    source_code: str, 
    user_requirements: str, 
    repair_history: list
) -> str:
    ''' 
    ### 런타임 에러일때 (rewriteNodeCode : 1)
    #### 프롬프트 구조
    전체 프롬프트 = ( 배경 설명 및 코드 생성 규칙 ) + ( 이전 코드 수정 히스토리 )
    #### useCase
    1. 런타임 에러 원인을 해결하기 위한 코드 수정 프롬프트 생성
    2. LLM에게 전달 가능한 완전한 문자열 반환
    '''
    # 수정 히스토리 문자열 정리
    history = ""
    if repair_history:
        for i, hist in enumerate(repair_history, start=1):
            history += f"\n--- {i}번째 코드 수정 히스토리 ---\n{hist}\n"
    else:
        history = "(이전 수정 기록 없음)\n"

    runtime_error_repair_prompt = f"""
    당신은 뛰어난 머신러닝 개발자이자 디버깅 전문가입니다.
    아래 코드는 실행 중 **런타임 오류(Runtime Error)** 가 발생하여 수정이 필요한 코드입니다.
    원래 코드의 알고리즘과 데이터 처리 로직은 그대로 유지하되,
    런타임 오류의 원인(변수 참조, 타입 오류, import 오류 등)을 해결하도록 코드를 수정하십시오.

    ## 지시사항
    - 모델 학습/검증 로직의 흐름은 변경하지 마십시오.
    - 파일 저장, result.json, 로그 기록 로직은 변경하지 마십시오.
    - 단순히 런타임 오류를 일으킨 부분만 수정하십시오.
    - 추가적인 import, 예외처리, 경로 생성 등은 필요하다면 허용됩니다.
    - 출력은 반드시 실행 가능한 파이썬 코드만 포함하십시오.

    ## 수정할 소스코드
    {source_code}

    ## 사용자 요구사항
    {user_requirements}

    ## 이전 코드 수정 히스토리
    {history}

    ## 출력 형식
    오직 수정된 실행 가능한 파이썬 코드만 출력하시오. (마크다운/설명 금지)
    """
    return runtime_error_repair_prompt