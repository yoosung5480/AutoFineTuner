'''
get_filesystem_inconsistency_rewrite_prompt를 구현해야한다.


def get_filesystem_inconsistency_rewrite_prompt(
    source_code: str, 
    user_requirements: str, 
    repair_history: list,
    json_skeleton : str,
    save_path:str
) -> str:
'''

def get_filesystem_inconsistency_rewrite_prompt(
    source_code: str, 
    user_requirements: str, 
    repair_history: list,
    json_skeleton : str,
    save_path:str
) -> str:
    '''
    ### 파일시스템 불일치일때 (rewriteNodeCode : 2)
    #### 프롬프트 구조
    전체 프롬프트 = ( 기본 배경 설명(파일시스템 불일치 수정) 및 코드 생성 규칙 ) + ( 이전 코드 수정 히스토리 )
    #### useCase
    1. 코드의 학습/모델 로직은 그대로 유지
    2. 파일 저장(result.json, log) 형식 및 경로 불일치 문제 수정
    '''
    history = ""
    if repair_history:
        for i, hist in enumerate(repair_history, start=1):
            history += f"\n--- {i}번째 코드 수정 히스토리 ---\n{hist}\n"
    else:
        history = "(이전 수정 기록 없음)\n"

    print(" ======================= [DEBUG] : history =======================")
    print(history)

    filesystem_fix_prompt = f"""
    당신은 뛰어난 머신러닝 엔지니어입니다.
    아래 코드는 학습은 정상적으로 실행되었으나,
    **파일 시스템 불일치(FileSystem Inconsistency)** 문제가 발생했습니다.
    즉, result.json 또는 log 파일이 지정된 규격에 맞게 저장되지 않았습니다.

    ## 지시사항
    - 데이터 로드 및 학습/검증 로직은 절대 수정하지 마십시오.
    - 반드시 result.json과 log 파일이 아래 경로에 올바른 이름으로 저장되도록 수정하십시오.
      예: {save_path}/⟦실행시간⟧/result.json, {save_path}/⟦실행시간⟧/⟦실행시간⟧.log
    - result.json에는 반드시 다음 키가 존재해야 합니다:
        ["validation_score", "train_score", "runtime_info", "execution_status", "system_env"]
    ### result.json의 필수인자가 들어간 skeleton 딕셔너리 형태
    {json_skeleton}
    - 로그 파일에는 학습 진행 및 완료 메시지가 포함되어야 합니다.
    - 저장 경로나 파일명 하드코딩 대신, 반드시 Path 객체를 사용하십시오.
    - 출력은 오직 실행 가능한 파이썬 코드여야 하며, 마크다운/설명 금지.

    ## 수정할 소스코드
    {source_code}

    ## 사용자 요구사항
    {user_requirements}

    ## 이전 코드 수정 히스토리
    {history}

    ## 출력 형식
    오직 수정된 실행 가능한 파이썬 코드만 출력하시오. (마크다운/설명 금지)
    """
    return filesystem_fix_prompt