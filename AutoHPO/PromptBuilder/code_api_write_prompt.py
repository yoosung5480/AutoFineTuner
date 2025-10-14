from AutoHPO.Tool.etc import make_safe_code



def get_api_prompt(refactored_code: str) -> str:
    """
    #### input
    - refactored_code: 리팩토링 완료된 코드 전문 (argparse 포함)

    #### output
    - LLM에 전달할 완성형 프롬프트 문자열

    #### 기능
    - 코드 내부에서 argparse, add_argument 등을 탐색하여
      실행 시 외부에서 지정 가능한 모든 매개변수(arguments)를 추출하도록 지시한다.
    - 각 인자의 이름, 타입, 기본값(default)을 포함하여 딕셔너리로 반환한다.
    - 코드 실행 가능성은 고려하지 않고, 코드 내용상 정의된 인자들을 기준으로 판단한다.
    """ 
    refactored_code = make_safe_code(refactored_code)
    prompt = f"""
     # 역할
    너는 Python 코드를 분석하여 코드 실행 시 외부에서 전달 가능한 모든 argparse 인자(argument)를 추출하는 분석가다.

    # 지시사항
    아래의 리팩토링된 파이썬 코드(refactored_code)를 분석하고,
    코드에 정의된 모든 argparse 인자(add_argument)를 찾아 최종적으로 하나의 JSON 딕셔너리로 요약하라.
    예시 출력(예시에서는 중괄호 대신 ⟦ ⟧ 표기를 사용. 실제 출력에서는 표준 JSON 중괄호 사용):

    ## 판단 기준
    - 대상: parser.add_argument("--name", type=..., default=..., action=...) 형태로 선언된 인자
    - 이름: 선행 '--'는 제거하여 저장 (예: --epochs → "epochs")
    - 타입: type 힌트가 있으면 반영. 없으면 default로 추론. 모두 직렬화 가능한 값으로 표현
    - flag: action="store_true"/"store_false"는 1 또는 0으로 표현
    - 경로형 인자(save_dir/save_path/train_path/test_path 등)가 있으면 포함
    - 코드에 존재하지 않는 임의 인자 추가 금지

    ## 출력 형식(중요)
    - 오직 하나의 JSON 딕셔너리만 출력
    - 최상위 키는 반드시 "API"
    - API 값은 ⟦ 인자명": 기본값 ⟧ 형태의 딕셔너리
    - 설명 문장, 마크다운, 코드펜스 출력 금지

    예시 입력(발췌):
      parser = argparse.ArgumentParser()
      parser.add_argument("--epochs", type=int, default=10)
      parser.add_argument("--batch_size", type=int, default=32)
      parser.add_argument("--save_path", type=str, default="./outputs")
      parser.add_argument("--healthcheck", action="store_true")

    
      ⟦
        "API": ⟦
          "epochs": 10,
          "batch_size": 32,
          "save_path": "./outputs",
          "healthcheck": 0
        ⟧
      ⟧

    # 분석 대상 코드
    {refactored_code}


    # 출력 규칙
    - 실제 출력 시에는 표준 JSON 중괄호를 사용하고, 오직 하나의 딕셔너리만 출력한다.
    - 최상위 키는 "API" 여야 하며, 그 값으로 모든 argparse 인자를 포함한다.
    - JSON 문법 오류가 없도록 완전한 구조로 출력한다.
    """
    return prompt