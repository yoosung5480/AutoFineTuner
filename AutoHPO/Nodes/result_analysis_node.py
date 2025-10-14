import AutoHPO.Nodes as Nodes 
from pydantic import BaseModel, Field   
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field

from pathlib import Path
import os

import AutoHPO.Tool.read_write as rw
from AutoHPO.Instance.container import Container
from AutoHPO.Tool.llms import llm_list
from AutoHPO.Tool.etc import make_safe_code, make_dict_safe_str
from AutoHPO.PromptBuilder import get_evaluate_result_promt, get_search_next_train_params_prompt




def process_last_result(last_excute_result: dict, evalutaion : str) -> str:
    """
    #### input
    last_excute_result : dict
        최근 훈련 실행 결과 (result.json 구조)
    evalutaion : str
        해당 훈련에 대한 평가

    #### output
    processed_str : str
        LLM이 이해하기 쉬운 형태로 요약된 문자열
    """
    if not isinstance(last_excute_result, dict):
        return "Invalid result format."

    try:
        # 첫 번째 키 (실행시간)
        run_id = list(last_excute_result.keys())[0]
        data = last_excute_result[run_id]

        val_score = data.get("validation_score")
        train_score = data.get("train_score")
        params = data.get("params", {})
        runtime_info = data.get("runtime_info", {})
        status = data.get("execution_status", {})

        # 과적합 여부
        if val_score is not None and train_score is not None:
            gap = round(train_score - val_score, 4)
            gap_desc = f"Overfitting Gap: {gap:+.4f}"
        else:
            gap_desc = "Overfitting Gap: N/A"

        # 파라미터 요약 문자열
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])

        # 상태
        success = status.get("success", False)
        error_msg = status.get("error_message") or "None"
        status_str = "success" if success else f"failed ({error_msg})"

        # 실행 시간
        elapsed = runtime_info.get("elapsed_time_sec", "N/A")

        # 최종 문자열
        processed = (
            f"[{run_id}]\n"
            f"- Validation Score: {val_score}\n"
            f"- Train Score: {train_score}\n"
            f"- {gap_desc}\n"
            f"- Params: {params_str}\n"
            f"- Runtime: {elapsed} sec\n"
            f"- Status: {status_str}\n"
            f"- Evaluation : {evalutaion}\n"
        )

        return processed.strip()

    except Exception as e:
        return f"[process_last_result] Error: {e}"




def create_code_api_parser(arguments: dict):
    """
    #### 기능
    주어진 arguments(dict)의 구조를 자동으로 반영하여
    LLM이 해당 딕셔너리와 동일한 형태의 JSON을 반환하도록 제약하는 Pydantic 모델과 Parser를 생성한다.

    #### input
    arguments : dict
        container['codeAPI']["env"]["arguments"] 형태의 동적 실행 인자 딕셔너리

    #### output
    code_api_parser : PydanticOutputParser
        LLM이 해당 구조에 맞게 결과를 반환하도록 하는 파서
    """
    # 1️. 예시 포맷을 자동 생성
    example_lines = []
    for k, v in arguments.items():
        v_type = type(v).__name__
        example_lines.append(f'    "{k}": {repr(v)}  # type: {v_type}')
    example_body = ",\n".join(example_lines)

    # 2. description 자동 구성
    description = (
        "이 필드는 코드 실행에 필요한 외부 인자(API) 구조를 정의합니다.\n"
        "LLM은 반드시 아래 예시와 동일한 key/value 구조의 JSON을 반환해야 합니다.\n\n"
        "예시 출력:\n"
        "{\n"
        '  "API": {\n'
        f"{example_body}\n"
        "  }\n"
        "}\n\n"
        "출력 규칙:\n"
        "1. 최상위 키는 반드시 'API' 여야 한다.\n"
        "2. 'API' 내부의 모든 key는 위 예시와 동일해야 한다.\n"
        "3. 각 값은 예시와 동일한 타입을 유지하되, 합리적 기본값 또는 실제 실행값으로 채워야 한다.\n"
        "4. 마크다운, 설명문, 코드펜스는 절대 포함하지 말고 오직 JSON만 출력할 것.\n"
    )

    # 3. 동적 Pydantic 모델 생성
    class CodeAPI(BaseModel):
        hyperParams : dict = Field(..., description=description)

    # 4. Parser 생성
    code_api_parser = PydanticOutputParser(pydantic_object=CodeAPI)
    return code_api_parser


def _make_finetuning_history_str(finetuning_history: list[str]) -> str:
    finetuning_history_str = ""
    for i, history in enumerate(finetuning_history):
        temp = f"---{i+1}번째 실행 결과----\n" + history
        finetuning_history_str = finetuning_history_str + temp
    return finetuning_history_str

#####################################################################
class FinetuingManager:
    def __init__(self):
        self.llm_evaluator = llm_list["gpt-5"]
        self.llm_HPO_selector = llm_list["gpt-5"]
        

    def evaluate_last_result(self, container : Container):
        '''
        #### input (Container)
        - lastExcuteResult : 가장 최신 코드 실행결과 result.json
        - userRequirements: Annotated[str, "사용자의 요구사항"]
        - finetuningHistory (list): "현재까지의 파인튜닝 히스토리, 선정된 하이퍼 파라미터와, 훈련결과, 분석내용으로 구성돼있다."

        #### output (Container)
        - evalutate (str) : 마지막 훈련코드 결과에 대한 평가 및 방향성

        #### useCase
        - 첫번째 수행이면, finetuningHistory의 훈련점수를 0으로 설정해놓는다.
        1. 이번 훈련결과를 finetuningHistory에 업데이트한다.
        2. finetuningHistory에서 최적의 파라미터 조합인 BestParams를 업데이트한다. (훈련 점수 기반으로 점수 비교로 선택)
        3. finetuningHistory, userRequirements 을 문맥으로 사용해서,  lastExcuteResult에 대한 평가와 앞을 훈련 방향성을 llm에게 받는다.
        4. finetuningHistory에 현재 훈련결과와, llm에게 받은 피드백을 저장한다.
        '''
        last_excute_result = container.get("lastExcuteResult")
        user_requirements = container.get("userRequirements")
        finetuning_history = container.get("finetuningHistory")
        finetuning_history_str = _make_finetuning_history_str(finetuning_history=finetuning_history)
        last_excute_result_str = make_dict_safe_str(last_excute_result)

        # 현재 훈련결과 프롬프트에 전달
        evaluate_result_promt = get_evaluate_result_promt(last_excute_result_str, user_requirements, finetuning_history_str)
        # 현재 훈련결과 평가내용 받기
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 머신러닝 연구자입니다. 해당 훈련결과를 평가하고 파인튜닝 방향에 대해서 제시해야합니다."),
            ("human", evaluate_result_promt),
        ])
        
        chain = prompt | self.llm_evaluator | StrOutputParser()
        result = chain.invoke({})

        # 히스토리 전처리 및 추가
        history_str = process_last_result(last_excute_result, result)
        finetuning_history.append(history_str)
        container.update({
            "evalutate" : result,
            "finetuningHistory" : finetuning_history
            })
        print("[DEBUG]=========이번 훈련 평가 결과============")
        print(result)
        return container
        
    
    def search_next_train_params(self, container : Container):
        '''
        #### input (Container)
        - codeAPI : 리펙토링 된 코드를 실행하기위한 모든 필요한 인자가 들어있는 딕셔너리
        - userRequirements: Annotated[str, "사용자의 요구사항"]
        - finetuningHistory (list): "현재까지의 파인튜닝 히스토리, 선정된 하이퍼 파라미터와, 훈련결과, 분석내용으로 구성돼있다."

        #### output (Container 업데이트)
        - codeAPI (dict) : 마지막 훈련코드 결과에 대한 평가 및 방향성

        #### useCase
        (- 앞서 make_api_for_next_train_excute를 실행해서,  가장 최신의 훈련 결과와 그 피드백도 이제 finetuningHistory에 모두다 들어있는 상황)
        userRequirements, finetuningHistory 내용을 기반으로 codeAPI["env"]["arguments"] 딕셔너리 형태를 토대로 반환 형태의 힌트를 받고, 
        다음 훈련 코드 실행을 위한 하이퍼 파라미터 값들을 생성한다.
        '''

        ### 입력 파싱
        codeAPI = container.get("codeAPI")
        user_requirements = container.get("userRequirements")
        finetuning_history = container.get("finetuningHistory")
        finetuning_history_str = _make_finetuning_history_str(finetuning_history=finetuning_history)
        args = container['codeAPI']["env"]["arguments"]
        print("[DEBUG]========= 기존 파라미터 =============")
        print(args)

        args_str = make_dict_safe_str(args)
    
        ### 맞춤형 파서 준비
        self.code_api_parser = create_code_api_parser(args)

        ### 프롬프트 받기
        search_next_train_params_prompt = get_search_next_train_params_prompt(args_str, user_requirements, finetuning_history_str)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 머신러닝 연구자입니다. 훈련 히스토리와 사용자의 요구사항을 토대로 다음 훈련에서 사용할 하이퍼 파라미터를 지정해야합니다."),
            ("human", search_next_train_params_prompt),
            ("human", "{format_instructions}")
        ]).partial(format_instructions=self.code_api_parser.get_format_instructions())
        chain = prompt | self.llm_evaluator | self.code_api_parser
        result = chain.invoke({})

        ### 하이퍼 파라미터 선정결과 받기
        new_args = result.model_dump()["hyperParams"]

        ### container에 api_json에 다시 로드하기
        codeAPI["env"]["arguments"] = new_args
        print("[DEBUG]========= 다음 훈련 적용 파라미터 =============")
        print(new_args)
        container.update(
           {"codeAPI" : codeAPI}
        )
        return container
    

finetuingManager = FinetuingManager()


#####################################################################
# 실질적 두뇌역할을 하는 llm 노드체인.
# 실질적 두뇌역할을 하는 llm 노드체인.
def result_analysis(container : Container) -> Container:  
    ''' 
    #### input (Container)
    codeAPI : 리펙토링 된 코드를 실행하기위한 모든 필요한 인자가 들어있는 딕셔너리
    lastExcuteResult : 가장 최근에 실행된 코드수행결과
    userRequirements: Annotated[str, "사용자의 요구사항"]
    finetuningHistory (list): "현재까지의 파인튜닝 히스토리, 선정된 하이퍼 파라미터와, 훈련결과, 분석내용으로 구성돼있다."
    maxFineTuningTries : Annotated[int, "파인튜닝 최대 코드 수행횟수 "]
    fineTuningTrieNum : Annotated[int, "현재까지 파인튜닝 실행 횟수"]

    #### output (Container)
    - codeAPI : 리펙토링 된 코드를 실행하기위한 모든 필요한 인자가 들어있는 딕셔너리

    #### useCase
    ''' 
    maxFineTuningTries = container.get("maxFineTuningTries")
    fineTuningTrieNum = container.get("fineTuningTrieNum")
    
    ## end_condition 확인 (횟수초과, 시간초과)
    if fineTuningTrieNum < maxFineTuningTries:
        fineTuningTrieNum += 1

    container = finetuingManager.evaluate_last_result(container)
    container = finetuingManager.search_next_train_params(container)
    container.update({
        "fineTuningTrieNum" : fineTuningTrieNum
    })
    return container