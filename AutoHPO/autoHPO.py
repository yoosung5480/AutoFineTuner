import AutoHPO.Nodes as Nodes 
from AutoHPO.Instance.container import Container
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START


def route_after_exec(container : Container):
    '''
    code_excute 실행후에 아래를 기준으로 판단한다.
    max_tries를 초과하면, 실패 출력문을 출력하고, 종료한다.

    - 런타임 오류발생시. (return code : 1)
            런타임 오류에 대한 내용을 기반으로 -> code_write를 위한 프롬프트 재작성 및, code_write를 재실행
    - 실행은 됐으나, 파일 시스템과 불일치 발생시 (return code : 2)
        1. 
        Log 파일이 "output/{실행시간}/{실행시간}.log" 위치에 읽히지 않거나, 
        "output/{실행시간}/result.json"이 존재하는지, "알고리즘적으로 확인" 
        존재하면 다음스텝으로 넘어가기. 없으면, inconsistency 발동 -> code_write를 위한 프롬프트 재작성 및, code_write를 재실행
        2. 
        "output/{실행시간}/result.json"이 존재할때, 정해진 형식과 맞는지
        "알고리즘적으로"확인. 형식이 일치하면 다음스텝으로 넘어가기. 
        없으면 inconsistency 발동. -> code_write를 위한 프롬프트 재작성 및, code_write를 재실행
    - 정상수행 됐을때. (return code : 0)
        현재 정상코드가 정상 실행됐음을, container에 전달후 종료.
    '''


def route_after_analyisis(container : Container):
    '''
    내부 종료조건 (최대 파인튜닝 코드 실행횟수, 시간제한)을 만족하면 END 함수로,
    그 외에는 새로 지정된 코드 api를 통해서, 코드 재실행 노드로 간다.
    '''
    

def get_workflow():
    '''

    '''
    ##### 노드 정의 ##### 
    workflow = StateGraph(Container)
    workflow.add_node("code_analysis", Nodes.coxsde_analysis)
    workflow.add_node("code_param_search", Nodes.code_param_search)
    workflow.add_node("code_rewrite", Nodes.code_rewrite)
    workflow.add_node("code_api_write", Nodes.code_api_write)
    workflow.add_node("code_excute", Nodes.code_excute)
    workflow.add_node("result_analysis", Nodes.result_analysis)

    ##### 워크플로우 정의 ##### 
    workflow.add_edge(START, "code_analysis")
    workflow.add_edge("code_analysis", "code_param_search")
    workflow.add_edge("code_param_search", "code_rewrite")
    workflow.add_edge("code_rewrite", "code_api_write")
    workflow.add_edge("code_api_write", "code_excute")
    workflow.add_conditional_edges(
        "code_excute",
        route_after_exec,
        {"code_rewrite": "code_rewrite", "result_analysis": "result_analysis"},
    )
    workflow.add_conditional_edges(
        "result_analysis",
        route_after_analyisis,
        {"END": END, "code_excute": "code_excute"},
    )
    workflow.compile()


class app:
    def __init__(self):
        self.workflow = get_workflow()
        self.container = Container()

    def start_HPO():
        '''
        
        '''

        
        


        