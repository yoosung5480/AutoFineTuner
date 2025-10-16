import AutoHPO.Nodes as Nodes 
from AutoHPO.Instance.container import Container
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START


def route_after_exec(container : Container) -> str:
    rewriteNodeCode = container.get("rewriteNodeCode")
    repairNum = container.get("repairNum")
    repairMaxTries = container.get("repairMaxTries")
    print("[DEBUG]=================route_after_exec===============")
    print("최대 리페어 제한 횟수 : ", repairMaxTries)
    print("현재까지 리페어 횟수 : ", repairNum)
    if repairNum <  repairMaxTries:
        if rewriteNodeCode == 0:
            return "result_analysis"
        elif rewriteNodeCode == 1 or rewriteNodeCode == 2:
            if rewriteNodeCode == 1:
                print("런타임 에러, 재작성")
            elif rewriteNodeCode == 2:
                print("파일 시스템 불일치, 재작성")
            return "code_rewrite"
        elif rewriteNodeCode == -1:
            print("리페터 횟수추가로 인한 시스템 종료.")
            return END
    else:
        print("리페어 횟수초과")
        return END



def route_after_analyisis(container : Container):
    max_finetuning_tries = container.get("maxFineTuningTries")
    finetuning_try_num = container.get("fineTuningTrieNum")

    print("[DEBUG]=================route_after_analyisis===============")
    print("최대 파인튜닝 제한 횟수 : ", max_finetuning_tries)
    print("현재까지 파인튜닝 횟수 : ", finetuning_try_num)
    if finetuning_try_num < max_finetuning_tries:
        return "code_excute"
    else:
        return END
    


def get_workflow():
    ##### 노드 정의 ##### 
    workflow = StateGraph(Container)
    workflow.add_node("code_analysis", Nodes.code_analysis)
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
        {"code_rewrite": "code_rewrite", "result_analysis": "result_analysis", END:END},
    )
    workflow.add_conditional_edges(
        "result_analysis",
        route_after_analyisis,
        {END : END, "code_excute": "code_excute"},
    )
    app = workflow.compile()
    return app


class AutoHPO:
    def __init__(self, container : Container):
        self.workflow = get_workflow()
        self.container = container

    def start(self, recursion_limit : int = 25):
        self.workflow.invoke(self.container, {"recursion_limit" : recursion_limit})
        
        


        