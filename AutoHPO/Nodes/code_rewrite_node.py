import AutoHPO.Tool.read_write as rw

from AutoHPO.Instance.container  import Container
from AutoHPO.Tool.llms import llm_list
from AutoHPO.PromptBuilder import get_code_rewrite_prompt

from pydantic import BaseModel, Field   
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pathlib import Path

class CodeResponse(BaseModel):
    code: str = Field(..., description="실행 가능한 파이썬 코드")

llm = llm_list["gpt-4.1"]
code_parser = PydanticOutputParser(pydantic_object=CodeResponse)

# Done.
def code_rewrite(container : Container) -> Container:
    print("code_rewrite")
    '''
    #### input (Container)
    - "sourceCode" (str) : 원본 소스코드 
    - "paramList" (list[str]) : 인자화할 파인튜닝 파라미터
    - "userRequirements" (str) : 사용자의 추가 요구사항 설명
    - "savePath" (Path) : 코드 부산물 (log파일, result파일) 저장경로 저장경로 **리펙토링된 코드의 저장위치가 아님**
    - "condaEnv" (str) : 파일 실행을 위한 콘다 가상환경 -> result.json으로 들어감
    - "pythonEnv"(str) : 파이썬 가상환경 -> result.json으로 들어감

    - repairMaxTries (int) : "코드 리페어 최대 반복 시도횟수"
    - repairNum (int) : "현재까지 코드 리페어 반복횟수"
    - rewriteNodeCode (int) :  "파일 실행 노드에서, 이 코드를 어떤 모드로 실행할지 지정, {0 : 기본 리펙토링 실행}, {1 : 런타임오류}, {2 : 파일 시스템 불일치}"

    #### output (Container 업데이트)
    - "refactoredCode" : 리펙토링된 소스코드

    #### useCase
    1. 유저의 요구사항과, 코드내용을 넣고과 부산물 저장위치, 콘다 실횅환경, 파이썬 가상환경 로컬주소를 넣고 코드 리펙토링 작성
    ouput/
        - {실행시간}/
            - {실행시간}.log
            - result.json
            - ... 그 외 코드 부산물 (*.pt 파일 등)
    위 파일 경로로 부산물 셍성되게 한다. 그리고 hyperParams의 인자들을 받아서 외부 인자화하게 만든다.
    2. 생성된 파일을 원본 소스코드와 같은 디렉토리 내에, output.py라는 파일로 생성시킨다.
    '''
    source_code = container.get("sourceCode")
    hyper_params = container.get("paramList")
    user_requirements = container.get("userRequirements")
    save_path = container.get("savePath")
    conda_env = container.get("condaEnv")
    python_env = container.get("pythonEnv")
    flag = container.get("rewriteNodeCode")
    repair_history = container.get("repairHistroy")

    repairMaxTries = container.get("repairMaxTries")
    repairNum = container.get("repairNum")

    ## 컨테이너 로직관리
    if repairNum <  repairMaxTries:
        if flag == 0:
            container["repairNum"] = 0
        elif flag == 1 or flag == 2:
            container["repairNum"] += 1

    if repairNum >= repairMaxTries:
        print("리페어 횟수초과")
        container["rewriteNodeCode"] = -1
        return container



    code_rewrite_prompt = get_code_rewrite_prompt(source_code=source_code, 
                                                              hyper_params=hyper_params, 
                                                              user_requirements=user_requirements,
                                                              save_path=str(save_path),
                                                              conda_env=conda_env,
                                                              python_env=str(python_env),
                                                              repair_history=repair_history,
                                                              flag = flag
                                                              )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise code generator. Output strictly follows the provided JSON schema."),
        ("human", code_rewrite_prompt),
        ("human", "{format_instructions}")
    ]).partial(format_instructions=code_parser.get_format_instructions())

    chain = prompt | llm | code_parser
    result = chain.invoke({})
    refactored_code = result.model_dump()["code"]

    source_code_path = container.get("sourceCodePath")
    output_save_path = source_code_path.parent / "output.py"
    rw.write_file(save_path=output_save_path, content=refactored_code) # "/home/jeongyuseong/바탕화면/private/오픈소스경진대회/AutoFineTuner_2/output.py" 
            
    print(refactored_code)
    container.update({
        "refactoredCode": refactored_code,
        "repairNum": repairNum,
        "rewriteNodeCode": flag
    })
    return container
    

