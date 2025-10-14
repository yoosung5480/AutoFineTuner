import AutoHPO.Tool.read_write as rw

from AutoHPO.Instance.container  import Container
from AutoHPO.Tool.llms import llm_list
from AutoHPO.PromptBuilder import get_api_prompt

from pydantic import BaseModel, Field   
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pathlib import Path
OUTPUT_PYTHON_NAME = "output.py"


class CodeAPI(BaseModel):
    API: dict = Field(..., description="코드 외부에서 지정할수있는 외부 변수들을 딕셔너리로 반환.")
    
code_api_parser = PydanticOutputParser(pydantic_object=CodeAPI)
llm = llm_list["gpt-4.1"]

# Done.
def code_api_write(container : Container) -> Container:
    print("code_api_write")
    '''
    #### input (Container)
    - "refactoredCode" (str) : 원본 소스코드 
    - "sourceCodePath" (Path) : 기존 소스코드의 주소
    - "savePath" (Path) : 코드 부산물 (log파일, result파일) 저장경로 저장경로 **리펙토링된 코드의 저장위치가 아님**
    - "condaEnv" (str) : 파일 실행을 위한 콘다 가상환경 -> result.json으로 들어감
    - "pythonEnv"(Path) : 파이썬 가상환경 -> result.json으로 들어감

    #### output (Container)
    - "codeAPI" (json) : 추후 노드에서 코드 실행을 위한 api를 담은 json파일. 이 파일만으로 코드수행 가능해야함. 코드 실행위한 모든인자는 여기에

    #### useCase
    1. 유저의 요구사항과, 재생성된 코드를 넣고, 재생성된 코드에서 지정한 arguments들을 가져온다.
    2. 해당 args들을 이용해서 api.json으로 저장한다.
    '''
    # 데이터 받아오기
    refactored_code = container.get("refactoredCode")
    conda_env =  container.get("condaEnv")
    python_env = container.get("pythonEnv")
    source_code_path = container.get("sourceCodePath")
    save_path = container.get("savePath")

    # 프롬프트 작성
    excute_api_prompt = get_api_prompt(refactored_code=refactored_code)

    # llm에 응답요청
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise code generator. Output strictly follows the provided JSON schema."),
        ("human", excute_api_prompt),
        ("human", "{format_instructions}")
    ]).partial(format_instructions=code_api_parser.get_format_instructions())
    chain = prompt | llm | code_api_parser
    result = chain.invoke({})

    # 코드 실행을 위한 매개변수 추출.
    arguments = result.model_dump()["API"]

    # 내부 알고리즘으로 해결할수있는 부분은 그냥 채워넣기.
    # arguments는 받아온 내용 기반으로 채워넣기. container에는 굳이 따로 더 저장하지 않는다.
    script_path = source_code_path.parent / OUTPUT_PYTHON_NAME
    code_api = {
        "env": {
            "CUDA_VISIBLE_DEVICES": None,
            "CONDA_ENV": conda_env,              
            "PYTHON_ENV_PATH": str(python_env), 
            "USE_CONDA": True,                       
            "script": str(script_path),
            "arguments": arguments
        }
    }
    api_output_save_path = save_path / "api.json"

    # 해당 json파일은 "outut/api.json"으로 전달
    print("arguments :", arguments)
    rw.write_json(save_path=api_output_save_path, content=code_api)
    container.update({
        "codeAPI" : code_api
    })
    
    return container
