import AutoHPO.Tool.read_write as rw

from AutoHPO.Instance.container  import Container
from AutoHPO.Tool.llms import llm_list
from AutoHPO.PromptBuilder import get_anaylsis_prompt

from pydantic import BaseModel, Field   
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate



class CodeAnalysis(BaseModel):
    isML: bool = Field(..., description="코드가 머신러닝 코드가 맞는지 판단. 맞으면 True, 아니면 False 반환")

# 파서지정
code_analysis_parser = PydanticOutputParser(pydantic_object=CodeAnalysis)

# llm 지정.
llm = llm_list["gpt-4.1-mini"]

# Done.
def code_analysis(container : Container) -> Container:
    print("code_analysis")
    '''
    #### input (Container)
    - "sourceCodePath" : 원본 코드 저장위치
    - "userRequirements": 사용자의 추가 요구사항 설명

    #### output (Container)
    - "sourceCode" : 저장위치의 실제 코드내용
    - "isML" : (bool) True : 머신러닝코드.  False : 그 외 목적 코드

    #### useCase
    1. 저장위치에 소스코드 파일 내용 읽어들임.
    2. 유저의 요구사항과, 코드내용을 넣고 해당 코드가 파인튜닝할 머신러닝 코드인지 판별
    '''
     # 기존 값 유지
    source_code_path = container.get("sourceCodePath")
    user_requirements = container.get("userRequirements")

    # 파일 읽기
    source_code = rw.read_file(save_path=source_code_path)

    # 프롬프트 생성
    analysis_prompt = get_anaylsis_prompt(source_code=source_code, user_requirements=user_requirements)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a machine learning developer. Output strictly follows the provided JSON schema."),
        ("human", analysis_prompt),
        ("human", "{format_instructions}")
    ]).partial(format_instructions=code_analysis_parser.get_format_instructions())

    # LLM 호출
    chain = prompt | llm | code_analysis_parser
    result = chain.invoke({})
    isML = result.model_dump()["isML"]
    print("isML :", isML)
    container.update({
        "sourceCode": source_code,
        "isML": isML
    })

    return container





