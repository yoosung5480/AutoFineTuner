import AutoHPO.Tool.read_write as rw

from AutoHPO.Instance.container  import Container
from AutoHPO.Tool.llms import llm_list
from AutoHPO.PromptBuilder import get_param_list_prompt

from pydantic import BaseModel, Field   
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate


class HyperParamListResponse(BaseModel):
    paramList: list[str] = Field(..., description="예시형식 : [\"epochs\", \"learning_rate\", \"drop_out\"] 외부 변수화 하이퍼파라미터 후보리스트")

# llm 선정
llm = llm_list["gpt-4.1-mini"] 
# 파서 생성
param_list_parser = PydanticOutputParser(pydantic_object=HyperParamListResponse)

# Done.
def code_param_search(container : Container) -> Container:
    print("code_param_search")
    '''
    #### input (Container)
    - "sourceCode" : 원본 코드 
    - "userRequirements" : 사용자의 추가 요구사항 설명

    #### output (Container)
    - "paramList" : 저장위치의 실제 코드내용

    #### useCase
    소스코드 내용과 유저의 요구사항에 맞춰서, 파인튜닝할 파라미터 리스트를 선정한다.
    ''' 
    source_code = container.get("sourceCode")
    user_requirements = container.get("userRequirements")

    hyper_param_prompt = get_param_list_prompt(source_code=source_code, user_requirements=user_requirements)
    prompt = ChatPromptTemplate.from_messages([
            ("system", "You are machine learning developer. Output strictly follows the provided JSON schema."),
            ("human", hyper_param_prompt),
            ("human", "{format_instructions}")
        ]).partial(format_instructions=param_list_parser.get_format_instructions())
    chain = prompt | llm | param_list_parser
    result = chain.invoke({})

    param_list = result.model_dump()["paramList"]
    print("param_list : ", param_list)
    container.update({
        "paramList" : param_list
    })
    
    return container