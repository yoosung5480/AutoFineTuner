from AutoHPO.ContextManager.ContextManager import get_code_refactoring_prompt, get_param_list_prompt, get_code_anaylsis_prompt, get_code_excute_api_prompt
from AutoHPO.tools.llms import llm_list

import AutoHPO.tools.read_write as rw


from pathlib import Path
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field   
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate

class CodeResponse(BaseModel):
    code: str = Field(..., description="실행 가능한 파이썬 코드")
    
class HyperParamListResponse(BaseModel):
    paramList: list[str] = Field(..., description="예시형식 : [\"epochs\", \"learning_rate\", \"drop_out\"] 외부 변수화 하이퍼파라미터 후보리스트")

class CodeAnalysis(BaseModel):
    isML: bool = Field(..., description="코드가 머신러닝 코드가 맞는지 판단. 맞으면 True, 아니면 False 반환")


class CodeAPI(BaseModel):
    API: dict = Field(..., description="코드 외부에서 지정할수있는 외부 변수들을 딕셔너리로 반환.")

class CodeRefactor:
    def __init__(self):
        self.llm = llm_list["gpt-4o"]
        self.code_parser = PydanticOutputParser(pydantic_object=CodeResponse)
        self.param_list_parser = PydanticOutputParser(pydantic_object=HyperParamListResponse)
        self.code_analysis_parser = PydanticOutputParser(pydantic_object=CodeAnalysis)
        self.code_api_parser = PydanticOutputParser(pydantic_object=CodeAPI)
        ...
    ## 1. 코드분석 -> 이거 사용자 요구사항대로 머신러닝 코드가 맞나?
    def _analysis_code(self, prompt):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are machine learning developer you have to check if the given code is machine learning code which has to be finetuned or not. Output strictly follows the provided JSON schema."),
            ("human", prompt),
            ("human", "{format_instructions}")
        ]).partial(format_instructions=self.code_analysis_parser.get_format_instructions())


        chain = prompt | self.llm | self.code_analysis_parser
        result = chain.invoke({})
        # print(result.model_dump()["code"])
        return result.model_dump()["isML"]



    ## 2. 코드분석
    def _get_param_list(self, prompt) -> list[str] :
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are machine learning developer. Output strictly follows the provided JSON schema."),
            ("human", prompt),
            ("human", "{format_instructions}")
        ]).partial(format_instructions=self.param_list_parser.get_format_instructions())
        chain = prompt | self.llm | self.param_list_parser
        result = chain.invoke({})
        return result.model_dump()["paramList"]
    


    ## 3. 코드재작성
    def _rewrite_code(self, prompt) :
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise code generator. Output strictly follows the provided JSON schema."),
            ("human", prompt),
            ("human", "{format_instructions}")
        ]).partial(format_instructions=self.code_parser.get_format_instructions())


        chain = prompt | self.llm | self.code_parser
        result = chain.invoke({})
        # print(result.model_dump()["code"])
        return result.model_dump()["code"]
    
    ## 4. 코드 사용 api 가져오기.
    def _get_excute_api(self, prompt) :
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise code generator. Output strictly follows the provided JSON schema."),
            ("human", prompt),
            ("human", "{format_instructions}")
        ]).partial(format_instructions=self.code_api_parser.get_format_instructions())


        chain = prompt | self.llm | self.code_api_parser
        result = chain.invoke({})
        return result.model_dump()["API"]


    def invoke(self,
               user_requirements : str,
               source_code_path : Path,
               conda_env : str,
               python_env : Path,
               save_path : Path,
               ) -> dict:
        ''' 
        #### input
        - user_requirements : "해당 코드를 파인튜닝 할꺼야. batch_size는 64로 고정해줘."
        - source_code_path : "/home/jeongyuseong/바탕화면/private/오픈소스경진대회/AutoFineTuner_2/target.py"
        - conda_env : "AI"
        - python_env : Path("/usr/bin/python")
        - save_path : Path("/home/jeongyuseong/바탕화면/private/오픈소스경진대회/AutoFineTuner_2/output")

        #### output
        - source_code (str) : 원본 소스코드이다.
        - refactored_code (str) : 내부에 정의된 형식으로 재가공된 소스코드이다.
        - hyper_params (list) : 이 코드에서 파인튜닝 실험 적용을 위한 하이퍼 파라미터명 리스트이다
        - code_excute_api (dict) : 코드 실행을 위한 api가 json형태로 들어있다. 이 json만으로 코드를 조합해서 실행코드를 생성할수있어야한다.

        #### 기능
        1. 현재 소스코드가 머신러닝 코드인지 판별한다. 
        2. 코드 분석 및 하이퍼 파리미터 분석.
        3. 재생성된 코드에서 args추출 
        '''
        source_code = rw.read_file(save_path=source_code_path)
        # 1. 현재 소스코드가 머신러닝 코드인지 판별한다. 
        analysis_prompt = get_code_anaylsis_prompt(source_code=source_code, user_requirements=user_requirements)
        is_finetunable = self._analysis_code(prompt=analysis_prompt)

        if is_finetunable:
            # 2. 코드 분석 및 하이퍼 파리미터 분석.
            hyper_param_prompt = get_param_list_prompt(source_code=source_code, user_requirements=user_requirements)
            hyper_params = self._get_param_list(prompt=hyper_param_prompt)

            # 3. 코드 재작성
            code_rewrite_prompt = get_code_refactoring_prompt(source_code=source_code, 
                                                              hyper_params=hyper_params, 
                                                              user_requirements=user_requirements,
                                                              save_path=save_path,
                                                              conda_env=conda_env,
                                                              python_env=python_env)
            refactored_code = self._rewrite_code(prompt=code_rewrite_prompt)
            
            # source_code_path의 실행위치와 같은곳에서 반드시 코드가 생성돼야한다.
            output_save_path = source_code_path.parent / "output.py"
            rw.write_file(save_path=output_save_path, content=refactored_code) # "/home/jeongyuseong/바탕화면/private/오픈소스경진대회/AutoFineTuner_2/output.py" 
            
            # 4. 재생성된 코드에서 args추출 
            excute_api_prompt = get_code_excute_api_prompt(refactored_code=refactored_code)
            arguments = self._get_excute_api(prompt=excute_api_prompt)
            code_api = {
                "env": {
                    "CUDA_VISIBLE_DEVICES": None,
                    "CONDA_ENV": conda_env,              
                    "PYTHON_ENV_PATH": str(python_env), 
                    "USE_CONDA": True,                       
                    "script": str(source_code_path),
                    "arguments": arguments
                }
            }
            api_output_save_path = save_path / "api.json"
            rw.write_json(save_path=api_output_save_path, content=code_api)
            return {
                "refactored_code" : refactored_code,
                "source_code" : source_code,
                "hyper_params" : hyper_params,
                "code_excute_api" : code_api
            }
        else :
            print("this code is not finetunalbe you may point wroing file.")
            exit()

