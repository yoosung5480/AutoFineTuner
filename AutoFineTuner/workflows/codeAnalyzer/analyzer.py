from AutoFineTuner.tool import codeLauncher
from AutoFineTuner.tool import codeMaker
from AutoFineTuner.tool import codeReader
from AutoFineTuner.tool import llms
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from typing import Annotated, List
from typing_extensions import TypedDict


prompt_to_search_param = PromptTemplate(
    template="""
# 지시사항
modelInfo에서 제안하는 내용과 userPrompt의 내용을 바탕으로 파인튜닝때 적용할 하이퍼파라미터 변수들을 구체화해야한다.

예를들어, 
### userPrompt의 일부
훈련시간의 여유는 충분히있어. 정확도를 우선으로 해서 모델의 성능을 반드시 개선하고싶어.
데이터셋의 크기는 이미지 대략 3600장이야. 이를 고려해서 에포크개수도 실험해줘.

### 예시 답변. 위 텍스트를 기반으로 아래와같이 구체적인 값들로 제안.
learning_rate = [0.01, 0.05, 0.10, 0.2, 0.3] # 시간이 충분함으로, 여러 학습률변수값을 제안.
epochs = [50, 100, 150, 200]    # 에포크값 실험. 데이터셋이 많지 않음으로 에포크를 크게설정.

# 소스코드
{{ sourceCode }}

# 유저프롬프트
{{ userPrompt }}

# Answer
간결하게 요약하고, 근거가 되는 코드 위치(함수/클래스/라인 키워드)를 함께 제시하라.
""",
    input_variables=["modelInfo", "userPrompt"],
    template_format="jinja2",
)

# 하이퍼파라미터 찾기, 선정하기
def searchHyperParam(sourceCode : str, userPrompt : str):
    chain = (
        prompt_to_search_param
        | llms.llm_list["gpt-5"]
        | StrOutputParser()
    )
    param_txt_content = chain.invoke({"sourceCode" : sourceCode,  "userPrompt" : userPrompt})
    return param_txt_content
