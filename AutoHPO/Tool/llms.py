
''' 
**모델 스펙**

- 링크: https://platform.openai.com/docs/models

| Model               | Input (1M) | Cached Input (1M) | Output (1M) | Context Window | Max Output Tokens | Knowledge Cutoff |
|---------------------|------------|-------------------|-------------|----------------|-------------------|------------------|
| gpt-4.1             | $2.00      | $0.50             | $8.00       | 1,047,576      | 32,768            | Jun 01, 2024     |
| gpt-4.1-mini        | $0.40      | $0.10             | $1.60       | 1,047,576      | 32,768            | Jun 01, 2024     |
| gpt-4.1-nano        | $0.10      | $0.025            | $0.40       | 1,047,576      | 32,768            | Jun 01, 2024     |
| gpt-4o              | $2.50      | $1.25             | $10.00      | 128,000        | 16,384            | Oct 01, 2023     |
| gpt-4o-mini         | $0.15      | $0.075            | $0.60       | 128,000        | 16,384            | Oct 01, 2023     |
| o1                  | $15.00     | $7.50             | $60.00      | 128,000        | 65,536            | Oct 01, 2023     |
| o1-mini             | $1.10      | $0.55             | $4.40       | 128,000        | 65,536            | Oct 01, 2023     |
| o1-pro              | $150.00    | –                 | $600.00     | 128,000        | 65,536            | Oct 01, 2023     |
| o3-mini             | $1.10      | $0.55             | $4.40       | 200,000        | 100,000           | Oct 01, 2023     |
| gpt-4.5-preview     | $75.00     | $37.50            | $150.00     | –              | –                 | –                |


#### codex모델 사용예시
codex = ChatOpenAI(model="gpt-5-codex",
                   api_key=...,
                    temperature=0,
                    max_retries=2,
                    reasoning={
                            "effort": "medium",  # can be "low", "medium", or "high"
                    }),
messages = [
    (
        "system",
        "You are a helpful developer who writes code for specific tasks",
    ),
    ("human", "write me a simple integer sorting code"),
]
ai_msg = codex[0].invoke(messages)
ai_msg



'''

from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_upstage import ChatUpstage
from openai import OpenAI # openai==1.52.2
from typing import Callable, Dict
from langchain_openai import ChatOpenAI
from collections import defaultdict
from langchain_teddynote import logging
import os
from dotenv import load_dotenv
api_key = os.getenv("UPSTAGE_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")
# load_dotenv() 


# reasoning_effort: Optional[str] = None
#     """Constrains effort on reasoning for reasoning models. For use with the Chat
#     Completions API.

#     Reasoning models only, like OpenAI o1, o3, and o4-mini.

#     Currently supported values are ``'minimal'``, ``'low'``, ``'medium'``, and
#     ``'high'``. Reducing reasoning effort can result in faster responses and fewer
#     tokens used on reasoning in a response.

#     .. versionadded:: 0.2.14
#     """
#     reasoning: Optional[dict[str, Any]] = None
#     """Reasoning parameters for reasoning models, i.e., OpenAI o-series models (o1, o3,
#     o4-mini, etc.). For use with the Responses API.

#     Example:

#     .. code-block:: python

#         reasoning={
#             "effort": "medium",  # can be "low", "medium", or "high"
#             "summary": "auto",  # can be "auto", "concise", or "detailed"
#         }

#     .. versionadded:: 0.3.24

#     """

# coding_llm = ChatOpenAI(model="gpt-5-codex") -> gpt-5-codex는 /v1/responses 전용 모델로 배포된 상태라서 아직 LangChain으론 사용안됨.
coding_llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
    reasoning={"effort": "high"}
)

next_train_decide_llm = ChatOpenAI(
                    model="gpt-5",
                    temperature=1.0,
                    max_retries=2,
                    reasoning={
                            "effort": "high",  # can be "low", "medium", or "high"
                    }
)

llm_list = {
    "gpt-5-codex" :  ChatOpenAI(model="gpt-5-codex",
                                temperature=0,
                                timeout=None,
                                max_retries=2),
    "gpt-5" : ChatOpenAI(model="gpt-5"),
    "gpt-5-nano" : ChatOpenAI(model="gpt-5-nano"),
    "gpt-5-mini" : ChatOpenAI(model="gpt-5-mini"),
    "gpt-4o" : ChatOpenAI(model="gpt-4o"),
    "gpt-4o-mini" : ChatOpenAI(model="gpt-4o-mini"),
    "gpt-4.1" : ChatOpenAI(model="gpt-4.1"),
    "gpt-4.1-mini" : ChatOpenAI(model="gpt-4.1-mini"),
    "gpt-4.1-nano" :ChatOpenAI(model="gpt-4.1-nano"),
    "o1" : ChatOpenAI(model="o1"),
    "o1-mini" : ChatOpenAI(model="o1-mini"),
    "o1-pro" : ChatOpenAI(model="o1-pro"),
    "o3-mini" : ChatOpenAI(model="o3-mini"),
    "gpt-4.5-preview" : ChatOpenAI(model="gpt-4.5-preview"), 
    "solar-pro" : ChatUpstage(model="solar-pro"),
    "solar-pro2" : ChatUpstage(model="solar-pro2"),
    "solar-mini" : ChatUpstage(model="solar-mini"),
}

# response = llm_list["gpt-4.1-nano"].invoke("나는 누구인가?")
# print(response)