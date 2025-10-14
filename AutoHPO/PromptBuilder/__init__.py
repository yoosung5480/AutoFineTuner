'''
코드가 하는 동작은 명확하다.
각 context에 맞는 프롬프트를 작성하는것이 중요하다.

10/15까지 적용할 코드에서는 각각의 준비된 하드코딩된 프롬프트를 제공하는것으로 마무리할것이다.
중간고사 이후에는 원래 lang-chain의 리트리버를 활용한 RAG or 웹서치 기반의 CRAG 기반의 프롬프트 관리 체인을 생성할것이다.
'''
from AutoHPO.PromptBuilder.code_analysis_prompt import get_anaylsis_prompt
from AutoHPO.PromptBuilder.code_param_search_prompt import get_param_list_prompt
from AutoHPO.PromptBuilder.code_rewrite_prompt import get_code_rewrite_prompt
from AutoHPO.PromptBuilder.code_api_write_prompt import get_api_prompt
from AutoHPO.PromptBuilder.result_analysis_prompt import get_evaluate_result_promt, get_search_next_train_params_prompt
# from AutoHPO.PromptBuilder.code_analysis_prompt import get_anaylsis_prompt
# from AutoHPO.PromptBuilder.code_analysis_prompt import get_anaylsis_prompt
