'''
get_code_rewrite_prompt를 구현해야한다.
해당 코드에 대한 구현 책임만 갖는다.

def get_code_rewrite_prompt(
        source_code: str, 
        hyper_params: list[str], 
        user_requirements: str,
        save_path: str,
        conda_env: str,
        python_env: str,
        repair_history : list,
        flag : int
    ) -> str:
'''

from AutoHPO.Tool.etc import make_safe_code
from AutoHPO.Tool.llms import llm_list
from AutoHPO.PromptBuilder.rewrite_prompt_builders.make_basic_code_rewrite_prompt import get_basic_code_rewrite_prompt
from AutoHPO.PromptBuilder.rewrite_prompt_builders.make_filesystem_inconsistency_rewrite_prompt import get_filesystem_inconsistency_rewrite_prompt
from AutoHPO.PromptBuilder.rewrite_prompt_builders.make_runtime_error_rewrite_prompt import get_runtime_error_rewrite_prompt

import asyncio
import os
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
import json

def get_result_skeleton(hyper_params, save_path, conda_env, python_env): 
    # 기본 스켈레톤 구조
    params_dict = {param: None for param in hyper_params}
    params_dict.update({
        "save_path": save_path,
        "healthcheack": 0
    })
    return {
        "실행시간1": {          
            "validation_score": None,
            "train_score": None,
            "params": params_dict,
            "runtime_info": {       
                "start_time": "<자동 기록>",
                "end_time": "<자동 기록>",
                "elapsed_time_sec": "<자동 계산>"
            },
            "execution_status": {   
                "success": True,
                "error_type": None,
                "error_message": None,
            },
            "system_env": {        
                "conda_env": conda_env,
                "cuda_visible_devices": "auto",
                "device": "cuda",
                "python_venv": python_env
            }
        }
    }


def get_code_rewrite_prompt(
        source_code: str, 
        hyper_params: list[str], 
        user_requirements: str,
        save_path: str,
        conda_env: str,
        python_env: str,
        repair_history : list,
        flag : int
    ) -> str:
    source_code = make_safe_code(source_code)


    json_skeleton = get_result_skeleton(hyper_params, save_path, conda_env, python_env)
    json_skeleton_str = (
        json.dumps(json_skeleton, ensure_ascii=False, indent=4)
        .replace("{", "⟦")
        .replace("}", "⟧")
    )

    if flag == 0 :
        return get_basic_code_rewrite_prompt(
            source_code=source_code,
            hyper_params=hyper_params, 
            user_requirements=user_requirements,
            save_path=save_path,
            json_skeleton=json_skeleton_str
        )
    elif flag == 1 :
        runtime_error_rewrite_prompt = get_runtime_error_rewrite_prompt(source_code=source_code, user_requirements=user_requirements, repair_history=repair_history)
        return make_safe_code(runtime_error_rewrite_prompt)
    elif flag == 2 :
        filesystem_inconsistency_rewrite_prompt = get_filesystem_inconsistency_rewrite_prompt(source_code=source_code, 
                                                           user_requirements=user_requirements, 
                                                           repair_history=repair_history,
                                                           json_skeleton=json_skeleton_str,
                                                           save_path=save_path)
        return make_safe_code(filesystem_inconsistency_rewrite_prompt)
    else :
        print("invalid rewrite_propmt flag number, it must be interger which is in [0, 1, 2] 0:refactor, 1:runtime error, 2:file system inconsistency")
        return ""