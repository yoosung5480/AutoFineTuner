# AutoFineTuner/engine/engine.py
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

# --- 내부 의존성들 --------------------------------------------------------------
from AutoFineTuner.tool import codeReader, codeMaker
from AutoFineTuner.tool.paths import get_output_paths, get_proj_paths
from AutoFineTuner.workflows.codeRepair.repair import repair_code
from AutoFineTuner.workflows.codeRefactor.refactor import start_refactor
from AutoFineTuner.workflows.finetuningManager.tuning_manager import start_finetuning


# ==============================================================================
# 1) 구성 객체
# ==============================================================================
@dataclass
class EngineConfig:
    proj_path: str
    save_dir: str
    target_name: str = "main.py"
    refactored_name: str = "refactored.py"
    conda_env_name: str = "base"
    user_prompt: str = "기본 프롬프트"
    max_steps: int = 10


# ==============================================================================
# 2) 순서도형 API (함수형)
# ==============================================================================
def api_refactor(config: EngineConfig) -> Dict[str, Any]:
    """
    원본 코드 → 리팩토링 코드 생성 및 검토용 결과 반환
    """
    proj_paths = get_proj_paths(
        proj_root=config.proj_path,
        target=config.target_name,
        refactored=config.refactored_name,
    )
    source = codeReader.read_text_strict(str(proj_paths.target))
    result = start_refactor(
        source_code_content=source,
        target_path=str(proj_paths.target),
        refactored_path=str(proj_paths.refactored),
    )
    return {"proj_paths": proj_paths, "result": result}


def api_repair(config: EngineConfig, refactored_code: Optional[str] = None) -> Dict[str, Any]:
    """
    리팩토링된 코드 실행 가능성 점검 및 자동 수정
    """
    proj_paths = get_proj_paths(
        proj_root=config.proj_path,
        target=config.target_name,
        refactored=config.refactored_name,
    )

    if refactored_code is None:
        refactored_code = codeReader.read_text_strict(str(proj_paths.refactored))

    source = codeReader.read_text_strict(str(proj_paths.target))
    repaired = repair_code(
        source_code_content=source,
        refactored_code_str=refactored_code,
        pyfile=str(proj_paths.refactored),
        conda_env_name=config.conda_env_name,
    )
    return {"proj_paths": proj_paths, "repaired": repaired}


def api_finetune(config: EngineConfig) -> Dict[str, Any]:
    """
    파인튜닝 매니저 전체 실행
    """
    proj_paths = get_proj_paths(
        proj_root=config.proj_path,
        target=config.target_name,
        refactored=config.refactored_name,
    )
    output_paths = get_output_paths(save_dir=config.save_dir)

    # 리팩토링 코드 보장
    if not Path(proj_paths.refactored).exists():
        raise FileNotFoundError(f"[engine] refactored 파일이 없습니다: {proj_paths.refactored}")

    work_state = start_finetuning(
        proj_path=proj_paths.proj_root,
        save_dir=output_paths.save_root,
        target_name=config.target_name,
        refactored_name=config.refactored_name,
        cur_conda_env=config.conda_env_name,
        user_prompt=config.user_prompt,
    )
    return {"proj_paths": proj_paths, "output_paths": output_paths, "workState": work_state}


def api_pipeline(config: EngineConfig, do_refactor=True, do_repair=True, do_finetune=True) -> Dict[str, Any]:
    """
    refactor → repair → finetune 파이프라인 일괄 실행
    """
    proj_paths = get_proj_paths(
        proj_root=config.proj_path,
        target=config.target_name,
        refactored=config.refactored_name,
    )
    output_paths = get_output_paths(save_dir=config.save_dir)

    res: Dict[str, Any] = {"proj_paths": proj_paths, "output_paths": output_paths}

    # 1) 리팩토링
    if do_refactor:
        source = codeReader.read_text_strict(str(proj_paths.target))
        ref = start_refactor(
            source_code_content=source,
            target_path=str(proj_paths.target),
            refactored_path=str(proj_paths.refactored),
        )
        res["refactor"] = ref

    # 2) (선택) 바로 수정 저장
    if do_repair:
        refactored_code = codeReader.read_text_strict(str(proj_paths.refactored))
        rep = repair_code(
            source_code_content=codeReader.read_text_strict(str(proj_paths.target)),
            refactored_code_str=refactored_code,
            pyfile=str(proj_paths.refactored),
            conda_env_name=config.conda_env_name,
        )
        res["repair"] = rep

    # 3) 파인튜닝
    if do_finetune:
        work_state = start_finetuning(
            proj_path=proj_paths.proj_root,
            save_dir=output_paths.save_root,
            target_name=config.target_name,
            refactored_name=config.refactored_name,
            cur_conda_env=config.conda_env_name,
            user_prompt=config.user_prompt,
        )
        res["workState"] = work_state

    return res


# ==============================================================================
# 3) 대화형(채팅) 인터페이스
# ==============================================================================
BANNER = r"""
AutoFineTuner Chat
Commands:
  /refactor            - 리팩토링 실행
  /repair              - 리팩토링 코드 실행 가능성 점검/수정
  /finetune            - 파인튜닝 매니저 실행
  /pipeline            - refactor → repair → finetune 일괄 실행
  /status              - 주요 경로/환경 출력
  /quit                - 종료
Tip:
  일반 텍스트를 입력해도 되지만, 위 슬래시 명령을 추천합니다.
"""

def chat(config: EngineConfig) -> None:
    proj_paths = get_proj_paths(
        proj_root=config.proj_path,
        target=config.target_name,
        refactored=config.refactored_name,
    )
    output_paths = get_output_paths(save_dir=config.save_dir)

    print(BANNER)
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[engine] bye.")
            break

        if not text:
            continue

        if text == "/quit":
            print("[engine] bye.")
            break

        if text == "/status":
            print(f"- proj_root     : {proj_paths.proj_root}")
            print(f"- target        : {proj_paths.target}")
            print(f"- refactored    : {proj_paths.refactored}")
            print(f"- save_root     : {output_paths.save_root}")
            print(f"- conda_env     : {config.conda_env_name}")
            print(f"- user_prompt   : {config.user_prompt}")
            continue

        if text == "/refactor":
            try:
                source = codeReader.read_text_strict(str(proj_paths.target))
                ref = start_refactor(
                    source_code_content=source,
                    target_path=str(proj_paths.target),
                    refactored_path=str(proj_paths.refactored),
                )
                print("[refactor] done:", bool(ref))
            except Exception as e:
                print("[refactor] ERROR:", e)
            continue

        if text == "/repair":
            try:
                if not Path(proj_paths.refactored).exists():
                    print("[repair] refactored 파일이 없습니다. 먼저 /refactor 를 실행하세요.")
                    continue
                refactored_code = codeReader.read_text_strict(str(proj_paths.refactored))
                rep = repair_code(
                    source_code_content=codeReader.read_text_strict(str(proj_paths.target)),
                    refactored_code_str=refactored_code,
                    pyfile=str(proj_paths.refactored),
                    conda_env_name=config.conda_env_name,
                )
                print("[repair] done:", bool(rep))
            except Exception as e:
                print("[repair] ERROR:", e)
            continue

        if text == "/finetune":
            try:
                if not Path(proj_paths.refactored).exists():
                    print("[finetune] refactored 파일이 없습니다. 먼저 /refactor 를 실행하세요.")
                    continue
                ws = start_finetuning(
                    proj_path=proj_paths.proj_root,
                    save_dir=output_paths.save_root,
                    target_name=config.target_name,
                    refactored_name=config.refactored_name,
                    cur_conda_env=config.conda_env_name,
                    user_prompt=config.user_prompt,
                )
                print("[finetune] done. keys:", list(ws.keys()))
            except Exception as e:
                print("[finetune] ERROR:", e)
            continue

        if text == "/pipeline":
            try:
                res = api_pipeline(config, do_refactor=True, do_repair=True, do_finetune=True)
                print("[pipeline] done. keys:", list(res.keys()))
            except Exception as e:
                print("[pipeline] ERROR:", e)
            continue

        # 일반 텍스트는 user_prompt 갱신 후 친절한 가이드
        config.user_prompt = text
        print(f"[engine] user_prompt를 갱신했습니다: {config.user_prompt}")
        print("원하시면 /finetune 또는 /pipeline 을 실행하세요.")


# ==============================================================================
# 4) CLI (python -m AutoFineTuner.engine.engine ...)
# ==============================================================================
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AutoFineTuner Engine CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser):
        sp.add_argument("--proj-path", required=True, help="프로젝트 루트")
        sp.add_argument("--save-dir", required=True, help="출력(결과) 저장 루트")
        sp.add_argument("--target-name", default="main.py", help="원본 실행 파일명")
        sp.add_argument("--refactored-name", default="refactored.py", help="리팩토링 파일명")
        sp.add_argument("--conda-env", default="base", help="실행 conda env 이름")
        sp.add_argument("--user-prompt", default="기본 프롬프트", help="파인튜닝 사용자 프롬프트")
        sp.add_argument("--max-steps", type=int, default=10)

    for name in ("refactor", "repair", "finetune", "pipeline", "chat"):
        sp = sub.add_parser(name)
        common(sp)

    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    p = _build_cli()
    args = p.parse_args(argv)

    config = EngineConfig(
        proj_path=args.proj_path,
        save_dir=args.save_dir,
        target_name=args.target_name,
        refactored_name=args.refactored_name,
        conda_env_name=args.conda_env,
        user_prompt=args.user_prompt,
        max_steps=args.max_steps,
    )

    if args.cmd == "refactor":
        out = api_refactor(config)
        print("[engine] refactor OK.")
        return 0

    if args.cmd == "repair":
        out = api_repair(config)
        print("[engine] repair OK.")
        return 0

    if args.cmd == "finetune":
        out = api_finetune(config)
        print("[engine] finetune OK.")
        return 0

    if args.cmd == "pipeline":
        out = api_pipeline(config, do_refactor=True, do_repair=True, do_finetune=True)
        print("[engine] pipeline OK.")
        return 0

    if args.cmd == "chat":
        chat(config)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
