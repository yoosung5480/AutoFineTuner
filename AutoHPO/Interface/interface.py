'''
사용자와 상호작용을 맡는 로직코드.
'''
from AutoHPO.Instance.container import Container, make_container
from AutoHPO.Tool.read_write import read_json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
import sys

# 아래 get 함수들의 코딩 프로토콜
# 1. 입력형식에 대한 확인로직, 2. exception 으로 감싸서, 잘못된 입력형식에는 에러반환 3. 사용자에게 input()으로 데이터받기
def get_sourceCodePath() -> Path:
    while True:
        try:
            user_input = input("리팩토링할 소스코드 경로를 입력하세요 (예: ./target.py): ").strip()
            path = Path(user_input)
            if not path.exists():
                raise FileNotFoundError("입력한 파일 경로가 존재하지 않습니다.")
            if not path.suffix == ".py":
                raise ValueError("Python 파일(.py) 경로를 입력해야 합니다.")
            return path
        except Exception as e:
            print(f"[오류] {e}\n다시 입력해주세요.\n")


def get_condaEnv() -> str:
    while True:
        try:
            env_name = input("사용할 Conda 환경 이름을 입력하세요 (예: ml_env): ").strip()
            if not env_name:
                raise ValueError("빈 문자열은 허용되지 않습니다.")
            return env_name
        except Exception as e:
            print(f"[오류] {e}\n다시 입력해주세요.\n")


def get_userRequirements() -> str:
    while True:
        try:
            req = input("사용자의 요구사항을 간단히 입력하세요: ").strip()
            if len(req) < 3:
                raise ValueError("요구사항은 3자 이상 입력해야 합니다.")
            return req
        except Exception as e:
            print(f"[오류] {e}\n다시 입력해주세요.\n")


def get_repairMaxTries() -> int:
    while True:
        try:
            value = int(input("코드 리페어 최대 반복 시도 횟수를 입력하세요 (예: 3): "))
            if value <= 0:
                raise ValueError("0보다 큰 정수를 입력해야 합니다.")
            return value
        except Exception as e:
            print(f"[오류] {e}\n다시 입력해주세요.\n")


def get_maxFineTuningTries() -> int:
    while True:
        try:
            value = int(input("파인튜닝 최대 코드 수행 횟수를 입력하세요 (예: 5): "))
            if value <= 0:
                raise ValueError("0보다 큰 정수를 입력해야 합니다.")
            return value
        except Exception as e:
            print(f"[오류] {e}\n다시 입력해주세요.\n")


def get_savePath() -> Path:
    while True:
        try:
            user_input = input("리팩토링된 코드의 부산물 저장 폴더 이름을 입력하세요 (예: output): ").strip()
            if not user_input:
                raise ValueError("빈 문자열은 허용되지 않습니다.")
            path = Path(user_input)
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception as e:
            print(f"[오류] {e}\n다시 입력해주세요.\n")


# 사용자가 잘못된 입력형식으로 입력해도, try-catch를 통해서 해당작업으로 다시 돌아가게한다.
def get_container_from_user() -> Container:
    '''
    ### 사용자에게 입력받아야할 정보
    sourceCodePath: Annotated[Path, "리펙토링할 소스코드 저장경로"]
    condaEnv: Annotated[str, "사용자가 지정하는 코드 실행을 위한 콘다 가상환경."]
    userRequirements: Annotated[str, "사용자의 요구사항"]
    repairMaxTries : Annotated[int, "코드 리페어 최대 반복 시도횟수"]
    maxFineTuningTries : Annotated[int, "파인튜닝 최대 코드 수행횟수 "]
    savePath: Annotated[Path, "리펙토링된 코드의 부산물 저장경로. 작업디렉토리/{savePath}/{실행시간}/.. 으로 하드코딩될꺼임."] 
    ### 
    '''
    # container = make_container()
    # sourceCodePath = get_sourceCodePath()
    # condaEnv = get_condaEnv()
    # userRequirements = get_userRequirements()
    # repairMaxTries = get_repairMaxTries()
    # maxFineTuningTries = get_maxFineTuningTries()
    # savePath = get_savePath()


    container = make_container()
    sourceCodePath = Path('./target_titanic.py')
    condaEnv = "ML"
    userRequirements = "훈련데이터셋과 검증데이터셋으로 나누는 그 비율도 파라미터 조정과정 일부로 사용해줘. 그리고 데이터셋에서 선택할 특성도 파라미터화해서 하이퍼 파라미터로 활용해. 대신 Pclass, Sex, Age 이 3개의 특성은 필수로 포함하고 나머지는 너가 훈련마다 적절히 선택해줘.  그리고 트랜스포머 모델 layer개수와 head개수도 인자화해서 그 개수를 적절히 나눠가면서 실험해봐. 참고로 훈련 데이터는 총 890개의 행이야. "
    repairMaxTries = 3
    maxFineTuningTries = 30
    savePath = Path("./output")

    container.update({
        "sourceCodePath": sourceCodePath,
        "condaEnv": condaEnv,
        "userRequirements": userRequirements,
        "repairMaxTries": repairMaxTries,
        "maxFineTuningTries": maxFineTuningTries,
        "savePath": savePath
    })
    return container


# 사용자가 잘못된 입력형식으로 입력해도, try-catch를 통해서 해당작업으로 다시 돌아가게한다.
def modify_container_from_user(container: Container) -> Container:
    '''
    ### 사용자에게 재입력 받을 정보
    1. sourceCodePath: Annotated[Path, "리펙토링할 소스코드 저장경로"]
    2. pythonEnv: Annotated[Path, "파이썬 실행 환경. 콘다 가상환경 없을때의 대안으로, 현재 가상환경을 자동으로 읽어와서 저장할거임. (ex. "ML")"]
    3. userRequirements: Annotated[str, "사용자의 요구사항"]
    4. repairMaxTries : Annotated[int, "코드 리페어 최대 반복 시도횟수"]
    5. maxFineTuningTries : Annotated[int, "파인튜닝 최대 코드 수행횟수 "]
    
    각 정보에 대해서, 현재 입력정보를 출력하고 (1). 수정, (2). 다음으로 중에서 고를수있고,
    수정을 누르면 다시 사용자가 입력한깂을 받아오는 함수.
    '''
    fields = [
        ("sourceCodePath", get_sourceCodePath),
        ("condaEnv", get_condaEnv),
        ("userRequirements", get_userRequirements),
        ("repairMaxTries", get_repairMaxTries),
        ("fineTuningTrieNum", get_maxFineTuningTries),
        ("savePath", get_savePath)
    ]

    for key, func in fields:
        print(f"\n현재 {key}: {container[key]}")
        choice = input(f"{key}를 수정하시겠습니까? (y/n): ").strip().lower()
        if choice == "y":
            container[key] = func()
    print("\n모든 수정이 완료되었습니다.")
    return container


def print_make_choice() -> None:
    ''' 
    각 번호별로, 선택 ui 출력문 제공
    1. 입력수정
    2. 현재 입력결과 보기
    3. 다음단계로
    4. 시스템종료
    그외. 잘못된 번호 입력 출력, (1~4 범위의 숫자 출력해야함과, 각 번호가 어떤기능인지 출력.)
    '''
    print("\n===== 선택 옵션 =====")
    print("1. 입력 수정")
    print("2. 현재 입력 결과 보기")
    print("3. 다음 단계로 진행")
    print("4. 시스템 종료")
    print("=====================")


def print_current_cotainer(container: Container) -> None:
    '''
    ### 출력할 필드
    sourceCodePath: Annotated[Path, "리펙토링할 소스코드 저장경로"]
    pythonEnv: Annotated[Path, "파이썬 실행 환경. 콘다 가상환경 없을때의 대안으로, 현재 가상환경을 자동으로 읽어와서 저장할거임. (ex. "ML")"]
    userRequirements: Annotated[str, "사용자의 요구사항"]
    repairMaxTries : Annotated[int, "코드 리페어 최대 반복 시도횟수"]
    maxFineTuningTries : Annotated[int, "파인튜닝 최대 코드 수행횟수 "]
    '''
    print("\n===== 현재 입력 정보 =====")
    for k, v in container.items():
        print(f"{k:20}: {v}")
    print("===========================")


def autoHPO_start_interface():
    '''
    
    '''
    os.system('cls' if os.name == 'nt' else 'clear')
    container = get_container_from_user()
    print_current_cotainer(container=container)
   
    while True:
        print_make_choice()
        try:
            user_choice = int(input("번호를 선택하세요: ").strip())
        except ValueError:
            print("숫자를 입력해야 합니다.")
            continue
        
        match user_choice:
            case 1:  # 수정
                container = modify_container_from_user(container=container)
            case 2:  # 현재 입력값 보기
                print_current_cotainer(container)
            case 3:  # 다음 실행으로 넘어가기
                break
            case 4:  # 종료
                print("Good bye 👋")
                sys.exit(0)
            case _:  # 그 외 입력
                print("잘못된 선택입니다. 1~4 중 하나를 입력해주세요.")
    return container


def analysis_experiment_savefile(save_path: Path):
    '''
    훈련 결과 CSV와 best/result.json 파일을 기반으로
    실험 리더보드를 시각화 및 요약 출력하고, 결과를 저장한다.
    
    저장 항목:
      - best/validation_leaderboard.png
      - best/train_leaderboard.png
      - best/leaderboard_summary.csv
    '''
    experiment_csv_path = save_path / 'experiment.csv'
    best_result_path = save_path / 'best' / 'result.json'
    save_dir = best_result_path.parent  # /best 디렉토리

    # ---------------------- 파일 로드 ----------------------
    try:
        df = pd.read_csv(experiment_csv_path)
    except Exception as e:
        print("[오류] experiment.csv 파일을 읽는 중 오류 발생:", e)
        return

    try:  
        best_result_json = read_json(best_result_path)  
    except Exception as e:
        print("[오류] best/result.json 파일을 읽는 중 오류 발생:", e)
        return

    # ---------------------- 데이터 요약 ----------------------
    print("\n===== 훈련 결과 요약 =====")
    print(f"총 실험 횟수: {len(df)}")
    print("상위 5개 결과:")
    print(df.sort_values("validation_score", ascending=False).head(5))

    # filename 문자열 변환 (스케일 문제 방지)
    df["filename"] = df["filename"].astype(str)

    # ---------------------- Validation Score Leaderboard ----------------------
    top_val = df.sort_values("validation_score", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=top_val,
        x="validation_score",
        y="filename",
        palette="Blues_r",
        orient="h"
    )
    plt.title("🏆 Validation Score Leaderboard (Top 10)")
    plt.xlabel("Validation Score")
    plt.ylabel("Experiment Filename")

    # 각 막대 끝에 수치 표시
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", label_type="edge", padding=3, fontsize=9, color="black")

    plt.tight_layout()
    val_plot_path = save_dir / "validation_leaderboard.png"
    plt.savefig(val_plot_path, dpi=200)
    plt.show()
    print(f"[저장됨] {val_plot_path}")

    # ---------------------- Train Score Leaderboard ----------------------
    top_train = df.sort_values("train_score", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=top_train,
        x="train_score",
        y="filename",
        palette="Greens_r",
        orient="h"
    )
    plt.title("🏋️‍♂️ Train Score Leaderboard (Top 10)")
    plt.xlabel("Train Score")
    plt.ylabel("Experiment Filename")

    # 각 막대 끝에 수치 표시
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", label_type="edge", padding=3, fontsize=9, color="black")

    plt.tight_layout()
    train_plot_path = save_dir / "train_leaderboard.png"
    plt.savefig(train_plot_path, dpi=200)
    plt.show()
    print(f"[저장됨] {train_plot_path}")

    # ---------------------- Leaderboard Summary Table ----------------------
    summary_path = save_dir / "leaderboard_summary.csv"
    merged_top = pd.merge(
        top_val[["filename", "validation_score"]],
        top_train[["filename", "train_score"]],
        on="filename",
        how="outer"
    )
    merged_top.to_csv(summary_path, index=False)
    print(f"[저장됨] {summary_path}")
