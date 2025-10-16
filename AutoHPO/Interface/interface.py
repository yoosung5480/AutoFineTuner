'''
사용자와 상호작용을 맡는 로직코드.
'''
from AutoHPO.Instance.container import Container
from pathlib import Path


# 아래 get 함수들의 코딩 프로토콜
# 1. 입력형식에 대한 확인로직, 2. exception 으로 감싸서, 잘못된 입력형식에는 에러반환 3. 사용자에게 input()으로 데이터받기
def get_sourceCodePath() -> Path:
    ...
def get_pythonEnv() -> str:
    ...
def get_userRequirements() -> Path:
    ...
def get_repairMaxTries() -> int:
    ...
def get_maxFineTuningTries() -> int:
    ...

# 사용자가 잘못된 입력형식으로 입력해도, try-catch를 통해서 해당작업으로 다시 돌아가게한다.
def get_container_from_user() -> Container:
    '''
    ### 사용자에게 입력받아야할 정보
    sourceCodePath: Annotated[Path, "리펙토링할 소스코드 저장경로"]
    pythonEnv: Annotated[Path, "파이썬 실행 환경. 콘다 가상환경 없을때의 대안으로, 현재 가상환경을 자동으로 읽어와서 저장할거임. (ex. "ML")"]
    userRequirements: Annotated[str, "사용자의 요구사항"]
    repairMaxTries : Annotated[int, "코드 리페어 최대 반복 시도횟수"]
    maxFineTuningTries : Annotated[int, "파인튜닝 최대 코드 수행횟수 "]

    ### 
    '''
    ...

# 사용자가 잘못된 입력형식으로 입력해도, try-catch를 통해서 해당작업으로 다시 돌아가게한다.
def modify_container_from_user(container : Container)->Container:
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
    
    ...

# 각각의 번호에 대한 출력 인터페이스. 
# 정해진 텍스트 형식(선택 번호별 설명)을 단순히 출력만한다.
def print_make_choice()->None:
    ''' 
    각 번호별로, 선택 ui 출력문 제공
    1. 입력수정
    2. 현재 입력결과 보기
    3. 다음단계로
    4. 시스템종료
    그외. 잘못된 번호 입력 출력, (1~4 범위의 숫자 출력해야함과, 각 번호가 어떤기능인지 출력.)
    '''
    ...

def print_current_cotainer(container : Container) -> None:
    '''
    ### 출력할 필드
    sourceCodePath: Annotated[Path, "리펙토링할 소스코드 저장경로"]
    pythonEnv: Annotated[Path, "파이썬 실행 환경. 콘다 가상환경 없을때의 대안으로, 현재 가상환경을 자동으로 읽어와서 저장할거임. (ex. "ML")"]
    userRequirements: Annotated[str, "사용자의 요구사항"]
    repairMaxTries : Annotated[int, "코드 리페어 최대 반복 시도횟수"]
    maxFineTuningTries : Annotated[int, "파인튜닝 최대 코드 수행횟수 "]
    '''
    ...


def autoHPO_start_interface():
    '''
    
    '''
    # 사용자에게 첫 컨테이너 받아오기
    container = get_container_from_user()
    print_current_cotainer(container=container)
   
    # 사용자에게 입력 전달받기.
    while True:
        # 유저가 해당 정보를
        print_make_choice()
        user_chice = int(input())
        
        match  user_chice:
            case 1: # 수정
                container = modify_container_from_user(container=container)
            case 2: # 현재 입력값 다시보기
                print_current_cotainer(container)
            case 3: # 다음 실행으로 넘어가기
                break
            case 4: # 종료
                print("good bye")
                exit()
            case _: # 그 외 입력
                print("wrong selection number")
    return container

# def print_current_step()->None:
#     ...

# def print_current_experiment_file()->None:
#     ...

