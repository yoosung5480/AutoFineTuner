import AutoHPO.Interface.interface as interface
from AutoHPO.Engine.autoHPO import AutoHPO
from AutoHPO.Instance.container import Container
from AutoHPO.Tool.file_system_init import init_filesystem

# from pathlib import Path
# save_path = Path("./output")
# print(save_path.absolute())
# init_filesystem(base_dir=save_path.parent)

def main():
    container = interface.get_container_from_user()
    print(container)
    save_path = container.get("savePath")
    init_filesystem(save_path.parent)
    recursion_limit = (container["repairMaxTries"] + container["maxFineTuningTries"] ) * 5
    print("[DEBUG] recursion_limit :", recursion_limit)
    auto_hpo = AutoHPO(container=container)
    auto_hpo.start(recursion_limit=recursion_limit)
    interface.analysis_experiment_savefile(save_path)

main()




