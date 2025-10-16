import AutoHPO.Interface.interface as interface
from AutoHPO.Engine.autoHPO import AutoHPO
from AutoHPO.Instance.container import Container
from AutoHPO.Tool.file_system_init import init_filesystem


def main():
    container = interface.get_container_from_user()
    print(container)
    init_filesystem(container.get("savePath"))
    recursion_limit = (container["repairMaxTries"] + container["fineTuningTrieNum"] ) * 2
    auto_hpo = AutoHPO(container=container)
    auto_hpo.start(recursion_limit=recursion_limit)


main()



