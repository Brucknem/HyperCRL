import hypercrl.tools.default_arg
from hypercrl.srl import DataCollector

if __name__ == "__main__":
    hparams = hypercrl.tools.default_arg.HP(env="door_pose", robot="Panda", vision=True, seed=777, resume=False,
                                            save_folder="./srl/door_pose")

    srl_collector = DataCollector(hparams)
    srl_collector.load()
    print("Yeet")
