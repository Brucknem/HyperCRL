import random
import time

import torch
from multiprocessing import Process
import os
import numpy as np

import hypercrl.tools.default_arg
from hypercrl.hypercl import MLP
from hypercrl.srl import DataCollector, build_vision_model_hnet
from hypercrl.srl.models import ResNet18Encoder
from hypercrl.srl.monitor import MonitorSRL
from hypercrl.srl.tools import augment_model


def train(hparams, lr):
    # print(lr)
    random.seed(123456789)
    np.random.seed(123456789)

    hparams.vision_params.lr_hyper = lr
    mlp = MLP(
        512,
        # hparams.vision_params.h_dims,
        hparams.state_dim,
        [hparams.state_dim] * 4,
        use_batch_norm=True,
        # dropout_rate=0.5,
    )
    encoder_mnet = ResNet18Encoder(mlp, hparams.vision_params)
    trainer_misc = hypercrl.srl.tools.generate_srl_losses(hparams)
    trainer_misc = (
        (trainer_misc[0], torch.optim.Adam(encoder_mnet.parameters(), lr=hparams.vision_params.lr_hyper)),
        trainer_misc[1],
        trainer_misc[2])
    monitor_srl = MonitorSRL(hparams, encoder_mnet, trainer_misc[1][0], trainer_misc[2][0], srl_collector, False)
    hypercrl.srl.tools.train(0, encoder_mnet, None, trainer_misc, monitor_srl, srl_collector, hparams, 0)


if __name__ == "__main__":
    hparams = hypercrl.tools.default_arg.HP(env="door_pose", robot="Panda", vision=True, seed=777, resume=False,
                                            save_folder="./srl/door_pose")
    hparams.model = "hnet"
    hypercrl.tools.default_arg.VisionParams.add_hnet_hparams(hparams.vision_params, hparams.env)
    # hparams.vision_params.load_max = 100

    srl_collector = DataCollector(hparams)
    srl_collector.load()

    train(hparams, 5e-3)

    threads = []
    for lr in [
        1e-3,
        3e-3,
        5e-3,
        # 1e-2,
        # 3e-2
    ]:
        threads.append(Process(target=train, args=(hparams, lr)))

    for thread in threads:
        thread.start()
        time.sleep(1.)

    for thread in threads:
        thread.join()

    print("Yeet")
