import random
import time

import torch
from multiprocessing import Process
import os
import numpy as np

import hypercrl.tools.default_arg
from hypercrl.envs.cl_env import CLEnvHandler
from hypercrl.hypercl import MLP
from hypercrl.srl import DataCollector, build_vision_model_hnet
from hypercrl.srl.models import ResNet18Encoder
from hypercrl.srl.monitor import MonitorSRL
from hypercrl.srl.tools import generate_srl_networks, generate_optimizer
from hypercrl.tools import reset_seed
from torchviz import make_dot

task_id = 0
import itertools


def train(hparams, lr_hyper=5e-3, lr_forward=1e-3, lr_inverse=1e-3):
    # print(lr)
    reset_seed(hparams.seed)

    hparams.vision_params.lr_hyper = lr_hyper
    hparams.vision_params.inverse_model_lr = lr_inverse
    hparams.vision_params.forward_model_lr = lr_forward

    networks = generate_srl_networks(hparams, action_space=([-1.] * 7, [1.] * 7))
    optimizer = generate_optimizer(networks)

    monitor_srl = MonitorSRL(hparams, networks, srl_collector, False)
    hypercrl.srl.tools.train(task_id, networks, optimizer, monitor_srl, srl_collector, hparams, task_id)


if __name__ == "__main__":
    hparams = hypercrl.tools.default_arg.HP(env="door_pose", robot="Panda", vision=True, seed=777, resume=False,
                                            save_folder="./srl/door_pose/inverse_bn_258_8_encoder_/")
    hparams.model = "hnet"
    hypercrl.tools.default_arg.VisionParams.add_hnet_hparams(hparams.vision_params, hparams.env)
    # hparams.vision_params.load_max = 100

    srl_collector = DataCollector(hparams)
    srl_collector.load()

    # train(hparams, 5e-3)

    lr_hypers = [1e-4, 1e-5]
    # lr_hypers = [1e-3]
    lr_forwards = [1e-3]
    lr_inverses = [1e-3, 1e-4, 1e-5]
    # lr_inverses = [1e-3]

    for lr_hyper in lr_hypers:
        for lr_forward in lr_forwards:
            threads = []
            for lr_inverse in lr_inverses:
                train(hparams, lr_hyper, lr_forward, lr_inverse)

                # threads.append(Process(target=train, args=(hparams, lr_hyper, lr_forward, lr_inverse)))
            # 
            # for thread in threads:
            #     thread.start()
            #     time.sleep(1)
            # 
            # for thread in threads:
            #     thread.join()

    print("Yeet")
