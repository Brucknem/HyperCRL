import random
import time
from pathlib import Path

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


def train(hparams, use_bn=False, latent_dim=512, lr_hyper=5e-3, lr_forward=1e-3, lr_inverse=1e-3):
    # print(lr)
    reset_seed(hparams.seed)

    hparams.vision_params.lr_hyper = lr_hyper
    hparams.vision_params.inverse_model_lr = lr_inverse
    hparams.vision_params.forward_model_lr = lr_forward

    hparams.state_dim = latent_dim
    hparams.out_dim = latent_dim
    hparams.vision_params.h_dims = [latent_dim] * 4
    if hparams.vision_params.add_sin_cos_to_state:
        hparams.vision_params.forward_model_dims = [latent_dim * 3] * 4
    else:
        hparams.vision_params.forward_model_dims = [latent_dim] * 4
    hparams.vision_params.inverse_model_dims = [latent_dim] * 4
    hparams.vision_params.use_batch_norm = use_bn

    mnet_params = f'.mnet_{lr_hyper}_{str(hparams.vision_params.h_dims)}_bn_{hparams.vision_params.use_batch_norm}'
    forward_params = f'.forward_{lr_forward}_{str(hparams.vision_params.forward_model_dims)}_bn_{True}'
    inverse_params = f'.inverse_{lr_inverse}_{str(hparams.vision_params.inverse_model_dims)}_bn_{True}'

    folder = f'ldim_{latent_dim}' + mnet_params
    folder = folder + forward_params if hparams.vision_params.use_forward_model else folder
    folder = folder + inverse_params if hparams.vision_params.use_inverse_model else folder
    folder = folder.replace(" ", "")

    hparams.save_folder = Path(save_folder) / folder

    networks = generate_srl_networks(hparams, action_space=([-1.] * 7, [1.] * 7))
    optimizer = generate_optimizer(networks)

    monitor_srl = MonitorSRL(hparams, networks, srl_collector, False)
    hypercrl.srl.tools.train(task_id, networks, optimizer, monitor_srl, srl_collector, hparams, task_id)


if __name__ == "__main__":
    save_folder = "./srl/door_pose/only_priors/"
    hparams = hypercrl.tools.default_arg.HP(env="door_pose", robot="Panda", vision=True, seed=777, resume=False,
                                            save_folder=save_folder)
    hparams.model = "gt"
    VisionParams.add_hnet_hparams(hparams.vision_params, hparams.env)
    # hparams.vision_params.load_max = 100

    srl_collector = DataCollector(hparams)
    srl_collector.load()

    # train(hparams, 5e-3)

    lr_hypers = [1e-2, 1e-3, 1e-4, 1e-5]
    # lr_hypers = [1e-3]
    lr_forwards = [1e-3]
    lr_inverses = [1e-4]
    # lr_inverses = [1e-3]

    while True:
        reset_seed(int(time.time()))
        lr_hyper = np.random.uniform(1e-2, 1e-5)
        lr_forward = np.random.uniform(1e-2, 1e-5)
        lr_inverse = np.random.uniform(1e-2, 1e-5)
        latent_dim = np.random.choice([128, 256, 512, 1024, 2048])
        use_bn = False

        train(hparams, use_bn, latent_dim, lr_hyper, lr_forward, lr_inverse)
