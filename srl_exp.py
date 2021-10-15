import random
import time
from pathlib import Path

import torch
from multiprocessing import Process
import os
import numpy as np
from torch.utils.data import DataLoader

import hypercrl.tools.default_arg
from hypercrl.envs.cl_env import CLEnvHandler
from hypercrl.hypercl import MLP
from hypercrl.srl import DataCollector, build_vision_model_hnet
from hypercrl.srl.default_arg import add_vision_params, default_vision_params_inverse, default_vision_params_forward, \
    default_vision_params_gt, default_vision_params_encoder
from hypercrl.srl.models import ResNet18Encoder
from hypercrl.srl.monitor import MonitorSRL
from hypercrl.srl.tools import generate_srl_networks, generate_optimizer
from hypercrl.tools import reset_seed

task_id = 0


def train(hparams, use_bn=False, latent_dim=512, encoder_depth=4, gt_depth=4, lr_hyper=5e-3, lr_forward=1e-3,
          lr_inverse=1e-3,
          lr_gt=1e-3):
    # print(lr)
    reset_seed(hparams.seed)

    hparams.state_dim = latent_dim
    hparams.vision_params.encoder_model = default_vision_params_encoder(hparams.vision_params, latent_dim=latent_dim,
                                                                        depth=encoder_depth)
    hparams.vision_params.gt_model = default_vision_params_gt(hparams.vision_params, latent_dim=latent_dim,
                                                              depth=gt_depth)
    hparams.vision_params.forward_model = default_vision_params_forward(hparams.vision_params, latent_dim=latent_dim)
    hparams.vision_params.inverse_model = default_vision_params_inverse(hparams.vision_params, latent_dim=latent_dim)

    hparams.vision_params.encoder_model.lr = lr_hyper
    hparams.vision_params.gt_model.lr = lr_gt
    hparams.vision_params.inverse_model.lr = lr_inverse
    hparams.vision_params.forward_model.lr = lr_forward
    hparams.vision_params.encoder_model.bn = use_bn

    folder = f'ldim:{latent_dim}'
    folder = folder + "#ENCODER_" + hparams.vision_params.encoder_model.to_filename()
    folder = folder + "#GT_" + hparams.vision_params.gt_model.to_filename() if hparams.vision_params.use_gt_model else folder
    folder = folder + "#FORWARD_" + hparams.vision_params.forward_model.to_filename() if hparams.vision_params.use_forward_model else folder
    folder = folder + "#INVERSE_" + hparams.vision_params.inverse_model.to_filename() if hparams.vision_params.use_inverse_model else folder
    folder = folder.replace(" ", "")

    hparams.save_folder = Path(save_folder) / folder

    networks = generate_srl_networks(hparams, action_space=([-1.] * 7, [1.] * 7))
    optimizer = generate_optimizer(networks)

    monitor_srl = MonitorSRL(hparams, networks, srl_collector, False)

    # Data Loader
    train_set, _ = srl_collector.get_dataset(task_id)
    train_loader = DataLoader(train_set, batch_size=hparams.vision_params.bs, shuffle=True, drop_last=True,
                              num_workers=hparams.num_ds_worker)
    hypercrl.srl.tools.train(task_id, networks, optimizer, monitor_srl, train_loader, srl_collector, hparams, task_id)


if __name__ == "__main__":
    save_folder = "./srl/door_pose/only_gt/"
    hparams = hypercrl.tools.default_arg.HP(env="door_pose", robot="Panda", seed=777, resume=False,
                                            save_folder=save_folder)
    hparams.model = "single"
    add_vision_params(hparams, "gt")

    srl_collector = DataCollector(hparams)
    srl_collector.load()

    # train(hparams, 5e-3)

    while True:
        reset_seed(int(time.time()))
        lr_hyper = np.random.uniform(1e-2, 1e-5)
        lr_forward = np.random.uniform(1e-2, 1e-5)
        lr_inverse = np.random.uniform(1e-2, 1e-5)
        lr_gt = np.random.uniform(1e-2, 1e-5)
        latent_dim = np.random.choice([512, 1024])
        encoder_depth = np.random.choice(range(4, 9))
        gt_depth = np.random.choice(range(4, 7))
        use_bn = np.random.random() < 0.5

        train(hparams, use_bn=use_bn, latent_dim=latent_dim, lr_hyper=lr_hyper, lr_forward=lr_forward,
              lr_inverse=lr_inverse, lr_gt=lr_gt, gt_depth=gt_depth, encoder_depth=encoder_depth)
