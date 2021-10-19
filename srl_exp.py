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

import tensorboard


def hparams_to_tensorboard_folder(hparams):
    folder = f'{time.time()}#LDIM:{hparams.state_dim}'
    folder = folder + "#ENCODER_" + hparams.vision_params.encoder_model.to_filename()
    folder = folder + "#GT_" + hparams.vision_params.gt_model.to_filename() if hparams.vision_params.use_gt_model else folder
    folder = folder + "#FORWARD_" + hparams.vision_params.forward_model.to_filename() if hparams.vision_params.use_forward_model else folder
    folder = folder + "#INVERSE_" + hparams.vision_params.inverse_model.to_filename() if hparams.vision_params.use_inverse_model else folder
    folder = folder.replace(" ", "")
    return folder


def train(hparams, use_bn: bool = False, dropout: float = -1, latent_dim: int = 512, encoder_h_dim: int = 4096,
          encoder_depth: int = 1, lr_hyper: float = 5e-3, lr_forward: float = 1e-3, lr_inverse: float = 1e-3,
          lr_gt: float = 1e-3):
    # print(lr)
    reset_seed(hparams.seed)

    hparams.state_dim = latent_dim
    hparams.vision_params.encoder_model = default_vision_params_encoder(hparams.vision_params, h_dims=encoder_h_dim,
                                                                        depth=encoder_depth)
    hparams.vision_params.gt_model = default_vision_params_gt(hparams.vision_params, latent_dim=latent_dim,
                                                              depth=encoder_depth)
    hparams.vision_params.forward_model = default_vision_params_forward(hparams.vision_params, latent_dim=latent_dim)
    hparams.vision_params.inverse_model = default_vision_params_inverse(hparams.vision_params, latent_dim=latent_dim)

    hparams.vision_params.encoder_model.lr = lr_hyper
    hparams.vision_params.gt_model.lr = lr_gt
    hparams.vision_params.inverse_model.lr = lr_inverse
    hparams.vision_params.forward_model.lr = lr_forward
    hparams.vision_params.encoder_model.bn = use_bn
    hparams.vision_params.encoder_model.dropout = dropout

    folder = hparams_to_tensorboard_folder(hparams)
    hparams.save_folder = Path(save_folder) / folder

    networks = generate_srl_networks(hparams, action_space=([-1.] * 7, [1.] * 7))
    optimizer = generate_optimizer(networks)

    monitor_srl = MonitorSRL(hparams, networks, srl_collector, False)
    task_id = 0

    # Data Loader
    train_set, _ = srl_collector.get_dataset(task_id)
    train_loader = DataLoader(train_set, batch_size=hparams.vision_params.bs, shuffle=True, drop_last=True,
                              num_workers=hparams.num_ds_worker)
    hypercrl.srl.tools.train(task_id, networks, optimizer, monitor_srl, train_loader, srl_collector, hparams, task_id)

    params_dict = {"use_bn": use_bn, "dropout": dropout, "latent_dim": latent_dim,
                   "encoder_h_dim": float(encoder_h_dim),
                   "gt_depth": encoder_depth, "lr_hyper": lr_hyper, "lr_forward": lr_forward, "lr_inverse": lr_inverse,
                   "lr_gt": lr_gt}

    for task_id, values in enumerate(monitor_srl.val_stats):
        metric_dict = {key: np.min(value) for key, value in values.items() if key != "time"}
        monitor_srl.writer.add_hparams({**{"task_id": task_id}, **params_dict}, metric_dict)


if __name__ == "__main__":
    save_folder = "./srl/door_pose/latent_to_gt/"
    hparams = hypercrl.tools.default_arg.HP(env="door_pose", robot="Panda", seed=777, resume=False,
                                            save_folder=save_folder)
    hparams.model = "single"
    add_vision_params(hparams, "gt")

    srl_collector = DataCollector(hparams)
    srl_collector.load()

    # train(hparams, 5e-3)

    threaded = False

    while True:
        threads = []
        for _ in range(8):
            lr_hyper = 1e-3
            lr_forward = 1e-3
            lr_inverse = 1e-3
            lr_gt = 1e-3
            latent_dim = 2048

            lr_hyper = np.random.uniform(1e-2, 1e-3)
            encoder_h_dim = np.random.choice([2048, 4196, 8192])
            use_bn = True  # np.random.random() < 0.5
            encoder_depth = 1  # np.random.choice([1])
            l = [-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            dropout = -1  # np.random.choice(l, p=[0.5] + [0.5 / (len(l) - 1)] * (len(l) - 1))

            latent_dim = np.random.choice([256, 512, 1024, 2048, 4196])

            lr_gt = np.random.uniform(1e-2, 1e-3)

            if not threaded:
                train(hparams, use_bn=use_bn, latent_dim=latent_dim, lr_hyper=lr_hyper, lr_forward=lr_forward,
                      lr_inverse=lr_inverse, lr_gt=lr_gt, encoder_depth=encoder_depth, encoder_h_dim=encoder_h_dim,
                      dropout=dropout)
            else:
                threads.append(Process(target=train, args=(
                    hparams, use_bn, dropout, latent_dim, encoder_h_dim, encoder_depth, lr_hyper, lr_forward,
                    lr_inverse,
                    lr_gt)))
            # time.sleep(2)

        for thread in threads:
            thread.start()
            time.sleep(2)

        for thread in threads:
            thread.join()
