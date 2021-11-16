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
from hypercrl.srl.tools import generate_srl_networks, generate_optimizer, RMSELoss
from hypercrl.tools import reset_seed

import tensorboard


def hparams_to_tensorboard_folder(hparams):
    folder = f'{time.time()}/LDIM:{hparams.state_dim}'
    folder = folder + "/ENCODER_" + hparams.vision_params.encoder_model.to_filename()
    folder = folder + "/GT_" + hparams.vision_params.gt_model.to_filename()
    folder = folder + "/FORWARD_" + hparams.vision_params.forward_model.to_filename() if hparams.vision_params.use_forward_model else folder
    folder = folder + "/INVERSE_" + hparams.vision_params.inverse_model.to_filename() if hparams.vision_params.use_inverse_model else folder
    folder = folder.replace(" ", "")
    return folder


def extend_params_dict(params_dict, other, name):
    for key, value in other.items():
        value = str(value)
        params_dict[f'{name}.{key}'] = value


def train(hparams, latent_dim, encoder, forward, inverse, gt):
    # print(lr)
    reset_seed(hparams.seed)

    hparams.state_dim = latent_dim
    hparams.vision_params.encoder_model = default_vision_params_encoder(**encoder)
    hparams.vision_params.gt_model = default_vision_params_gt(**gt)
    hparams.vision_params.forward_model = default_vision_params_forward(**forward)
    hparams.vision_params.inverse_model = default_vision_params_inverse(**inverse)

    params_dict = {"latent_dim": int(latent_dim)}
    extend_params_dict(params_dict, encoder, "encoder")
    extend_params_dict(params_dict, forward, "forward")
    extend_params_dict(params_dict, inverse, "inverse")
    extend_params_dict(params_dict, gt, "gt")

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

    for task_id, values in enumerate(monitor_srl.val_stats):
        metric_dict = {key: float(np.min(value)) for key, value in values.items() if key != "time"}
        monitor_srl.writer.add_hparams({**{"task_id": task_id}, **params_dict}, metric_dict)


if __name__ == "__main__":
    save_folder = "./srl/door_pose/latent_to_gt_with_normalization/"
    hparams = hypercrl.tools.default_arg.HP(env="door_pose", robot="Panda", seed=777, resume=False,
                                            save_folder=save_folder)
    hparams.model = "single"
    add_vision_params(hparams, "gt")

    srl_collector = DataCollector(hparams)
    srl_collector.load()

    threaded = True
    # threaded = False

    while True:
        threads = []
        for _ in range(8):
            l = [0.3, 0.5]

            latent_dim = np.random.choice([128, 256, 512])
            # latent_dim = np.random.choice([512])

            encoder = {
                "lr": np.random.uniform(1e-2, 1e-3),
                "h_dims": 512 * np.random.choice([1, 2]),
                "bn": True,  # np.random.random() < 0.5,
                "depth": np.random.choice([1, 2, 4, 8]),
                "dropout": np.random.choice([-1, np.random.choice(l)]),
                "reg_str": np.random.choice([0, np.random.uniform(1e-3, 1e-6)])
            }

            gt = {
                "h_dims": latent_dim,  # np.random.choice([latent_dim, 512, 1024, 2048, 4196, 8192]),
                "lr": np.random.uniform(1e-2, 1e-3),
                "depth": 0,  # np.random.choice(range(2)),
                "bn": False,  # np.random.random() < 0.5
                "loss_fn": np.random.choice([torch.nn.L1Loss(), RMSELoss()])
            }

            forward = {
                "h_dims": np.random.choice([latent_dim, 512, 1024, 2048]),
                "lr": np.random.uniform(1e-2, 1e-3),
                "depth": 1,  # np.random.choice(range(3)),
                "bn": np.random.random() < 0.5
            }

            inverse = {
                "h_dims": np.random.choice([latent_dim, 512, 1024, 2048]),
                "lr": np.random.uniform(1e-2, 1e-3),
                "depth": 1,  # np.random.choice(range(3)),
                "bn": np.random.random() < 0.5
            }

            if not threaded:
                train(hparams, latent_dim=latent_dim, encoder=encoder, forward=forward, inverse=inverse, gt=gt)
            else:
                threads.append(Process(target=train, args=(
                    hparams, latent_dim, encoder, forward, inverse, gt)))
            # time.sleep(2)

        for thread in threads:
            thread.start()
            time.sleep(2)

        for thread in threads:
            thread.join()
