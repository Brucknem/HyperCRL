import cv2
import numpy as np
import torch
from torch.utils.data import TensorDataset

import hypercrl.dataset.datautil
from hypercrl.hypercl import HyperNetwork
from hypercrl.srl import ResNet18Encoder


class DataCollector:
    """
    Image data collector for the SRL module
    """

    def __init__(self, hparams):
        self.images = {}
        self.actions = {}

        # MASTER_THESIS Remove nexts for better RAM usage
        self.nexts = {}
        self.train_inds = {}
        self.val_inds = {}

        self.next_mode = hparams.dnn_out
        self.normalize_xu = hparams.normalize_xu
        self.env_name = hparams.env

        self.image_dims = (-1, 3, hparams.vision_params.camera_widths, hparams.vision_params.camera_heights)

        self.max_capacity = hparams.vision_params.collector_max_capacity

    def num_tasks(self):
        return len(self.images)

    def add(self, x_t, u, x_tt, task_id):
        # Convert Format
        if isinstance(u, torch.Tensor):
            u = u.detach().cpu().numpy()
        if u.ndim == 1:
            u = u[:, None]
            
        if task_id in self.images:
            self.images[task_id].append(x_t)
            self.actions[task_id].append(u)
            self.nexts[task_id].append(x_tt)
        else:
            self.images[task_id] = [x_t]
            self.actions[task_id] = [u]
            self.nexts[task_id] = [x_tt]
        # Train or val
        is_train = (np.random.random() <= 0.75)

        ind = len(self.images[task_id]) - 1
        if is_train:
            if task_id in self.train_inds:
                self.train_inds[task_id].append(ind)
            else:
                self.train_inds[task_id] = [ind]
        else:
            if task_id in self.val_inds:
                self.val_inds[task_id].append(ind)
            else:
                self.val_inds[task_id] = [ind]

        self.delete_on_max_capacity(task_id)

    def delete_on_max_capacity(self, task_id):
        if self.max_capacity <= 0:
            return

        while len(self.images[task_id]) > self.max_capacity:
            is_train = True if len(self.val_inds[task_id]) <= 0 else (
                False if len(self.train_inds[task_id]) <= 0 else np.random.random() <= 0.75)
            if is_train:
                idx = np.random.randint(0, max(1, len(self.train_inds[task_id]) // 2))
                elem = self.train_inds[task_id][idx]
                del self.train_inds[task_id][idx]
            else:
                idx = np.random.randint(0, max(1, len(self.val_inds[task_id]) // 2))
                elem = self.val_inds[task_id][idx]
                del self.val_inds[task_id][idx]

            self.train_inds[task_id] = list(map(lambda x: x if x < elem else x - 1, self.train_inds[task_id]))
            self.val_inds[task_id] = list(map(lambda x: x if x < elem else x - 1, self.val_inds[task_id]))

            del self.images[task_id][elem]
            del self.actions[task_id][elem]

        # print(len(self.states[task_id]))

    def get_dataset(self, task_id, ds_range=None):
        """
        Return a pytorch dataset of (state, actions, next_state)
        states, actions are normalized to N(0, 1)
        """

        images = torch.FloatTensor(np.hstack(self.images[task_id])).reshape(self.image_dims)
        actions = torch.FloatTensor(np.hstack(self.actions[task_id])).T
        nexts = torch.FloatTensor(np.hstack(self.nexts[task_id])).reshape(self.image_dims)

        train_inds = self.train_inds[task_id]
        val_inds = self.val_inds[task_id]

        if ds_range == "second_half":
            train_inds = train_inds[len(train_inds) // 2:]
        train_set = TensorDataset(images[train_inds], actions[train_inds], nexts[train_inds])
        val_set = TensorDataset(images[val_inds], actions[val_inds], nexts[val_inds])

        return train_set, val_set

    def get_whole_dataset(self, task_id):
        images = torch.FloatTensor(np.hstack(self.images[task_id])).reshape(self.image_dims)
        actions = torch.FloatTensor(np.hstack(self.actions[task_id])).T
        nexts = torch.FloatTensor(np.hstack(self.nexts[task_id])).reshape(self.image_dims)

        train_set = TensorDataset(images, actions, nexts)

        return train_set

    def convert(self, task_id: int, encoder: ResNet18Encoder, hnet: HyperNetwork,
                collector: hypercrl.dataset.datautil.DataCollector, gpuid: str):
        collector.clear()

        train_loader = torch.utils.data.DataLoader(self.get_whole_dataset(task_id), batch_size=64,
                                                   shuffle=True, drop_last=False, num_workers=0)
        encoder.to(gpuid)
        hnet.to(gpuid)

        encoder.eval()
        hnet.eval()

        encoder_weights = hnet.forward(task_id)

        for i, data in enumerate(train_loader):
            states = encoder.forward(data[0].to(gpuid), encoder_weights).detach().cpu().numpy()
            nexts = encoder.forward(data[2].to(gpuid), encoder_weights).detach().cpu().numpy()

            for x_t, u, x_tt in zip(states, data[1], nexts):
                collector.add(x_t, u, x_tt, task_id)
