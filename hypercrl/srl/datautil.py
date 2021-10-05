from collections import defaultdict

import cv2
import numpy as np
import time
import torch
from torch.utils.data import TensorDataset

import hypercrl.dataset.datautil
from hypercrl.hypercl import HyperNetwork
from hypercrl.srl import ResNet18EncoderHnet

import pathlib
import pandas as pd


class DataCollector:
    """
    Image data collector for the SRL module
    """

    def __init__(self, hparams):
        self.images = {}
        self.actions = {}

        # MASTER_THESIS Remove nexts for better RAM usage
        self.nexts = {}
        self.rewards = {}

        self.train_inds = {}
        self.val_inds = {}

        self.same_actions = {}

        self.next_mode = hparams.dnn_out
        self.normalize_xu = hparams.normalize_xu
        self.env_name = hparams.env

        self.image_dims = (-1, 3, hparams.vision_params.camera_widths, hparams.vision_params.camera_heights)

        self.max_capacity = hparams.vision_params.collector_max_capacity

        self.save_path = hparams.vision_params.save_path
        self.save_every = hparams.vision_params.save_every

        self.load_max = hparams.vision_params.load_max
        self.load_suffix = hparams.vision_params.load_suffix

    def num_tasks(self):
        return len(self.images)

    def add(self, x_t, u, x_tt, r, task_id):
        # Convert Format
        if isinstance(u, torch.Tensor):
            u = u.detach().cpu().numpy()
        if u.ndim == 1:
            u = u[:, None]

        if task_id in self.images:
            self.images[task_id].append(x_t)
            self.actions[task_id].append(u)
            self.nexts[task_id].append(x_tt)
            self.rewards[task_id].append(r)
        else:
            self.images[task_id] = [x_t]
            self.actions[task_id] = [u]
            self.nexts[task_id] = [x_tt]
            self.rewards[task_id] = [r]
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

        if task_id not in self.same_actions:
            self.same_actions[task_id] = {}
        action = tuple(np.squeeze(u))
        if action not in self.same_actions[task_id]:
            self.same_actions[task_id][action] = []
        else:
            print("Yeet")
        self.same_actions[task_id][action].append(ind)

        self.save()

        self.delete_on_max_capacity(task_id)

    def delete_on_max_capacity(self, task_id):
        if self.max_capacity <= 0:
            return

        while len(self.images[task_id]) > self.max_capacity:
            probs = np.array(self.rewards[task_id])
            inds = np.where(probs < np.median(probs))[0]
            probs = probs[inds]
            probs = -probs
            probs = probs - min(probs)
            probs = probs / sum(probs)
            idx = np.random.choice(inds, p=probs)

            if idx in self.train_inds[task_id]:
                self.train_inds[task_id].remove(idx)
            else:
                self.val_inds[task_id].remove(idx)

            self.train_inds[task_id] = list(map(lambda x: x if x < idx else x - 1, self.train_inds[task_id]))
            self.val_inds[task_id] = list(map(lambda x: x if x < idx else x - 1, self.val_inds[task_id]))

            to_del = []
            for key, value in self.same_actions[task_id].items():
                idx in value and value.remove(idx)
                if len(value) == 0:
                    to_del.append(key)
                self.same_actions[task_id][key] = list(map(lambda x: x if x < idx else x - 1, value))

            for key in to_del:
                del self.same_actions[task_id][key]

            del self.images[task_id][idx]
            del self.actions[task_id][idx]
            del self.nexts[task_id][idx]
            del self.rewards[task_id][idx]

        # print(len(self.states[task_id]))

    def get_dataset(self, task_id, ds_range=None):
        """
        Return a pytorch dataset of (state, actions, next_state)
        states, actions are normalized to N(0, 1)
        """

        indices = torch.IntTensor(list(range(len(self.images[task_id]))))
        images = torch.FloatTensor(np.hstack(self.images[task_id])).reshape(self.image_dims)
        actions = torch.FloatTensor(np.hstack(self.actions[task_id])).T
        nexts = torch.FloatTensor(np.hstack(self.nexts[task_id])).reshape(self.image_dims)
        rewards = torch.FloatTensor(np.hstack(self.rewards[task_id])).T

        train_inds = self.train_inds[task_id]
        val_inds = self.val_inds[task_id]

        if ds_range == "second_half":
            train_inds = train_inds[len(train_inds) // 2:]
        train_set = TensorDataset(indices[train_inds], images[train_inds], actions[train_inds], nexts[train_inds],
                                  rewards[train_inds])
        val_set = TensorDataset(indices[val_inds], images[val_inds], actions[val_inds], nexts[val_inds],
                                rewards[val_inds])

        return train_set, val_set

    def get_whole_dataset(self, task_id):
        indices = torch.IntTensor(list(range(len(self.images[task_id]))))
        images = torch.FloatTensor(np.hstack(self.images[task_id])).reshape(self.image_dims)
        actions = torch.FloatTensor(np.hstack(self.actions[task_id])).T
        nexts = torch.FloatTensor(np.hstack(self.nexts[task_id])).reshape(self.image_dims)

        train_set = TensorDataset(indices, images, actions, nexts)

        return train_set

    def get_same_actions(self, task_id, idx, train=True):
        indices = self.same_actions[task_id][tuple(self.actions[task_id][idx].squeeze())]
        if train:
            indices = [i for i in indices if i in self.train_inds[task_id]]
        else:
            indices = [i for i in indices if i in self.val_inds[task_id]]

        images = np.array([])
        nexts = np.array([])
        rewards = np.array([])

        if indices:
            indices.remove(int(idx))
            if indices:
                indices = np.array(indices)
                images = np.array([self.images[task_id][i] for i in indices]).reshape(self.image_dims)
                nexts = np.array([self.nexts[task_id][i] for i in indices]).reshape(self.image_dims)
                rewards = np.array([self.rewards[task_id][i] for i in indices]).T

        return images, nexts, rewards

    def get_by_action(self, task_id, u):
        if isinstance(u, torch.Tensor):
            u = u.detach().cpu().numpy()
        if u.ndim == 1:
            u = u[:, None]
        u = np.squeeze(u)

        ts = time.time()
        keys = list(self.same_actions[task_id].keys())
        indices = np.where(np.linalg.norm(np.array(keys) - u, axis=-1) < 1e-5)
        indices = [self.same_actions[task_id][keys[i[0]]] for i in indices]
        # print(f'Search same actions time: {time.time() - ts}')
        if not indices:
            return torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

        ts = time.time()
        indices = np.squeeze(indices)

        images = torch.from_numpy(np.array(self.images[task_id]))[indices].reshape(self.image_dims)
        actions = torch.from_numpy(np.array(self.actions[task_id]))[indices]
        nexts = torch.from_numpy(np.array(self.nexts[task_id]))[indices].reshape(self.image_dims)
        # print(f'Create Tensors time: {time.time() - ts}')

        return images, actions, nexts

    def convert(self, task_id: int, encoder: ResNet18EncoderHnet, hnet: HyperNetwork,
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
            states = encoder.forward(data[1].to(gpuid), encoder_weights).detach().cpu().numpy()
            nexts = encoder.forward(data[3].to(gpuid), encoder_weights).detach().cpu().numpy()

            for x_t, u, x_tt in zip(states, data[2], nexts):
                collector.add(x_t, u, x_tt, task_id)

    def sample_action(self, task_id):
        idx = np.random.randint(0, len(self.actions[task_id]))
        action = self.actions[task_id][idx]
        # print(f'Sampled action: {idx} -> {action}')
        return action

    def empty(self, task_id):
        return len(self.images) == 0 or len(self.images[task_id]) == 0

    def save(self):
        if self.save_every < 0:
            return

        for task_id in self.images.keys():
            if len(self.images[task_id]) < self.save_every:
                continue

            save_path = pathlib.Path(self.save_path).joinpath(str(int(time.time())))
            save_path = pathlib.Path(save_path).joinpath(str(task_id))
            image_path = save_path.joinpath("images")
            nexts_path = save_path.joinpath("nexts")

            for p in [save_path, image_path, nexts_path]:
                p.mkdir(parents=True, exist_ok=True)

            actions_path = save_path.joinpath("actions.csv")
            rewards_path = save_path.joinpath("rewards.csv")
            indices_path = save_path.joinpath("indices.csv")

            actions = pd.DataFrame(np.array(self.actions[task_id]).squeeze())
            rewards = pd.DataFrame(np.array(self.rewards[task_id]).squeeze())
            indices = pd.DataFrame(
                [1 if i in self.train_inds[task_id] else 0 for i in range(len(self.images[task_id]))])
            last_saved_index = 0
            if actions_path.exists():
                old_actions = pd.read_csv(actions_path, index_col=False)
                old_rewards = pd.read_csv(rewards_path, index_col=False)
                old_indices = pd.read_csv(indices_path, index_col=False)
                old_actions.columns = actions.columns
                old_rewards.columns = rewards.columns
                old_indices.columns = indices.columns

                last_saved_index = len(old_actions)
                actions = old_actions.append(actions)
                rewards = old_rewards.append(rewards)
                indices = old_indices.append(indices)

            actions.to_csv(actions_path, index=False)
            rewards.to_csv(rewards_path, index=False)
            indices.to_csv(indices_path, index=False)

            for i in range(len(self.images[task_id])):
                cv2.imwrite(str(image_path.joinpath(f"img_{i + last_saved_index}.png")),
                            self.images[task_id][i].reshape(
                                (self.image_dims[2], self.image_dims[3], self.image_dims[1])))
                cv2.imwrite(str(nexts_path.joinpath(f"next_{i + last_saved_index}.png")),
                            self.nexts[task_id][i].reshape(
                                (self.image_dims[2], self.image_dims[3], self.image_dims[1])))

            self.images[task_id] = []
            self.nexts[task_id] = []
            self.actions[task_id] = []
            self.rewards[task_id] = []

    def load(self):
        load_path_base = pathlib.Path(self.save_path).joinpath(self.load_suffix)
        for task_id in [x.name for x in load_path_base.iterdir() if x.is_dir()]:
            task_id = int(task_id)
            self.images[task_id] = []
            self.nexts[task_id] = []
            self.actions[task_id] = []
            self.rewards[task_id] = []
            self.train_inds[task_id] = []
            self.val_inds[task_id] = []

            load_path = load_path_base.joinpath(str(task_id))
            image_path = load_path.joinpath("images")
            nexts_path = load_path.joinpath("nexts")

            actions = pd.read_csv(str(load_path.joinpath("actions.csv"))).to_numpy()
            actions = list(np.expand_dims(actions, axis=2))

            rewards = list(pd.read_csv(str(load_path.joinpath("rewards.csv"))).to_numpy().squeeze())
            indices = [True if np.random.random() < 0.75 else False for _ in
                       range(len(rewards))]  # pd.read_csv(str(load_path.joinpath("indices.csv"))).to_numpy().squeeze()
            train_inds = list(np.where(indices)[0])

            p = rewards
            p = p - min(p)
            p = p / sum(p)
            p = None
            indices_to_keep = [i for i in np.random.choice(range(len(rewards)), size=self.load_max, replace=False, p=p)]

            tmp_train_inds = []
            tmp_val_inds = []

            for ind in indices_to_keep:
                running_index = len(self.images[task_id])
                if running_index % 100 == 0:
                    print(f'Loaded {running_index} / {len(indices_to_keep)}')
                img_path = image_path.joinpath(f'img_{ind}.png')
                x_t = cv2.imread(str(img_path)).reshape(self.image_dims[1:]).flatten()
                next_path = nexts_path.joinpath(f'next_{ind}.png')
                x_tt = cv2.imread(str(next_path)).reshape(self.image_dims[1:]).flatten()
                u = actions[ind]
                r = rewards[ind]

                self.add(x_t, u, x_tt, r, task_id)

                if ind in train_inds:
                    tmp_train_inds.append(running_index)
                else:
                    tmp_val_inds.append(running_index)

            self.train_inds[task_id] = tmp_train_inds
            self.val_inds[task_id] = tmp_val_inds
