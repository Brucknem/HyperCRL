import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

from hypercrl.srl import layer_names
from hypercrl.srl.tools import calculate_priors, calculate_forward_model, calculate_inverse_model, calculate_gt_model, \
    RMSELoss
from hypercrl.srl.utils import calc_mean
from hypercrl.tools import MonitorBase


class MonitorSRL(MonitorBase):
    def __init__(self, hparams, networks, collector, btest):
        super(MonitorSRL, self).__init__(hparams, networks["encoder"]["model"], collector, btest)
        self.eval_every = hparams.vision_params.eval_every
        self.print_train_every = hparams.vision_params.print_train_every
        self.print_model = "hnet_srl"
        self.model_to_save = {'mnet_srl': networks["encoder"]["model"]}
        self.log_hist_every = 500

        self.forward_model = networks["forward"]["model"] if "forward" in networks else None
        self.inverse_model = networks["inverse"]["model"] if "inverse" in networks else None
        self.gt_model = networks["gt"]["model"] if "gt" in networks else None

        self.loss_task = 0
        self.loss_reg = 0

        self.loss_robotic_priors = 0
        self.loss_slowness = 0
        self.loss_variability = 0
        self.loss_proportionality = 0
        self.loss_repeatability = 0
        self.loss_causality = 0

        self.loss_gt_model = 0
        self.loss_forward_model = 0
        self.loss_inverse_model = 0
        self.srl_print_train_every = hparams.vision_params.print_train_every
        # for model_misc in networks.values():
        #     self.writer.add_graph(model_misc["model"])
        self.save_model(self.tflog_dir, hparams, networks, collector)

    def save_model(self, save_folder, hparams, networks, collector):
        train_set, _ = collector.get_dataset(0)

        batch = train_set[:10]

        i, x_t, a_t, x_tt, r, gt_x_t, gt_x_tt = [x.to(hparams.gpuid) for x in batch]
        x_t = networks["encoder"]["model"].forward(x_t, None)
        x_tt = networks["encoder"]["model"].forward(x_tt, None)

        save_path = Path(save_folder)
        make_dot(x_t, params=(dict(list(networks["encoder"]["model"].named_parameters())))).render(
            save_path / "encoder", format="png")

        x_t = torch.zeros_like(x_t)
        val = networks["gt"]["model"].forward(x_t, None)
        make_dot(val, params=(dict(list(networks["gt"]["model"].named_parameters())))).render(
            save_path / "gt", format="png")

        if hparams.vision_params.use_forward_model:
            x_a_t = torch.hstack((x_t, a_t))
            x_a_t = torch.zeros_like(x_a_t)
            val = networks["forward"]["model"].forward(x_a_t, None)
            make_dot(val, params=(dict(list(networks["forward"]["model"].named_parameters())))).render(
                save_path / "forward", format="png")

        if hparams.vision_params.use_inverse_model:
            x_ttt = torch.hstack((x_t, x_tt))
            x_ttt = torch.zeros_like(x_ttt)
            val = networks["inverse"]["model"].forward(x_ttt, None)
            make_dot(val, params=(dict(list(networks["inverse"]["model"].named_parameters())))).render(
                save_path / "inverse", format="png")

    def log_losses(self, i, loss_tot, loss_task, loss_reg_srl, loss_robotic_priors, loss_slowness,
                   loss_variability, loss_proportionality, loss_repeatability, loss_causality,
                   loss_gt_model, loss_forward_model, loss_inverse_model, train=True):
        prefix = "train" if train else "eval"

        print(f"[Vision - {prefix}] Batch: {i}, Loss: {loss_tot:.5f}, " +
              f"Task L: {loss_task:.5f}, Reg L: {loss_reg_srl:.5f}, " +
              f"Prior L: {loss_robotic_priors:.5f}, " +
              f"Slow L: {loss_slowness:.5f}, Vari L: {loss_variability:.5f}, " +
              f"Prop L: {loss_proportionality:.5f}, Rept L: {loss_repeatability:.5f}, " +
              f"Caus L: {loss_causality:.5f}, " +
              f"GrTr L: {loss_gt_model:.5f}, Forward L: {loss_forward_model:.5f}, Inverse L: {loss_inverse_model:.5f}")

        self.writer.add_scalar(f'{prefix}_srl/full_loss', loss_tot, i)
        self.writer.add_scalar(f'{prefix}_srl/task_loss', loss_task, i)
        self.writer.add_scalar(f'{prefix}_srl/regularizer', loss_reg_srl, i)

        self.writer.add_scalar(f'{prefix}_srl/loss_gt_model', loss_gt_model, i)

        if self.hparams.vision_params.use_priors:
            self.writer.add_scalar(f'{prefix}_srl/robotic_priors', loss_robotic_priors, i)
            self.writer.add_scalar(f'{prefix}_priors/_total', loss_robotic_priors, i)
            self.writer.add_scalar(f'{prefix}_priors/slowness_prior', loss_slowness, i)
            self.writer.add_scalar(f'{prefix}_priors/variability_prior', loss_variability, i)
            self.writer.add_scalar(f'{prefix}_priors/proportionality_prior', loss_proportionality, i)
            self.writer.add_scalar(f'{prefix}_priors/repeatability_prior', loss_repeatability, i)
            self.writer.add_scalar(f'{prefix}_priors/causality_prior', loss_causality, i)

        if self.hparams.vision_params.use_forward_model:
            self.writer.add_scalar(f'{prefix}_srl/loss_forward_model', loss_forward_model, i)
        if self.hparams.vision_params.use_inverse_model:
            self.writer.add_scalar(f'{prefix}_srl/loss_inverse_model', loss_inverse_model, i)

    def train_vision_step(self, loss_task, loss_robotic_priors, loss_slowness, loss_variability,
                          loss_proportionality, loss_repeatability, loss_causality, loss_gt_model,
                          loss_forward_model, loss_inverse_model, networks):
        self.loss_task += loss_task.item()
        # self.loss_reg_srl += loss_reg.item()

        self.loss_robotic_priors += float(loss_robotic_priors)
        self.loss_slowness += float(loss_slowness)
        self.loss_variability += float(loss_variability)
        self.loss_proportionality += float(loss_proportionality)
        self.loss_repeatability += float(loss_repeatability)
        self.loss_causality += float(loss_causality)

        self.loss_forward_model += float(loss_forward_model)
        self.loss_inverse_model += float(loss_inverse_model)
        self.loss_gt_model += float(loss_gt_model)

        every_iter = self.srl_print_train_every

        if self.train_iter > 0 and self.train_iter % every_iter == 0:
            self.loss_task /= every_iter
            self.loss_reg /= every_iter
            loss_tot = self.loss_reg + self.loss_task

            self.loss_robotic_priors /= every_iter
            self.loss_slowness /= every_iter
            self.loss_variability /= every_iter
            self.loss_proportionality /= every_iter
            self.loss_repeatability /= every_iter
            self.loss_causality /= every_iter

            self.loss_forward_model /= every_iter
            self.loss_inverse_model /= every_iter
            self.loss_gt_model /= every_iter

            self.log_losses(self.train_iter, loss_tot, self.loss_task, self.loss_reg, self.loss_robotic_priors,
                            self.loss_slowness, self.loss_variability, self.loss_proportionality,
                            self.loss_repeatability, self.loss_causality,
                            self.loss_gt_model, self.loss_forward_model, self.loss_inverse_model,
                            train=True)

            self.loss_task = 0
            self.loss_reg = 0

            self.loss_robotic_priors = 0
            self.loss_slowness = 0
            self.loss_variability = 0
            self.loss_proportionality = 0
            self.loss_repeatability = 0

            self.loss_forward_model = 0
            self.loss_inverse_model = 0
            self.loss_gt_model = 0

        if self.train_iter % self.log_hist_every == 0:
            for model_name, model_misc in networks.items():
                self.log_weights(model_misc, model_name, True)
                self.log_weights(model_misc, model_name, False)
        self.train_iter += 1

    def log_weights(self, model_misc, model_name, linear_layers):
        skipped = 0
        for i, (name, weight) in enumerate(
                zip(model_misc["model"].weight_names, list(model_misc["model"].parameters()))):
            # print(i, name)
            if linear_layers is str(name).startswith("batch_norm"):
                skipped += 1
                continue
            self.writer.add_histogram(f'{model_name}_weight/{i - skipped}_{name}', weight.flatten(), self.train_iter)
            self.writer.add_histogram(f'{model_name}_grad/{i - skipped}_{name}', weight.grad.flatten(), self.train_iter)
            self.writer.add_scalar(f'{model_name}/grad/{i - skipped}_{name}', float(weight.grad.norm()),
                                   self.train_iter)

    def validate(self, networks, srl_collector):
        if (self.train_iter % self.eval_every) == 0:
            num_tasks = self.collector.num_tasks()

            for i in range(num_tasks):
                is_training = (i == (num_tasks - 1))
                # Only evaluate current task in single task model
                if self.hparams.model == "single" and (not is_training):
                    continue
                if len(self.val_stats) <= i:
                    self.val_stats.append({"time": [],
                                           "loss": [],
                                           "priors": [],
                                           "forward": [],
                                           "inverse": [],
                                           "gt": []})

                with torch.no_grad():
                    # Determine if we are validating the currently training task
                    loss, priors, forward, inverse, gt = self.validate_task(i, networks, srl_collector, is_training)

                self.val_stats[i]['time'].append(self.train_iter)
                self.val_stats[i]['loss'].append(float(loss))
                self.val_stats[i]['priors'].append(float(priors))
                self.val_stats[i]['forward'].append(float(forward))
                self.val_stats[i]['inverse'].append(float(inverse))
                self.val_stats[i]['gt'].append(float(gt))

            # Other Sfuff
            # self.btest.plot()

    def validate_task(self, task_id, networks, srl_collector, is_training=False):
        gpuid = self.hparams.gpuid

        losses_gt_model = 0
        losses_forward_model = 0
        losses_inverse_model = 0
        losses_priors, losses_slowness, losses_variability, losses_proportionality, losses_repeatability, losses_causality = 0, 0, 0, 0, 0, 0
        losses = 0
        states = 0
        states_diff = 0
        states_error = 0
        latent_states = 0
        actions_diff = []
        actions_normalized_diff = []

        bs = self.hparams.bs

        _, val_sets = srl_collector.get_dataset(task_id)
        loader = DataLoader(val_sets, batch_size=bs, num_workers=self.hparams.num_ds_worker, drop_last=True)

        num_batches = len(loader)
        if num_batches <= 0:
            return [-1] * 5

        with torch.no_grad():
            for _, data in enumerate(loader):
                idx, x_t, a_t, x_tt, rewards, gt_x_t, gt_x_tt = [x.to(gpuid) for x in data]

                x_t = networks["encoder"]["model"].forward(x_t, None)
                x_tt = networks["encoder"]["model"].forward(x_tt, None)

                loss_gt_model, predicted_gt_x_t = calculate_gt_model(networks, x_t, gt_x_t)
                loss_gtt_model, _ = calculate_gt_model(networks, x_tt, gt_x_tt)
                loss_gt_model = loss_gt_model + loss_gtt_model
                loss_gt_model = loss_gt_model / 2.
                losses_gt_model += loss_gt_model

                loss_forward_model, _ = calculate_forward_model(networks, x_t, a_t, x_tt)
                losses_forward_model += loss_forward_model

                loss_inverse_model, predicted_actions, predicted_actions_normalized = \
                    calculate_inverse_model(networks, x_t, a_t, x_tt)
                losses_inverse_model += loss_inverse_model

                loss_priors, loss_slowness, loss_variability, loss_proportionality, loss_repeatability, loss_causality \
                    = calculate_priors(networks, gpuid, self.hparams, idx, srl_collector, task_id, x_t, x_tt, rewards)

                losses_priors += loss_priors
                losses_slowness += loss_slowness
                losses_variability += loss_variability
                losses_proportionality += loss_proportionality
                losses_repeatability += loss_repeatability
                losses_causality += loss_causality

                loss = loss_priors + loss_forward_model + loss_inverse_model
                if self.hparams.vision_params.train_on_gt_model:
                    loss = loss + loss_gt_model
                losses += loss

                latent_states += (x_t.mean(0))
                states += (predicted_gt_x_t.mean(0))
                states_diff += (gt_x_t - predicted_gt_x_t).mean(0)
                states_error += torch.abs(gt_x_t - predicted_gt_x_t).mean(0)

        states = states / num_batches
        states_diff = states_diff / num_batches
        states_error = states_error / num_batches
        latent_states = latent_states / num_batches
        with np.errstate(divide="ignore"):
            states_relative_error = states_error.detach().cpu().numpy() / srl_collector.ranges[task_id]['real_state']
        states_relative_error[states_relative_error == np.inf] = 0
        self.writer.add_histogram(f'eval_states/{task_id}', states, self.train_iter)
        self.writer.add_histogram(f'eval_states_diff/{task_id}', states_diff, self.train_iter)
        self.writer.add_histogram(f'eval_states_error_absolute/{task_id}', states_error, self.train_iter)
        self.writer.add_histogram(f'eval_states_error_relative/{task_id}', states_relative_error, self.train_iter)
        self.writer.add_histogram(f'eval_latent_states/{task_id}', latent_states, self.train_iter)

        # actions = torch.cat(actions)
        # actions_normalized = torch.cat(actions_normalized)
        # actions_mean = torch.mean(actions, 0)
        # actions_std = torch.std(actions, 0)
        # actions_normalized_mean = torch.mean(actions_normalized, 0)
        # actions_normalized_std = torch.std(actions_normalized, 0)
        #
        # for i in range(len(actions_mean)):
        #     self.writer.add_scalar(f'actions/{task_id}/{i}/mean', actions_mean[i], self.train_iter)
        #     self.writer.add_scalar(f'actions/{task_id}/{i}/std', actions_std[i], self.train_iter)
        #     self.writer.add_scalar(f'actions_normalized/{task_id}/{i}/mean', actions_normalized_mean[i],
        #                            self.train_iter)
        #     self.writer.add_scalar(f'actions_normalized/{task_id}/{i}/std', actions_normalized_std[i], self.train_iter)

        losses /= num_batches
        losses_priors /= num_batches
        losses_forward_model /= num_batches
        losses_inverse_model /= num_batches
        losses_gt_model /= num_batches

        self.log_losses(self.train_iter,
                        losses,
                        losses,
                        0,
                        losses_priors,
                        losses_slowness / num_batches,
                        losses_variability / num_batches,
                        losses_proportionality / num_batches,
                        losses_repeatability / num_batches,
                        losses_causality / num_batches,
                        loss_gt_model,
                        losses_forward_model,
                        losses_inverse_model,
                        train=False)

        return losses, losses_priors, losses_forward_model, losses_inverse_model, losses_gt_model


class MonitorSRLHnet(MonitorSRL):
    def __init__(self, hparams, mnet, hnet, collector, btest):
        super(MonitorSRL, self).__init__(hparams, mnet, collector, btest)
        self.model_to_save = {'mnet_srl': mnet, 'hnet_srl': hnet}
