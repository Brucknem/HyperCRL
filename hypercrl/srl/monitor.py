import os

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hypercrl.srl.tools import calculate_priors
from hypercrl.tools import MonitorBase


class MonitorSRL(MonitorBase):
    def __init__(self, hparams, mnet, collector, btest):
        super(MonitorSRL, self).__init__(hparams, mnet, collector, btest)
        self.eval_every = hparams.vision_params.eval_every
        self.print_train_every = hparams.vision_params.print_train_every
        self.print_model = "hnet_srl"
        self.model_to_save = {'mnet_srl': mnet}
        self.log_hist_every = 500

        self.loss_task = 0
        self.loss_reg = 0

        self.loss_task_srl = 0
        self.loss_reg_srl = 0

        self.loss_robotic_priors = 0
        self.loss_slowness = 0
        self.loss_variability = 0
        self.loss_proportionality = 0
        self.loss_repeatability = 0
        self.loss_causality = 0

        self.loss_forward_model = 0
        self.loss_inverse_model = 0

        self.srl_print_train_every = hparams.vision_params.print_train_every

    def train_vision_step(self, loss_task, loss_robotic_priors, loss_slowness, loss_variability,
                          loss_proportionality, loss_repeatability, loss_causality, loss_forward_model,
                          loss_inverse_model, loss_reg, dTheta, grad_tloss,
                          weights, x_t):
        self.loss_task_srl += loss_task.item()
        self.loss_reg_srl += loss_reg.item()

        self.loss_robotic_priors += float(loss_robotic_priors)
        self.loss_slowness += float(loss_slowness)
        self.loss_variability += float(loss_variability)
        self.loss_proportionality += float(loss_proportionality)
        self.loss_repeatability += float(loss_repeatability)
        self.loss_causality += float(loss_causality)

        self.loss_forward_model += loss_forward_model.item()
        self.loss_inverse_model += loss_inverse_model.item()

        every_iter = self.srl_print_train_every

        if self.train_iter > 0 and self.train_iter % every_iter == 0:
            self.loss_task_srl /= every_iter
            self.loss_reg_srl /= every_iter
            loss_tot = self.loss_reg_srl + self.loss_task_srl

            self.loss_robotic_priors /= every_iter
            self.loss_slowness /= every_iter
            self.loss_variability /= every_iter
            self.loss_proportionality /= every_iter
            self.loss_repeatability /= every_iter
            self.loss_causality /= every_iter

            self.loss_forward_model /= every_iter
            self.loss_inverse_model /= every_iter

            print(f"[Vision] Batch: {self.train_iter}, Loss: {loss_tot:.5f}, " +
                  f"Task L: {self.loss_task_srl:.5f}, Reg L: {self.loss_reg_srl:.5f}, " +
                  f"Prior L: {self.loss_robotic_priors:.5f}, " +
                  f"Slow L: {self.loss_slowness:.5f}, Vari L: {self.loss_variability:.5f}, " +
                  f"Prop L: {self.loss_proportionality:.5f}, Rept L: {self.loss_repeatability:.5f}, " +
                  f"Caus L: {self.loss_causality:.5f}, " +
                  f"Forward L: {self.loss_forward_model:.5f}, Inverse L: {self.loss_inverse_model:.5f}")

            i = self.train_iter

            self.writer.add_scalar('srl/full_loss', loss_tot, i)
            self.writer.add_scalar('srl/regularizer', self.loss_reg_srl, i)

            self.writer.add_scalar('srl/robotic_priors', self.loss_robotic_priors, i)
            self.writer.add_scalar('priors/_total', self.loss_robotic_priors, i)
            self.writer.add_scalar('priors/slowness_prior', self.loss_slowness, i)
            self.writer.add_scalar('priors/variability_prior', self.loss_variability, i)
            self.writer.add_scalar('priors/proportionality_prior', self.loss_proportionality, i)
            self.writer.add_scalar('priors/repeatability_prior', self.loss_repeatability, i)
            self.writer.add_scalar('priors/causality_prior', self.loss_causality, i)

            self.writer.add_scalar('srl/loss_forward_model', self.loss_forward_model, i)
            self.writer.add_scalar('srl/loss_inverse_model', self.loss_inverse_model, i)

            if dTheta is not None:
                dT_norm = torch.norm(torch.cat([d.view(-1) for d in dTheta]), 2)
                self.writer.add_scalar('srl/dTheta_norm', dT_norm, i)
            if grad_tloss is not None:
                (grad_tloss, grad_full, grad_diff_norm, grad_cos) = grad_tloss
                self.writer.add_scalar('srl/full_grad_norm',
                                       torch.norm(grad_full, 2), i)
                self.writer.add_scalar('srl/reg_grad_norm',
                                       grad_diff_norm, i)
                self.writer.add_scalar('srl/cosine_task_reg',
                                       grad_cos, i)

            self.loss_task_srl = 0
            self.loss_reg_srl = 0

            self.loss_robotic_priors = 0
            self.loss_slowness = 0
            self.loss_variability = 0
            self.loss_proportionality = 0
            self.loss_repeatability = 0

            self.loss_forward_model = 0
            self.loss_inverse_model = 0

        if self.train_iter % self.log_hist_every == 0:
            for i, weight in enumerate(weights):
                self.writer.add_histogram(f'srl/weight/{i}', weight.flatten(), self.train_iter)
        self.train_iter += 1

    def validate(self, priors, forward_misc, inverse_misc):
        if (self.train_iter % self.eval_every) == 0:
            self.model.eval()

            bs = self.hparams.bs
            num_tasks = self.collector.num_tasks()

            for i in range(num_tasks):
                is_training = (i == (num_tasks - 1))
                # Only evaluate current task in single task model
                if self.hparams.model == "single" and (not is_training):
                    continue
                if len(self.val_stats) <= i:
                    self.val_stats.append({"time": [],
                                           "nll": [], "diff": []})
                _, val_sets = self.collector.get_dataset(i)
                loader = DataLoader(val_sets, batch_size=bs,
                                    num_workers=self.hparams.num_ds_worker)

                # Determine if we are validating the currently training task
                val_nll, val_diff = self.validate_task(i, loader, priors, forward_misc, inverse_misc, is_training)

                self.val_stats[i]['time'].append(self.train_iter)
                self.val_stats[i]['nll'].append(val_nll.item())
                self.val_stats[i]['diff'].append(val_diff.mean().item())

            self.model.train()
            # Other Sfuff
            # self.btest.plot()

    def validate_task(self, task_id, loader, priors, forward_misc, inverse_misc, is_training=False):
        gpuid = self.hparams.gpuid

        # Initialize Stats
        val_loss = 0
        val_diff = 0
        states = []

        N = len(loader)

        forward_model, forward_model_optimizer, forward_model_loss_fn = forward_misc
        inverse_model, inverse_model_optimizer, inverse_model_loss_fn = inverse_misc

        with torch.no_grad():
            for _, data in enumerate(loader):
                idx, x_t, a_t, x_tt, r = [x.to(gpuid) for x in data]

                if is_training:
                    # Inference in weight space
                    x_t = self.model.forward(x_t, None)
                    x_tt = self.model.forward(x_tt, None)
                else:
                    raise NotImplementedError("Idk")
                    # Inference in function space
                    # output = self.model(x_t, a_t, task_id=task_id)

                x_a_t = torch.hstack((x_t, a_t))
                predicted_x_tt = forward_model(x_a_t)
                loss_forward_model = forward_model_loss_fn(x_tt, predicted_x_tt)

                predicted_actions = inverse_model(x_t)
                loss_inverse_model = inverse_model_loss_fn(a_t, predicted_actions)

                loss_priors, loss_slowness, loss_variability, loss_proportionality, loss_repeatability, loss_causality = calculate_priors(
                    priors, gpuid, self.hparams, idx, self.model, None, self.collector, task_id, x_t, x_tt, r)

                diff = torch.abs(x_tt - x_t).mean(0)

                loss = loss_priors + loss_forward_model + loss_inverse_model

                val_loss += loss
                val_diff += diff
                states += [x_t.mean(0)]

            val_loss = val_loss / N
            val_diff = val_diff / N
            states = torch.cat(states).mean(0)

        self.writer.add_histogram(f'srl_states/{task_id}', states, self.train_iter)

        print(f"Iter {self.train_iter}, Task: {task_id}, " + \
              f"Val Loss: {val_loss.item():.5f}, Val Diff: {val_diff.mean().item()}")

        return val_loss, val_diff


class MonitorSRLHnet(MonitorSRL):
    def __init__(self, hparams, mnet, hnet, collector, btest):
        super(MonitorSRL, self).__init__(hparams, mnet, collector, btest)
        self.model_to_save = {'mnet_srl': mnet, 'hnet_srl': hnet}
