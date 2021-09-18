import torch
import torchvision.utils

from .tools import MonitorRL
import numpy as np


class MonitorHnet(MonitorRL):
    def __init__(self, hparams, agent, mnet, hnet, collector, encoder_mnet=None, encoder_hnet=None):
        super(MonitorHnet, self).__init__(hparams, agent, mnet, collector, None)
        self.mnet = mnet
        self.hnet = hnet
        self.model_to_save = {'mnet': mnet, 'hnet': hnet, 'encoder_mnet': encoder_mnet, 'encoder_hnet': encoder_hnet}

        self.loss_task = 0
        self.loss_reg = 0

        self.loss_task_srl = 0
        self.loss_reg_srl = 0

        self.loss_robotic_priors = 0
        self.loss_slowness = 0
        self.loss_variability = 0
        self.loss_proportionality = 0
        self.loss_repeatability = 0

        self.loss_forward_model = 0
        self.loss_inverse_model = 0

        self.srl_print_train_every = hparams.vision_params.print_train_every

    def train_step(self, loss_task, loss_reg, dTheta, grad_tloss, weights):
        self.loss_task += loss_task.item()
        self.loss_reg += loss_reg.item()
        if self.train_iter > 0 and self.train_iter % self.print_train_every == 0:
            self.loss_task /= self.print_train_every
            self.loss_reg /= self.print_train_every
            loss_tot = self.loss_reg + self.loss_task
            print(f"[Dynamics] Batch: {self.train_iter}, Loss: {loss_tot:.5f}, " +
                  f"Task L: {self.loss_task:.5f}, Reg L: {self.loss_reg:.5f}")

            i = self.train_iter

            self.writer.add_scalar('train/mse_loss', self.loss_task, i)
            self.writer.add_scalar('train/regularizer', self.loss_reg, i)
            self.writer.add_scalar('train/full_loss', loss_tot, i)
            if dTheta is not None:
                dT_norm = torch.norm(torch.cat([d.view(-1) for d in dTheta]), 2)
                self.writer.add_scalar('train/dTheta_norm', dT_norm, i)
            if grad_tloss is not None:
                (grad_tloss, grad_full, grad_diff_norm, grad_cos) = grad_tloss
                self.writer.add_scalar('train/full_grad_norm',
                                       torch.norm(grad_full, 2), i)
                self.writer.add_scalar('train/reg_grad_norm',
                                       grad_diff_norm, i)
                self.writer.add_scalar('train/cosine_task_reg',
                                       grad_cos, i)

            self.loss_task = 0
            self.loss_reg = 0

        if (self.train_iter % self.log_hist_every == 0):
            for i, weight in enumerate(weights):
                self.writer.add_histogram(f'train/weight/{i}', weight.flatten(), self.train_iter)
        self.train_iter += 1

    def train_vision_step(self, loss_task, loss_robotic_priors, loss_slowness, loss_variability, loss_proportionality,
                          loss_repeatability, loss_forward_model, loss_inverse_model, loss_reg, dTheta, grad_tloss,
                          weights, x_t):
        self.loss_task_srl += loss_task.item()
        self.loss_reg_srl += loss_reg.item()

        self.loss_robotic_priors += float(loss_robotic_priors)
        self.loss_slowness += float(loss_slowness)
        self.loss_variability += float(loss_variability)
        self.loss_proportionality += float(loss_proportionality)
        self.loss_repeatability += float(loss_repeatability)

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

            self.loss_forward_model /= every_iter
            self.loss_inverse_model /= every_iter

            print(f"[Vision] Batch: {self.train_iter}, Loss: {loss_tot:.5f}, " +
                  f"Task L: {self.loss_task_srl:.5f}, Reg L: {self.loss_reg_srl:.5f}, " +
                  f"Prior L: {self.loss_robotic_priors:.5f}, " +
                  f"Slow L: {self.loss_slowness:.5f}, Vari L: {self.loss_variability:.5f}, " +
                  f"Prop L: {self.loss_proportionality:.5f}, Rept L: {self.loss_repeatability:.5f}, " +
                  f"Forward L: {self.loss_forward_model:.5f}, Inverse L: {self.loss_inverse_model:.5f}")

            i = self.train_iter

            mins = torch.hstack([x_t.min(dim=1)[0].unsqueeze(1)] * len(x_t[0]))
            maxs = torch.hstack([x_t.max(dim=1)[0].unsqueeze(1)] * len(x_t[0]))
            x_t = (x_t - mins) / (maxs - mins)
            x_t = torch.hstack([torch.kron(x_t, torch.ones(1, 1050 // len(x_t[0]), device=x_t.get_device())).unsqueeze(
                1)] * 100).unsqueeze(1)
            self.writer.add_image('states', torchvision.utils.make_grid(x_t, nrow=1), i)

            self.writer.add_scalar('srl/full_loss', loss_tot, i)
            self.writer.add_scalar('srl/regularizer', self.loss_reg_srl, i)

            self.writer.add_scalar('srl/robotic_priors', self.loss_robotic_priors, i)
            self.writer.add_scalar('priors/slowness_prior', self.loss_slowness, i)
            self.writer.add_scalar('priors/variability_prior', self.loss_variability, i)
            self.writer.add_scalar('priors/proportionality_prior', self.loss_proportionality, i)
            self.writer.add_scalar('priors/repeatability_prior', self.loss_repeatability, i)

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

        if (self.train_iter % self.log_hist_every == 0):
            for i, weight in enumerate(weights):
                self.writer.add_histogram(f'srl/weight/{i}', weight.flatten(), self.train_iter)
        self.train_iter += 1

    def data_aggregate_step(self, x_tt, task_id, it):
        if self.hparams.env == "lqr10":
            l2_pos = np.linalg.norm(x_tt[:10])
            l2_vel = np.linalg.norm(x_tt[10:])
            self.writer.add_scalar(f'lqr10/{task_id}/l2_pos', l2_pos, it)
            self.writer.add_scalar(f'lqr10/{task_id}/l2_vel', l2_vel, it)

    def validate_task(self, task_id, loader, mll, is_training=False):
        self.mnet.eval()
        self.hnet.eval()

        gpuid = self.hparams.gpuid

        # Initialize Stats
        val_loss = 0
        val_diff = 0
        N = len(loader)

        with torch.no_grad():
            mnet_weights = self.hnet.forward(task_id)

            for _, data in enumerate(loader):
                x_t, a_t, x_tt = data
                x_t, a_t, x_tt = x_t.to(gpuid), a_t.to(gpuid), x_tt.to(gpuid)
                X = torch.cat((x_t, a_t), dim=-1)

                Y = self.mnet.forward(X, mnet_weights)

                loss = mll(Y, x_tt, mnet_weights)
                if self.hparams.out_var:
                    Y, _ = torch.split(Y, Y.size(-1) // 2, dim=-1)
                diff = torch.abs(Y - x_tt).mean(dim=0)

                val_loss += loss
                val_diff += diff

            val_loss = val_loss / N
            val_diff = val_diff / N

        print(f"Iter {self.train_iter}, Task: {task_id}, " + \
              f"Val Loss: {val_loss.item():.5f}, Val Diff: {val_diff.mean().item()}")

        return val_loss, val_diff
