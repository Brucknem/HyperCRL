import time

import cv2
import torch
from torch.utils.data import TensorDataset
import numpy as np

import hypercrl.srl.models
from hypercrl.hypercl import MLP
from hypercrl.model.tools import initialize_hnet, calculate_compression_ratio, generate_hnet
from hypercrl.srl import ResNet18EncoderHnet
from hypercrl.srl.datautil import DataCollector
from hypercrl.hypercl.utils import hnet_regularizer as hreg
from hypercrl.srl.robotic_priors import RoboticPriors, SlownessPrior, CausalityPrior, RepeatabilityPrior, \
    ReferencePointPrior, VariabilityPrior, ProportionalityPrior
from torch.utils.data import DataLoader

from hypercrl.srl.utils import scale_to_action_space
from hypercrl.tools import reset_seed, str_to_act
from hypercrl.tools import MonitorHnet, HP, Hparams
from hypercrl.control import RandomAgent, MPC
from hypercrl.envs.cl_env import CLEnvHandler
from hypercrl.dataset.datautil import DataCollector
from hypercrl.srl.datautil import DataCollector as SRLDataCollector

from hypercrl.model import build_model_hnet as build_model

from hypercrl.model import reload_model_hnet as reload_model
from hypercrl.hypercl.utils import hnet_regularizer as hreg
from hypercrl.hypercl.utils import ewc_regularizer as ewc
from hypercrl.hypercl.utils import si_regularizer as si
from hypercrl.hypercl.utils import optim_step as opstep
from hypercrl.srl.models import ResNet18Encoder

from contextlib import contextmanager
from contextlib import contextmanager


@contextmanager
def timeit_context(arr, name=""):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    if name:
        print(f'{name}: {elapsed_time}')
    arr.append(elapsed_time)


def build_vision_model_hnet(hparams):
    vparams = hparams.vision_params
    encoder_mnet = MLP(n_in=vparams.in_dim,
                       n_out=hparams.state_dim, hidden_layers=vparams.h_dims,
                       no_weights=True, out_var=vparams.out_var,
                       use_batch_norm=vparams.use_batch_norm,
                       dropout_rate=vparams.dropout_rate)
    print('Constructed Vision MLP with shapes: ', encoder_mnet.param_shapes)
    encoder_mnet = ResNet18EncoderHnet(encoder_mnet, vparams)
    hnet = generate_hnet(vparams.model, encoder_mnet.param_shapes, vparams.hnet_arch, vparams.emb_size,
                         vparams.hnet_act)
    calculate_compression_ratio(hnet, vparams, encoder_mnet.param_shapes)
    initialize_hnet(hnet, vparams)

    return encoder_mnet, hnet


def reload_vision_model_hnet(hparams):
    encoder_mnet, encoder_hnet = build_vision_model_hnet(hparams)

    checkpoint = None
    collector = DataCollector(hparams)
    # MASTER_THESIS really load models

    return encoder_mnet, encoder_hnet, checkpoint, collector


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y) + 1e-8)


def generate_srl_networks(hparams, action_space):
    gpuid = hparams.gpuid
    networks = {}

    mlp = MLP(
        hparams.vision_params.in_dim,
        hparams.state_dim,
        hparams.vision_params.encoder_model.h_dims,
        use_batch_norm=hparams.vision_params.encoder_model.bn,
        dropout_rate=hparams.vision_params.encoder_model.dropout,
    )
    encoder = mlp  # ResNet18Encoder(mlp, hparams.vision_params)
    encoder.to(gpuid)
    encoder.train()
    networks["encoder"] = {'model': encoder, 'params': encoder.parameters(),
                           'lr': hparams.vision_params.encoder_model.lr,
                           'loss': (SlownessPrior(), VariabilityPrior(), ProportionalityPrior(), RepeatabilityPrior(),
                                    CausalityPrior()),
                           'reg_str': hparams.vision_params.encoder_model.reg_str
                           }

    latent_dim = hparams.state_dim
    if hparams.vision_params.add_sin_cos_to_state:
        latent_dim *= 3

    gt_model = MLP(latent_dim,
                   hparams.numerical_state_dim,
                   hparams.vision_params.gt_model.h_dims,
                   use_batch_norm=False,
                   )
    gt_model.to(gpuid)
    gt_model.train()
    networks["gt"] = {'model': gt_model, 'params': gt_model.parameters(),
                      'lr': hparams.vision_params.gt_model.lr,
                      'loss': RMSELoss(),
                      'reg_str': hparams.vision_params.gt_model.reg_str
                      }

    if hparams.vision_params.use_forward_model:
        forward_model = MLP(latent_dim + hparams.control_dim,
                            hparams.state_dim,
                            hparams.vision_params.forward_model_dims,
                            use_batch_norm=True,
                            )
        forward_model.to(gpuid)
        forward_model.train()
        networks["forward"] = {'model': forward_model, 'params': forward_model.parameters(),
                               'lr': hparams.vision_params.forward_model_lr,
                               'loss': RMSELoss(),
                               'reg_str': hparams.vision_params.forward_model.reg_str
                               }

    if hparams.vision_params.use_inverse_model:
        inverse_model = MLP(2 * latent_dim,
                            hparams.control_dim,
                            hparams.vision_params.inverse_model_dims,
                            use_batch_norm=True,
                            out_fn=scale_to_action_space(action_space, gpuid, activation='tanh')
                            )
        inverse_model.to(gpuid)
        inverse_model.train()

        networks["inverse"] = {'model': inverse_model, 'params': inverse_model.parameters(),
                               'lr': hparams.vision_params.inverse_model_lr,
                               'loss': RMSELoss(),
                               'reg_str': hparams.vision_params.inverse_model.reg_str
                               }

    return networks


def generate_optimizer(networks):
    return torch.optim.Adam([net for net in networks.values()])


def calculate_gt_model(networks, x_t, gt_x_tt):
    if "gt" not in networks:
        return 0, None

    if "times" not in networks["gt"]:
        networks["gt"]["times"] = []
    gt_misc = networks["gt"]

    with timeit_context(gt_misc["times"]):
        predicted_gt_x_t = gt_misc["model"](x_t)
        return gt_misc["loss"](gt_x_tt, predicted_gt_x_t), predicted_gt_x_t


def calculate_forward_model(networks, x_t, a_t, x_tt):
    if "forward" not in networks:
        return 0, None

    if "times" not in networks["forward"]:
        networks["forward"]["times"] = []
    forward_misc = networks["forward"]

    with timeit_context(forward_misc["times"]):
        x_a_t = torch.hstack((x_t, a_t))
        predicted_x_tt = forward_misc["model"](x_a_t)
        return forward_misc["loss"](x_tt, predicted_x_tt), predicted_x_tt


def calculate_inverse_model(networks, x_t, a_t, x_tt):
    if "inverse" not in networks:
        return 0, None, None

    if "times" not in networks["inverse"]:
        networks["inverse"]["times"] = []
    inverse_misc = networks["inverse"]

    with timeit_context(inverse_misc["times"]):
        stacked_states = torch.hstack([x_t, x_tt])
        predicted_actions_normalized, predicted_actions = inverse_misc["model"](stacked_states)
        # print("Predicted actions range: ", torch.min(predicted_actions), torch.max(predicted_actions))
        return inverse_misc["loss"](a_t, predicted_actions_normalized), predicted_actions, predicted_actions_normalized


def calculate_regularization(networks, model):
    if model not in networks:
        return 0

    if "reg_str" not in networks[model]:
        return 0

    if networks[model]["reg_str"] == 0:
        return 0

    return networks[model]["reg_str"] * sum([torch.sum(torch.sum(torch.abs(s))) for s in (networks[model]["params"])])


def train(task_id, networks, optimizer, logger, train_loader, srl_collector, hparams, train_it):
    # MASTER_THESIS Here is where the magic happens. We need to add the SRL to the training loop and add encoding prior to the dynamics update!
    # MASTER_THESIS Implement real training

    print("Training Vision-Based SRL")

    # GPUID
    gpuid = hparams.gpuid

    total_time_iter = []
    total_time_hnet = []
    total_time_hnet_optimize = []
    total_time_forward = []
    total_time_forward_optimize = []
    total_time_inverse = []
    total_time_inverse_optimize = []
    total_time_priors = []

    mnet_weights = None

    it = 0
    while it <= hparams.vision_params.train_vision_iters:
        for net in networks.values():
            net["model"].to(gpuid)
            net["model"].train()

        for i, data in enumerate(train_loader):
            with timeit_context(total_time_iter):
                #  if it % 200 == 0:
                #     print(f'Iteration: {it} / {hparams.vision_params.train_vision_iters}')

                optimizer.zero_grad()
                idx, x_t, a_t, x_tt, rewards, gt_x_t, gt_x_tt = [x.to(gpuid) for x in data]

                x_t = networks["encoder"]["model"].forward(x_t, mnet_weights)
                x_tt = networks["encoder"]["model"].forward(x_tt, mnet_weights)

                loss_gt_model, _ = calculate_gt_model(networks, x_t, gt_x_t)
                loss_gtt_model, _ = calculate_gt_model(networks, x_tt, gt_x_tt)
                loss_gt_model = loss_gt_model + loss_gtt_model
                loss_gt_model = loss_gt_model / 2.

                loss_forward_model, _ = calculate_forward_model(networks, x_t, a_t, x_tt)
                loss_inverse_model, _, _ = calculate_inverse_model(networks, x_t, a_t, x_tt)
                loss_priors, loss_slowness, loss_variability, loss_proportionality, loss_repeatability, loss_causality \
                    = calculate_priors(networks, gpuid, hparams, idx, srl_collector, task_id, x_t, x_tt, rewards)

                with timeit_context(total_time_hnet_optimize):
                    # We already compute the gradients, to then be able to compute delta
                    # theta.
                    # loss_gt_model = loss_gt_model + calculate_regularization(networks, "gt")
                    # loss_inverse_model = loss_inverse_model + calculate_regularization(networks, "inverse")
                    # loss_forward_model = loss_forward_model + calculate_regularization(networks, "forward")
                    # loss_encoder_model = calculate_regularization(networks, "encoder")

                    loss_task = loss_priors + loss_forward_model + loss_inverse_model
                    if hparams.vision_params.train_on_gt_model:
                        loss_task = loss_task + loss_gt_model
                    else:
                        loss_gt_model.backward()
                    loss_task.backward()
                    optimizer.step()

                logger.train_vision_step(loss_task, loss_priors, loss_slowness, loss_variability, loss_proportionality,
                                         loss_repeatability, loss_causality, loss_gt_model, loss_forward_model,
                                         loss_inverse_model, networks)

                # Validate
                logger.validate(networks, srl_collector)

                it += 1
                if it > hparams.vision_params.train_vision_iters:
                    break

    # print(f"Vision-Based SRL training time", sum(total_time_iter))
    # logger.writer.add_scalar('time/train_srl', sum(total_time_iter), (train_it + 1) * it)
    # logger.writer.add_scalar('time/train_srl_iter', np.mean(total_time_iter), (train_it + 1) * it)
    # logger.writer.add_scalar('time/train_srl_hnet', np.mean(total_time_hnet), (train_it + 1) * it)
    # logger.writer.add_scalar('time/train_srl_hnet_optimize', np.mean(total_time_hnet_optimize), (train_it + 1) * it)
    # logger.writer.add_scalar('time/train_srl_forward', np.mean(total_time_forward), (train_it + 1) * it)
    # logger.writer.add_scalar('time/train_srl_forward_optimize', np.mean(total_time_forward_optimize),
    #                          (train_it + 1) * it)
    # logger.writer.add_scalar('time/train_srl_inverse', np.mean(total_time_inverse), (train_it + 1) * it)
    # logger.writer.add_scalar('time/train_srl_inverse_optimize', np.mean(total_time_inverse_optimize),
    #                          (train_it + 1) * it)
    # logger.writer.add_scalar('time/calculate_priors', np.mean(total_time_priors), (train_it + 1) * it)


def calculate_priors(networks, gpuid, hparams, idx, srl_collector, task_id, x_t, x_tt, r):
    if "times" not in networks["encoder"]:
        networks["encoder"]["times"] = []

    with timeit_context(networks["encoder"]["times"]):
        loss_priors = 0
        if not hparams.vision_params.use_priors:
            return 0, 0, 0, 0, 0, 0

        slowness_prior, variability_prior, proportionality_prior, repeatability_prior, causality_prior = \
            networks["encoder"]["loss"]

        loss_slowness = slowness_prior(x_t, x_tt)
        loss_priors += loss_slowness
        loss_variability = variability_prior(x_t, x_tt[torch.randperm(x_tt.size()[0])])
        loss_priors += loss_variability

        if hparams.vision_params.use_fast_priors:
            return loss_priors, loss_slowness, loss_variability, 0, 0, 0

        states = []
        next_states = []
        rewards = []
        other_states = []
        next_other_states = []
        other_rewards = []
        with timeit_context([]):  # , "Get same actions"):
            for index, id in enumerate(idx):
                # MASTER_THESIS Pick only single other same action to have same batch size
                other_x_t, other_x_tt, other_r = srl_collector.get_same_actions(task_id, id, train=True)
                num_other = len(other_x_t)
                if num_other <= 0:
                    continue
                states = states + [x_t[index].unsqueeze(dim=0)] * num_other
                next_states = next_states + [x_tt[index].unsqueeze(dim=0)] * num_other
                rewards = rewards + [r[index].unsqueeze(dim=0)] * num_other
                other_states = other_states + [torch.from_numpy(other_x_t)]
                next_other_states = next_other_states + [torch.from_numpy(other_x_tt)]
                other_rewards = other_rewards + [torch.from_numpy(other_r)]

        if not states:
            return loss_priors, loss_slowness, loss_variability, 0, 0, 0

        with timeit_context([]):  # , "Calculate slow priors"):
            sample_indices = np.random.choice(len(states), size=len(x_t))
            states, next_states, other_states, next_other_states, rewards, other_rewards = \
                [torch.cat(x)[sample_indices].to(gpuid) for x in
                 [states, next_states, other_states, next_other_states, rewards, other_rewards]]

            other_states = networks["encoder"]["model"].forward(other_states, None)
            next_other_states = networks["encoder"]["model"].forward(next_other_states, None)

            loss_proportionality = proportionality_prior(states, next_states, other_states, next_other_states)
            loss_repeatability = repeatability_prior(states, next_states, other_states, next_other_states)
            loss_causality = causality_prior(states, next_states, rewards, other_rewards)
            # loss_causality = 0

            loss_priors += loss_proportionality
            loss_priors += loss_repeatability
            loss_priors += loss_causality
        return loss_priors, loss_slowness, loss_variability, loss_proportionality, loss_repeatability, loss_causality
