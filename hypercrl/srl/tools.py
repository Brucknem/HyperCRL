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


def generate_srl_losses(hparams):
    priors = (SlownessPrior(), VariabilityPrior(), ProportionalityPrior(), RepeatabilityPrior(), CausalityPrior())
    gpuid = hparams.gpuid

    if hparams.vision_params.use_forward_model:
        forward_model = MLP(hparams.state_dim + hparams.control_dim,
                            hparams.state_dim,
                            # hparams.vision_params.forward_model_dims,
                            [hparams.state_dim] * 4,
                            use_batch_norm=True,
                            )
        forward_model.to(gpuid)
        forward_model.train()
        forward_misc = (
            forward_model, torch.optim.Adam(forward_model.parameters(), lr=hparams.vision_params.forward_model_lr),
            torch.nn.MSELoss())
    else:
        forward_misc = (None,) * 3
    if hparams.vision_params.use_inverse_model:
        inverse_model = MLP(hparams.state_dim,
                            hparams.control_dim,
                            # hparams.vision_params.inverse_model_dims,
                            [hparams.state_dim] * 4,
                            use_batch_norm=True,
                            )
        inverse_model.to(gpuid)
        inverse_model.train()
        inverse_misc = (
            inverse_model, torch.optim.Adam(inverse_model.parameters(), lr=hparams.vision_params.inverse_model_lr),
            torch.nn.MSELoss())
    else:
        inverse_misc = (None,) * 3

    return priors, forward_misc, inverse_misc


def augment_model(task_id, mnet, hnet, collector, hparams):
    # Regularizer targets.
    targets = hreg.get_current_targets(task_id, hnet)

    # Add new hypernet embeddings and Loss Function
    hnet.add_task(task_id, hparams.std_normal_temb)

    # if hparams.model == "hnet_mt":
    #     # Loss Function
    #     mll = TaskLossMT(hparams, mnet, hnet, collector, task_id)
    # elif hparams.model == "hnet_replay":
    #     mll = TaskLossReplay(hparams, mnet, hnet, collector, task_id)
    # else:
    #     mll = TaskLoss(hparams, mnet)

    # MASTER_THESIS Use separate priors

    # (Re)Put model to GPU
    gpuid = hparams.gpuid
    mnet.to(gpuid)
    hnet.to(gpuid)

    # Optimize over the GP model params and likelihood param
    mnet.train()
    hnet.train()

    # # Collect Fisher estimates for the reg computation.
    # fisher_ests = None
    # if hparams.ewc_weight_importance and task_id > 0:
    #     fisher_ests = []
    #     n_W = len(hnet.target_shapes)
    #     for t in range(task_id):
    #         ff = []
    #         for i in range(n_W):
    #             _, buff_f_name = ewc._ewc_buffer_names(t, i, False)
    #             ff.append(getattr(mnet, buff_f_name))
    #         fisher_ests.append(ff)
    #
    # # Register SI buffers for new task
    # si_omega = None
    # if hparams.model == "hnet_si":
    #     si.si_register_buffer(mnet, hnet, task_id)
    #     if task_id > 0:
    #         si_omega = si.get_si_omega(mnet, task_id)

    regularized_params = list(hnet.theta)
    if task_id > 0 and hparams.plastic_prev_tembs:
        for i in range(task_id):  # for all previous task embeddings
            regularized_params.append(hnet.get_task_emb(i))
    theta_optimizer = torch.optim.Adam(regularized_params, lr=hparams.vision_params.lr_hyper)
    # We only optimize the task embedding corresponding to the current task,
    # the remaining ones stay constant.
    emb_optimizer = torch.optim.Adam([hnet.get_task_emb(task_id)], lr=hparams.vision_params.lr_hyper)

    trainer_misc = (targets, *generate_srl_losses(hparams), theta_optimizer, emb_optimizer,
                    regularized_params)  # , fisher_ests, si_omega)

    return trainer_misc


def train(task_id, mnet, hnet, trainer_misc, logger, srl_collector, hparams, train_it):
    # MASTER_THESIS Here is where the magic happens. We need to add the SRL to the training loop and add encoding prior to the dynamics update!
    # MASTER_THESIS Implement real training

    print("Training Vision-Based SRL")
    ts = time.time()

    # Data Loader
    train_set, _ = srl_collector.get_dataset(task_id)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams.vision_params.bs, shuffle=True,
                                               drop_last=False, num_workers=hparams.num_ds_worker)

    # GPUID
    gpuid = hparams.gpuid

    regged_outputs = None

    fisher_ests, si_omega = None, None
    if hnet:
        targets, priors, forward_misc, inverse_misc, theta_optimizer, emb_optimizer, regularized_params = trainer_misc
    else:
        priors, forward_misc, inverse_misc = trainer_misc
        priors, theta_optimizer = priors
        regularized_params = list(mnet.parameters())
    forward_model, forward_model_optimizer, forward_model_loss_fn = forward_misc
    inverse_model, inverse_model_optimizer, inverse_model_loss_fn = inverse_misc

    # Whether the regularizer will be computed during training?
    calc_reg = task_id > 0 and hparams.beta > 0

    train_forward_model = hparams.vision_params.use_forward_model
    train_inverse_model = hparams.vision_params.use_inverse_model

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
    while it < hparams.vision_params.train_vision_iters:
        mnet.to(gpuid)
        hnet and hnet.to(gpuid)
        train_forward_model and forward_model.to(gpuid)
        train_inverse_model and inverse_model.to(gpuid)
        mnet.train()
        hnet and hnet.train()
        train_forward_model and forward_model.train()
        train_inverse_model and inverse_model.train()

        for i, data in enumerate(train_loader):
            regularized_params = list(mnet.parameters()) if not hnet else regularized_params

            iter_time = time.time()
            if it % 20 == 0:
                print(f'Iteration: {it} / {hparams.vision_params.train_vision_iters}')

            # Train theta and task embedding.
            theta_optimizer.zero_grad()
            if hnet:
                emb_optimizer.zero_grad()

            if hnet:
                with timeit_context(total_time_hnet):
                    mnet_weights = hnet.forward(task_id)

            idx, x_t, a_t, x_tt, rewards = data
            x_t, a_t, x_tt, rewards = x_t.to(gpuid), a_t.to(gpuid), x_tt.to(gpuid), rewards.to(gpuid)

            if hparams.model == "hnet_si":
                si.si_update_optim_step(mnet, mnet_weights, task_id)
                for weight in mnet_weights:
                    weight.retain_grad()  # save grad for calculate si path integral

            x_t = mnet.forward(x_t, mnet_weights)
            x_tt = mnet.forward(x_tt, mnet_weights)

            if train_forward_model:
                # Forward model loss
                with timeit_context(total_time_forward):
                    forward_model_optimizer.zero_grad()
                    x_a_t = torch.hstack((x_t, a_t))
                    predicted_x_tt = forward_model(x_a_t)
                    loss_forward_model = forward_model_loss_fn(x_tt, predicted_x_tt)
            else:
                loss_forward_model = 0

            if train_inverse_model:
                # Inverse model loss
                with timeit_context(total_time_inverse):
                    inverse_model_optimizer.zero_grad()
                    predicted_actions = inverse_model(x_t)
                    loss_inverse_model = inverse_model_loss_fn(a_t, predicted_actions)
            else:
                loss_inverse_model = 0

            # Robotic Priors loss
            with timeit_context(total_time_priors):
                loss_priors, loss_slowness, loss_variability, loss_proportionality, loss_repeatability, loss_causality \
                    = calculate_priors(priors, gpuid, hparams, idx, mnet, mnet_weights, srl_collector, task_id, x_t,
                                       x_tt, rewards)

            with timeit_context(total_time_hnet_optimize):
                # We already compute the gradients, to then be able to compute delta
                # theta.
                loss_task = loss_priors + loss_forward_model + loss_inverse_model
                loss_task.backward(retain_graph=True,
                                   create_graph=hnet is not None and hparams.vision_params.backprop_dt and calc_reg)
                # train_forward_model and loss_forward_model.backward(retain_graph=True)
                # train_inverse_model and loss_inverse_model.backward(retain_graph=True)
                hnet and torch.nn.utils.clip_grad_norm_(hnet.get_task_emb(task_id), hparams.grad_max_norm)

                # The task embedding is only trained on the task-specific loss.
                # Note, the gradients accumulated so far are from "loss_task".
                hnet and emb_optimizer.step()

                # SI
                if hparams.model == "hnet_si":
                    torch.nn.utils.clip_grad_norm_(mnet_weights, hparams.grad_max_norm)
                    si.si_update_grad(mnet, mnet_weights, task_id)

                # Update Regularization
                loss_reg = torch.tensor(0., requires_grad=False)
                dTheta = None
                grad_tloss = None
                if calc_reg:
                    if i % 1000 == 0:  # Just for debugging: displaying grad magnitude.
                        grad_tloss = torch.cat([d.grad.clone().view(-1) for d in
                                                hnet.theta])
                    if hparams.no_look_ahead:
                        dTheta = None
                    else:
                        dTheta = opstep.calc_delta_theta(theta_optimizer,
                                                         hparams.use_sgd_change, lr=hparams.lr_hyper,
                                                         detach_dt=not hparams.backprop_dt)

                    if hparams.plastic_prev_tembs:
                        dTembs = dTheta[-task_id:]
                        dTheta = dTheta[:-task_id] if dTheta is not None else None
                    else:
                        dTembs = None

                    loss_reg = hreg.calc_fix_target_reg(hnet, task_id,
                                                        targets=targets, dTheta=dTheta, dTembs=dTembs, mnet=mnet,
                                                        inds_of_out_heads=regged_outputs,
                                                        fisher_estimates=fisher_ests,
                                                        si_omega=si_omega)

                    loss_reg = loss_reg * hparams.beta * x_t.size(0)

                    loss_reg.backward()

                    if grad_tloss is not None:  # Debug
                        grad_full = torch.cat([d.grad.view(-1) for d in hnet.theta])
                        # Grad of regularizer.
                        grad_diff = grad_full - grad_tloss
                        grad_diff_norm = torch.norm(grad_diff, 2)

                        # Cosine between regularizer gradient and task-specific
                        # gradient.
                        if dTheta is None:
                            dTheta = opstep.calc_delta_theta(theta_optimizer,
                                                             hparams.use_sgd_change, lr=hparams.lr_hyper,
                                                             detach_dt=not hparams.backprop_dt)
                        dT_vec = torch.cat([d.view(-1).clone() for d in dTheta])
                        grad_cos = torch.nn.functional.cosine_similarity(grad_diff.view(1, -1),
                                                                         dT_vec.view(1, -1))

                        grad_tloss = (grad_tloss, grad_full, grad_diff_norm, grad_cos)

                torch.nn.utils.clip_grad_norm_(regularized_params, hparams.vision_params.grad_max_norm)
                theta_optimizer.step()

            if train_forward_model:
                with timeit_context(total_time_forward_optimize):
                    # Train forward model
                    forward_model_optimizer.step()

            if train_inverse_model:
                with timeit_context(total_time_inverse_optimize):
                    # Train inverse model
                    inverse_model_optimizer.step()

            total_time_iter.append(time.time() - iter_time)

            mnet_weights = mnet.parameters()
            logger.train_vision_step(loss_task, loss_priors, loss_slowness, loss_variability, loss_proportionality,
                                     loss_repeatability, loss_causality, loss_forward_model, loss_inverse_model,
                                     loss_reg, dTheta, grad_tloss, mnet_weights, forward_model.parameters(),
                                     inverse_model.parameters(), x_t)
            # Validate
            # logger.validate(priors, forward_misc, inverse_misc)

            it += 1
            if it >= hparams.vision_params.train_vision_iters:
                break

    print(f"Vision-Based SRL training time", sum(total_time_iter))
    logger.writer.add_scalar('time/train_srl', sum(total_time_iter), (train_it + 1) * it)
    logger.writer.add_scalar('time/train_srl_iter', np.mean(total_time_iter), (train_it + 1) * it)
    logger.writer.add_scalar('time/train_srl_hnet', np.mean(total_time_hnet), (train_it + 1) * it)
    logger.writer.add_scalar('time/train_srl_hnet_optimize', np.mean(total_time_hnet_optimize), (train_it + 1) * it)
    logger.writer.add_scalar('time/train_srl_forward', np.mean(total_time_forward), (train_it + 1) * it)
    logger.writer.add_scalar('time/train_srl_forward_optimize', np.mean(total_time_forward_optimize),
                             (train_it + 1) * it)
    logger.writer.add_scalar('time/train_srl_inverse', np.mean(total_time_inverse), (train_it + 1) * it)
    logger.writer.add_scalar('time/train_srl_inverse_optimize', np.mean(total_time_inverse_optimize),
                             (train_it + 1) * it)
    logger.writer.add_scalar('time/calculate_priors', np.mean(total_time_priors), (train_it + 1) * it)


def calculate_priors(priors, gpuid, hparams, idx, mnet, mnet_weights, srl_collector, task_id, x_t, x_tt, r):
    loss_priors = 0
    if not hparams.vision_params.use_priors:
        return 0, 0, 0, 0, 0

    slowness_prior, variability_prior, proportionality_prior, repeatability_prior, causality_prior = priors

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
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        other_states = torch.cat(other_states)
        next_other_states = torch.cat(next_other_states)
        rewards = torch.cat(rewards)
        other_rewards = torch.cat(other_rewards)
        priors_loader = torch.utils.data.DataLoader(
            TensorDataset(states, next_states, other_states, next_other_states, rewards, other_rewards),
            batch_size=hparams.bs,
            shuffle=True, drop_last=False, num_workers=hparams.num_ds_worker)
        loss_proportionality = 0
        loss_repeatability = 0
        loss_causality = 0
        for i, data in enumerate(priors_loader):
            states, next_states, other_states, next_other_states, rewards, other_rewards = data
            other_states = mnet.forward(other_states.to(gpuid), mnet_weights)
            next_other_states = mnet.forward(next_other_states.to(gpuid), mnet_weights)
            loss_proportionality += proportionality_prior(states, next_states, other_states, next_other_states)
            loss_repeatability += repeatability_prior(states, next_states, other_states, next_other_states)
            loss_causality += causality_prior(states, next_states, rewards.to(gpuid), other_rewards.to(gpuid))

        # scale = 10
        # loss_repeatability *= scale
        # loss_proportionality *= scale
        loss_priors += loss_proportionality
        loss_priors += loss_repeatability
        loss_priors += loss_causality
    return loss_priors, loss_slowness, loss_variability, loss_proportionality, loss_repeatability, loss_causality
