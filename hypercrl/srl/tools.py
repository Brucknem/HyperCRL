import torch

from hypercrl.hypercl import MLP
from hypercrl.model.tools import initialize_hnet, calculate_compression_ratio, generate_hnet
from hypercrl.srl import ResNet18Encoder
from hypercrl.srl.datautil import DataCollector
from hypercrl.hypercl.utils import hnet_regularizer as hreg
from hypercrl.srl.robotic_priors import RoboticPriors


def build_vision_model_hnet(hparams):
    encoder_mnet = MLP(n_in=hparams.vision_params.in_dim,
                       n_out=hparams.state_dim, hidden_layers=hparams.vision_params.hdims,
                       no_weights=True, out_var=hparams.vision_params.out_var,
                       mlp_var_minmax=hparams.mlp_var_minmax)

    print('Constructed Vision MLP with shapes: ', encoder_mnet.param_shapes)
    encoder_mnet = ResNet18Encoder(encoder_mnet, hparams)

    hnet = generate_hnet(hparams, encoder_mnet.param_shapes)
    calculate_compression_ratio(hnet, hparams, encoder_mnet.param_shapes)
    initialize_hnet(hnet, hparams)

    return encoder_mnet, hnet


def reload_vision_model_hnet(hparams):
    encoder_mnet, encoder_hnet = build_vision_model_hnet(hparams)

    checkpoint = None
    collector = DataCollector(hparams)
    # MASTER_THESIS really load models

    return encoder_mnet, encoder_hnet, checkpoint, collector


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

    mll = RoboticPriors()

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
    theta_optimizer = torch.optim.Adam(regularized_params, lr=hparams.lr_hyper)
    # We only optimize the task embedding corresponding to the current task,
    # the remaining ones stay constant.
    emb_optimizer = torch.optim.Adam([hnet.get_task_emb(task_id)], lr=hparams.lr_hyper)

    trainer_misc = (targets, mll, theta_optimizer, emb_optimizer, regularized_params)  # , fisher_ests, si_omega)

    return trainer_misc
