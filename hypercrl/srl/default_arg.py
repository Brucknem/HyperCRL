import time

from hypercrl.srl import RMSELoss
from hypercrl.tools import Hparams


def add_vision_params(hparams, srl_model):
    if srl_model == "gt":
        return default_vision_params(hparams, True)
    raise "Not implemented"


class VisionParams(Hparams):
    def __init__(self):
        pass

    @staticmethod
    def add_hnet_hparams(hparams, env):
        # Hypernetwork
        # hparams.hnet_arch = hdims_to_hnet_arch(hparams.h_dims)

        if env == "door":
            hparams.hnet_act = "elu"
        elif env == "door_pose":
            hparams.hnet_act = "relu"
        elif env == "pusher":
            hparams.hnet_act = "elu"
        else:
            hparams.hnet_act = 'relu'

        # Embedding
        hparams.emb_size = 10
        # Initialization
        hparams.use_hyperfan_init = False
        hparams.hnet_init = "xavier"  # or "normal"
        hparams.std_normal_init = 0.02
        hparams.std_normal_temb = 1  # std when initializing task embedding

        # Training param
        # MASTER_THESIS Check LR
        # hparams.lr_hyper = 0.0001
        hparams.grad_max_norm = 5

        if env == "door_pose" or env == "pusher_slide":
            hparams.beta = 0.5
        else:
            hparams.beta = 0.05

        hparams.no_look_ahead = False  # False=use two step optimization
        hparams.plastic_prev_tembs = False  # Allow adaptation of past task embeddings
        hparams.backprop_dt = False  # Allow backpropagation through delta theta in the regularizer
        hparams.use_sgd_change = False  # Approximate change with in delta theta with SGD
        hparams.ewc_weight_importance = False  # Use fisher matrix to regularize
        # model weights generated from hnet
        hparams.n_fisher = -1  # Number of training samples to be used for the ' +
        # 'estimation of the diagonal Fisher elements. If ' +
        # "-1", all training samples are us

        hparams.si_eps = 1e-3
        hparams.mlp_var_minmax = True

        return hparams


def default_vision_params_model(h_dims=4096, depth=1, bn=True, dropout=-1, lr=1e-3, reg_str=1e-3, loss_fn=RMSELoss()):
    params = Hparams()
    params.h_dims = h_dims if isinstance(h_dims, list) or isinstance(h_dims, tuple) else [h_dims] * depth
    params.lr = lr
    params.bn = bn
    params.reg_str = reg_str
    params.dropout = dropout
    params.loss_fn = loss_fn
    params.out_var = False
    return params


def default_vision_params_encoder(h_dims=4096, depth=1, bn=True, dropout=-1, lr=1e-3, reg_str=1e-3, loss_fn=RMSELoss()):
    return default_vision_params_model(h_dims=h_dims, depth=depth, bn=bn, dropout=dropout, lr=lr, reg_str=reg_str,
                                       loss_fn=loss_fn)


def default_vision_params_forward(h_dims=512, depth=4, bn=True, dropout=-1, lr=1e-3, reg_str=1e-3, loss_fn=RMSELoss()):
    return default_vision_params_model(h_dims=h_dims, depth=depth, bn=bn, dropout=dropout, lr=lr, reg_str=reg_str,
                                       loss_fn=loss_fn)


def default_vision_params_inverse(h_dims=4096, depth=1, bn=True, dropout=-1, lr=1e-3, reg_str=1e-3, loss_fn=RMSELoss()):
    return default_vision_params_model(h_dims=h_dims, depth=depth, bn=bn, dropout=dropout, lr=lr, reg_str=reg_str,
                                       loss_fn=loss_fn)


def default_vision_params_gt(h_dims=[], depth=1, bn=True, dropout=-1, lr=1e-3, reg_str=1e-3, loss_fn=RMSELoss()):
    return default_vision_params_model(h_dims=h_dims, depth=depth, bn=bn, dropout=dropout, lr=lr, reg_str=reg_str,
                                       loss_fn=loss_fn)


def default_vision_params(hparams, gt=False):
    hparams.numerical_state_dim = hparams.state_dim
    hparams.state_dim = 512
    hparams.out_dim = hparams.state_dim
    hparams.dnn_out = "state"
    # hparams.normalize_xu = False

    vision_params = Hparams()
    vision_params.eval_every = 5000

    vision_params.model = "gt"
    vision_params.grad_max_norm = 5

    vision_params.camera_widths = 224
    vision_params.camera_heights = 224
    vision_params.collector_max_capacity = 1000000000

    vision_params.dont_train_srl = False
    vision_params.bs = 100

    vision_params.in_dim = 512

    vision_params.srl_update_every = 1000
    vision_params.train_vision_iters = 50_000
    vision_params.print_train_every = 1000
    vision_params.sample_known_action_prob = 0.75

    vision_params.add_sin_cos_to_state = False
    vision_params.encoder_model = default_vision_params_encoder()
    vision_params.train_on_gt_model = True
    vision_params.gt_model = default_vision_params_gt()
    vision_params.use_forward_model = False
    vision_params.forward_model = default_vision_params_forward()
    vision_params.use_inverse_model = False
    vision_params.inverse_model = default_vision_params_inverse()
    vision_params.use_priors = False
    vision_params.use_only_fast_priors = False

    vision_params.debug_visualization = True
    vision_params.save_path = "/mnt/local_data/datasets/master-thesis"
    vision_params.save_every = 100
    vision_params.exit_after_save = True
    vision_params.load_max = -1

    vision_params.load_suffix = "100000/1634326474"

    vision_params.save_suffix = int(time.time())

    hparams.vision_params = vision_params

    return hparams
