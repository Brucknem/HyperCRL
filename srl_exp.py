import torch

import hypercrl.tools.default_arg
from hypercrl.srl import DataCollector, build_vision_model_hnet
from hypercrl.srl.models import ResNet18Encoder
from hypercrl.srl.monitor import MonitorSRL
from hypercrl.srl.tools import augment_model

if __name__ == "__main__":
    hparams = hypercrl.tools.default_arg.HP(env="door_pose", robot="Panda", vision=True, seed=777, resume=False,
                                            save_folder="./srl/door_pose")
    hparams.model = "hnet"
    hypercrl.tools.default_arg.Hparams.add_hnet_hparams(hparams.vision_params, hparams.env)
    srl_collector = DataCollector(hparams)
    srl_collector.load()

    mlp = hypercrl.srl.models.MLP(512, hparams.vision_params.h_dims, hparams.state_dim)
    encoder_mnet = ResNet18Encoder(mlp, hparams.vision_params)
    monitor_srl = MonitorSRL(hparams, encoder_mnet, srl_collector, False)
    trainer_misc = hypercrl.srl.tools.generate_srl_losses(hparams)
    trainer_misc = (
        (trainer_misc[0], torch.optim.Adam(encoder_mnet.parameters(), lr=hparams.vision_params.lr_hyper)),
        trainer_misc[1],
        trainer_misc[2])
    hypercrl.srl.tools.train(0, encoder_mnet, None, trainer_misc, monitor_srl, srl_collector, hparams, 0)

    print("Yeet")
