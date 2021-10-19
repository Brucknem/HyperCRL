import torch
from torchvision import models

from hypercrl.control import RandomAgent
from hypercrl.envs.cl_env import CLEnvHandler
from hypercrl.hnet_exp import collect_random_data
from hypercrl.srl import DataCollector, add_vision_params
from hypercrl.tools import reset_seed, HP


def run(hparams):
    reset_seed(hparams.seed)
    image_dims = (hparams.vision_params.camera_widths, hparams.vision_params.camera_heights, 3)

    envs = CLEnvHandler(hparams.env, hparams.robot, hparams.seed, image_dims=image_dims)

    for task_id in range(hparams.num_tasks):
        # New Task with different friction
        env = envs.add_task(task_id, render=False)

        srl_collector = DataCollector(hparams)

        # Random Policy
        rand_pi = RandomAgent(hparams, env.action_spec)

        with torch.no_grad():
            collect_random_data(task_id, env, hparams, None, rand_pi, srl_collector)


if __name__ == "__main__":
    for x in [100000, 1_000_000]:
        for y in [1, 2, 5]:
            size = x * y
            hparams = HP(env="door_pose", robot="Panda", seed=777, resume=False, save_folder=f"{size}")
            hparams.model = "single"
            hparams.num_tasks = 1
            add_vision_params(hparams, "gt")
            hparams.vision_params.save_path += f'/{size}'
            hparams.init_rand_steps = size
            run(hparams)
