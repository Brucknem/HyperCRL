import time
from collections import OrderedDict
import numpy as np
import os

from robosuite.utils.transform_utils import convert_quat

from robosuite.environments.robot_env import RobotEnv
from robosuite.robots import SingleArm
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string, string_to_array

from robosuite.models.arenas import Arena, TableArena
from robosuite.models.tasks import Task, UniformRandomSampler, TableTopTask

from robosuite.models.grippers import PandaGripper
from robosuite.models.objects import MujocoXMLObject
from gym.envs.robotics.rotations import (quat2euler, subtract_euler, quat_mul, \
                                         quat2axisangle, quat_conjugate, quat2mat)

import hypercrl.srl
from hypercrl.envs.rs import PandaDoor
from hypercrl.srl import SRL, SRLTrainer, SRLDataSet

if __name__ == "__main__":
    # Create dict to hold options that will be passed to env creation call
    options = {}

    options["env_name"] = "PandaDoor"
    options["handle_type"] = "lever"
    options["robots"] = "Panda"
    options["camera_names"] = ['birdview', 'frontview', 'sideview']

    # Choose controller
    controller_name = "OSC_POSE"
    options["pose_control"] = True

    from robosuite.controllers import load_controller_config

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    import robosuite as suite
    import os
    import cv2

    render = "LD_PRELOAD" in os.environ and "/usr/lib/x86_64-linux-gnu/libGLEW.so" in os.environ["LD_PRELOAD"]

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=render,
        has_offscreen_renderer=not render,
        ignore_done=False,
        use_camera_obs=not render,
        control_freq=10,
    )
    print("Model Timestep", env.model_timestep, "Control Timestep", env.control_timestep)
    x_t = env.reset()
    # env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec
    print(x_t, low, high)

    srl = SRL(128)
    srl_dataset = SRLDataSet()
    trainer = SRLTrainer(srl=srl)

    observation_name = options["camera_names"][0] + "_image"

    episode = 0

    # do visualization
    for i in range(20000):
        # env.render()

        if i % 200 == 0:
            print("Step:", i)

        index = -1
        if len(srl_dataset.data_points) > 0 and np.random.randint(0, 100) < 20:
            index, action = srl_dataset.get_known_action()
        else:
            action = np.random.uniform(low, high)

        obs, reward, done, _ = env.step(action)
        image = obs[observation_name]

        srl_dataset.add_datapoint(episode=episode, observation=image, action=action, reward=reward)

        if done:
            episode += 1

        if render:
            env.render()
        else:
            image = cv2.flip(image, 0)
            cv2.imshow(observation_name, image)
            cv2.waitKey(1)
        if done:
            env.reset()

    if not render:
        cv2.destroyAllWindows()

    srl_dataset.calculate_same_action_pairs()
    trainer.train(srl_dataset=srl_dataset, batch_size=128)
    srl_dataset.clear()
