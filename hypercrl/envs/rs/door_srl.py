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
from hypercrl.srl import SRL, SRLTrainer


class PandaDoorSRL(PandaDoor):
    """
    This class corresponds to the lifting task for a single robot arm.
    """

    def __init__(
            self,
            robots,
            handle_type="lever",
            mass_scale=1.0,
            handle_ypos=0.0,
            joint_range=[-1.57, 1.57],
            pose_control=False,
            controller_configs=None,
            gripper_types="PandaGripper",
            gripper_visualizations=False,
            initialization_noise="default",
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=2.25,
            reward_shaping=True,
            placement_initializer=None,
            use_indicator_object=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=10,
            horizon=220,
            ignore_done=False,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
    ):
        super().__init__(
            robots=robots,
            handle_type=handle_type,
            mass_scale=mass_scale,
            handle_ypos=handle_ypos,
            joint_range=joint_range,
            pose_control=pose_control,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            gripper_visualizations=gripper_visualizations,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_scale=reward_scale,
            reward_shaping=reward_shaping,
            placement_initializer=placement_initializer,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths
        )
        """
        Args:
            robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
                (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
                Note: Must be a single single-arm robot!

            controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
                custom controller. Else, uses the default controller for this specific task. Should either be single
                dict if same controller is to be used for all robots or else it should be a list of the same length as
                "robots" param

            gripper_types (str or list of str): type of gripper, used to instantiate
                gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
                with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
                overrides the default gripper. Should either be single str if same gripper type is to be used for all
                robots or else it should be a list of the same length as "robots" param

            gripper_visualizations (bool or list of bool): True if using gripper visualization.
                Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all
                robots or else it should be a list of the same length as "robots" param

            initialization_noise (float or list of floats): The scale factor of uni-variate Gaussian random noise
                applied to each of a robot's given initial joint positions. Setting this value to "None" or 0.0 results
                in no noise being applied. Should either be single float if same noise value is to be used for all
                robots or else it should be a list of the same length as "robots" param

            use_camera_obs (bool): if True, every observation includes rendered image(s)

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_scale (float): Scales the normalized reward function by the amount specified

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering

            render_camera (str): Name of camera to render if `has_renderer` is True.

            render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

            control_freq (float): how many control signals to receive in every second. This sets the amount of
                simulation time that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_names (str or list of str): name of camera to be rendered. Should either be single str if
                same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
                Note: At least one camera must be specified if @use_camera_obs is True.
                Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                    convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                    robot's camera list).

            camera_heights (int or list of int): height of camera frame. Should either be single int if
                same height is to be used for all cameras' frames or else it should be a list of the same length as
                "camera names" param.

            camera_widths (int or list of int): width of camera frame. Should either be single int if
                same width is to be used for all cameras' frames or else it should be a list of the same length as
                "camera names" param.

            camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
                bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
                "camera names" param.

        """


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
        srl=hypercrl.srl.SRL
    )
    print("Model Timestep", env.model_timestep, "Control Timestep", env.control_timestep)
    x_t = env.reset()
    # env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec
    print(x_t, low, high)

    # do visualization
    for i in range(10000):
        # env.render()
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)

        for key, value in obs.items():
            if str(key).endswith("_image"):
                image = cv2.flip(value, 0)
                cv2.imshow(key, image)
        cv2.waitKey(1)

        if render:
            env.render()
        if done:
            env.reset()

    cv2.destroyAllWindows()
