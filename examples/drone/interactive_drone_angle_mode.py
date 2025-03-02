import torch
import math
import genesis as gs
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from pynput import keyboard

import argparse
import os
import pickle

from rsl_rl.runners import OnPolicyRunner


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class DroneController:
    def __init__(self):
        self.running = True
        self.pressed_keys = set()

        self.commands = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=gs.device, dtype=gs.tc_float
        )
        self.altitude_delta = 0.05
        self.roll_delta = 0.5
        self.pitch_delta = 0.5
        self.yaw_delta = 0.5


    def on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.running = False
                return False
            self.pressed_keys.add(key)
            print(f"Key pressed: {key}")
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.discard(key)
        except KeyError:
            pass

    def update_setpoints(self):
        # Altitide up
        if keyboard.Key.shift_l in self.pressed_keys:
            self.commands[0, 0] += self.altitude_delta

        # Altitide down
        if keyboard.Key.ctrl_l in self.pressed_keys:
            self.commands[0, 0] -= self.altitude_delta

        # Roll up
        if keyboard.Key.left in self.pressed_keys:
            self.commands[0, 1] -= self.roll_delta

        # Roll down
        if keyboard.Key.right in self.pressed_keys:
            self.commands[0, 1] += self.roll_delta

        # Pitch up
        if keyboard.Key.up in self.pressed_keys:
            self.commands[0, 2] += self.roll_delta

        # Pitch down
        if keyboard.Key.down in self.pressed_keys:
            self.commands[0, 2] -= self.roll_delta

        # Yaw ccw
        if keyboard.Key.alt_r in self.pressed_keys:
            self.commands[0, 3] += self.yaw_delta

        # Yaw cw
        if keyboard.Key.ctrl_r in self.pressed_keys:
            self.commands[0, 3] -= self.yaw_delta


        # return to (x, 0.0, 0.0, y)

        if keyboard.Key.up not in self.pressed_keys and keyboard.Key.down not in self.pressed_keys and self.commands[0, 2] != 0.0:
            self.commands[0, 2] -= self.pitch_delta if self.commands[0, 2] > 0 else -self.pitch_delta

        if keyboard.Key.left not in self.pressed_keys and keyboard.Key.right not in self.pressed_keys and self.commands[0, 1] != 0.0:
            self.commands[0, 1] -= self.roll_delta if self.commands[0, 1] > 0 else -self.roll_delta

        return self.commands


def update_camera(scene, drone: DroneEntity):
    """Updates the camera position to follow the drone"""
    if not scene.viewer:
        return

    drone_pos = drone.get_pos()

    # Camera position relative to drone
    offset_x = 0.0  # centered horizontally
    offset_y = -2.0  # 4 units behind (in Y axis)
    offset_z = 1.0  # 2 units above

    camera_pos = (
        float(drone_pos[0, 0] + offset_x),
        float(drone_pos[0, 1] + offset_y),
        float(drone_pos[0, 2] + offset_z),
    )

    # Update camera position and look target
    scene.viewer.set_camera_pose(
        pos=camera_pos, lookat=tuple(drone_pos.cpu().flatten().tolist())
    )


class HoverEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer=False,
        device="cuda",
    ):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            show_FPS=False,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone
        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"], device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=self.device
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone: DroneEntity = self.scene.add_entity(
            gs.morphs.Drone(file="urdf/drones/cf2x.urdf")
        )

        # build scene
        self.scene.build(n_envs=num_envs)

        # # prepare reward functions and multiply reward scales by dt
        # self.reward_functions, self.episode_sums = dict(), dict()
        # for name in self.reward_scales.keys():
        #     self.reward_scales[name] *= self.dt
        #     self.reward_functions[name] = getattr(self, "_reward_" + name)
        #     self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        # self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)

        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float
        )

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.commands[envs_idx, 1] = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.commands[envs_idx, 2] = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )

    def step(self, actions):
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        exec_actions = self.actions.cpu()

        # 14468 is hover rpm
        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            )
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # check termination and reset
        self.crash_condition = (
            self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"]
        )
        self.reset_buf = (
            self.episode_length_buf > self.max_episode_length
        ) | self.crash_condition

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.clip(
                    (self.base_euler[:, 0:] - self.commands[:, 1:])
                    * self.obs_scales["rel_angle"],
                    -1,
                    1,
                ),
                torch.clip(self.base_pos[:, 2:] * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        return self.obs_buf

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(
            self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )
        self.drone.set_quat(
            self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        # self.extras["episode"] = {}
        # for key in self.episode_sums.keys():
        #     self.extras["episode"]["rew_" + key] = (
        #         torch.mean(self.episode_sums[key][envs_idx]).item()
        #         / self.env_cfg["episode_length_s"]
        #     )
        #     self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="simple_roll-pitch-yaw_2")
    parser.add_argument("--ckpt", type=int, default=800)
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"logs/{args.exp_name}/cfgs.pkl", "rb")
    )
    reward_cfg["reward_scales"] = {}

    # for video recording
    env_cfg["visualize_camera"] = args.record
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60

    env = HoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    # env.drone.set_pos([0,0,5.0], zero_velocity=True)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    controller = DroneController()

    listener = keyboard.Listener(on_press=controller.on_press, on_release=controller.on_release)
    listener.start()

    obs, _ = env.reset()

    with torch.no_grad():
        while controller.running:
            update_camera(env.scene, env.drone)
            controller.update_setpoints()
            env.commands = controller.commands
            actions = policy(obs)

            # actions += 0.4 if env.commands[0, 0] > 1.0 else -0.1
            actions += 0.3

            obs = env.step(actions)
            print(
                f"Altitide: {env.commands[:, 0].tolist()[0]:4.2f} -> {env.base_pos[:, 2].tolist()[0]:4.2f} | Roll: {env.commands[:, 1].tolist()[0]:4.2f} -> {env.base_euler[:, 0].tolist()[0]:4.2f} | Pitch: {env.commands[:, 2].tolist()[0]:4.2f} -> {env.base_euler[:, 1].tolist()[0]:4.2f}"
            )
    listener.stop()


if __name__ == "__main__":
    main()
