import threading
import torch
import genesis as gs
from genesis.engine.entities.drone_entity import DroneEntity

from pynput import keyboard
import inputs

from hover_env import HoverEnv

import argparse
import os
import pickle

from rsl_rl.runners import OnPolicyRunner
import csv
import time

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class DroneController:
    def __init__(self):
        self.running = True
        self.pressed_keys = set()

        self.commands = torch.tensor(
            [[3.0, 0.0, 0.0, 0.0]], device=gs.device, dtype=gs.tc_float
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

    def handle_controller_input(self):
        while self.running:
            events = inputs.get_gamepad()
            for event in events:
                if event.ev_type == "Absolute":
                    # Normalize and update persistent command values
                    if event.code == "ABS_Y":  # Left stick Y-axis
                        self.commands[0, 0] = -event.state / 32767.0  # alt
                        pass
                    elif event.code == "ABS_X":  # Left stick X-axis
                        self.commands[0, 3] = event.state / 32767.0 * 3.14  # yaw rate
                    elif event.code == "ABS_RY":  # Right stick Y-axis
                        self.commands[0, 2] = -event.state / 32767.0 * 45.0  # pitch
                    elif event.code == "ABS_RX":  # Right stick X-axis
                        self.commands[0, 1] = event.state / 32767.0 * 45.0  # roll

            # Ensure values are within range
            self.commands = torch.clip(
                self.commands,
                torch.tensor([[0.0, -45.0, -45.0, -3.14]], device=gs.device, dtype=gs.tc_float),
                torch.tensor([[3.0, 45.0, 45.0, 3.14]], device=gs.device, dtype=gs.tc_float)
            )

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

class Path:
    def __init__(self, path, interval):
        self.commands = path
        self.interval = interval
        self.current_step = 0

        self.current_command = self.commands[self.current_step]

    def start(self):
        self.last_interval = time.time()

    def setpoint(self):
        if (time.time() - self.last_interval) > self.interval:
            self.current_step += 1
            if self.current_step == len(self.commands):
                return None

            self.current_command = self.commands[self.current_step]
            self.last_interval = time.time()

        return self.current_command

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="simple_roll-pitch-yaw_12")
    parser.add_argument("--ckpt", type=int, default=300)
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

    # controller = DroneController()

    # listener = keyboard.Listener(on_press=controller.on_press, on_release=controller.on_release)
    # listener.start()

    # controller_thread = threading.Thread(target=controller.handle_controller_input, daemon=True)
    path = Path(torch.tensor([[[0.0, 30.0, 0.0]],
                              [[0.0, 30.0, 0.0]],
                              [[0.0, 0.0, 0.0]]], device=gs.device, dtype=gs.tc_float), 2)
    obs, _ = env.reset()


    log_filename = f"flight_log_{args.exp_name}.csv"

    # Open the log file and write headers
    with open(log_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Throttle",
                        "Roll_Ref", "Roll_Actual", 
                        "Pitch_Ref", "Pitch_Actual", 
                        "Yaw_Rate_Ref", "Yaw_Rate_Actual"])

    def scale_vector(motor_speeds, throttle):
        """
        Scale a vector of motor speeds (-1 to 1) based on a throttle value (0 to 1).
        
        :param motor_speeds: List of motor speeds (values between -1 and 1)
        :param throttle: Scaling factor (between 0 and 1)
        :return: Scaled motor speeds within the valid range (-1 to 1)
        """
        # Find the maximum absolute value to preserve proportions
        max_value = torch.max(torch.abs(motor_speeds))

        if max_value == 0:
            return motor_speeds  # Avoid division by zero if all values are zero

        # Compute the scaling factor needed to keep values in range while applying throttle
        scale_factor = (throttle+1) / max_value

        # Apply the scale factor but ensure values remain within [-1, 1]
        scaled_speeds = torch.clip(motor_speeds * scale_factor, -1.0, 1.0)

        return scaled_speeds

    with torch.no_grad():
        # controller_thread.start()
        path.start()
        # while controller.running:
        while (commands := path.setpoint()) is not None:
            update_camera(env.scene, env.drone)
            # controller.update_setpoints()
            # env.commands = controller.commands[:, 1:]
            env.commands = commands

            actions = policy(obs)
            # actions = scale_vector(actions, controller.commands[:, 0])
            obs = env.step(actions)[0]
            
            # Get reference and actual values
            timestamp = time.time()

            # throttle = controller.commands[:, 0].tolist()[0]

            roll_ref = env.commands[:, 0].tolist()[0]
            roll_actual = env.base_euler[:, 0].tolist()[0]

            pitch_ref = env.commands[:, 1].tolist()[0]
            pitch_actual = env.base_euler[:, 1].tolist()[0]

            yaw_rate_ref = env.commands[:, 2].tolist()[0]
            yaw_rate_actual = env.base_ang_vel[:, 2].tolist()[0]

            throttle = env.actions[:].tolist()[0]

            # Write to log file
            with open(log_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, throttle,
                                roll_ref, roll_actual, 
                                pitch_ref, pitch_actual, 
                                yaw_rate_ref, yaw_rate_actual])

            # Print log
            print(
                # f"Throttle: {throttle:4.2f} | "
                f"Actions: {actions} | "
                f"Roll: {roll_ref:4.2f} -> {roll_actual:4.2f} | "
                f"Pitch: {pitch_ref:4.2f} -> {pitch_actual:4.2f} | "
                f"Yaw_rate: {yaw_rate_ref:4.2f} -> {yaw_rate_actual:4.2f}")

    # listener.stop()

if __name__ == "__main__":
    main()
