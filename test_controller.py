# import inputs
# import torch
# import threading
# import time

# # Persistent command tensor
# commands = torch.zeros((1, 4), dtype=torch.float32)

# def handle_controller_input():
#     global commands
#     while True:
#         events = inputs.get_gamepad()
#         for event in events:
#             if event.ev_type == "Absolute":
#                 # Normalize and update persistent command values
#                 if event.code == "ABS_Y":  # Left stick Y-axis
#                     commands[0, 0] = -event.state / 32767.0  
#                 elif event.code == "ABS_X":  # Left stick X-axis
#                     commands[0, 3] = event.state / 32767.0 * 180.0
#                 elif event.code == "ABS_RY":  # Right stick Y-axis
#                     commands[0, 2] = -event.state / 32767.0 * 45.0
#                 elif event.code == "ABS_RX":  # Right stick X-axis
#                     commands[0, 1] = event.state / 32767.0 * 45.0

#         # Ensure values are within range
#         commands = torch.clip(
#             commands,
#             torch.tensor([[0.0, -45.0, -45.0, -180.0]], dtype=torch.float32),
#             torch.tensor([[1.0, 45.0, 45.0, 180.0]], dtype=torch.float32)
#         )

# # Start controller input in a separate thread
# controller_thread = threading.Thread(target=handle_controller_input, daemon=True)
# controller_thread.start()

# # Example simulation loop
# while True:
#     print("Simulation running... Current commands:", commands)
#     time.sleep(0.1)  # Simulate a time step (adjust as needed)


import time
import torch
# values = torch.zeros((100,))

# env_idx = torch.randint(0, 100, (10,)).unique()
# choices = torch.randint(1, 4, (len(env_idx),))

# print(choices)


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

commands = torch.tensor([[[0.0, 0.0, 0.0]],
                         [[0.0, 1.0, 0.0]],
                         [[0.0, 0.0, 2.0]],])

path = Path(commands, 1)

path.start()

while (commands := path.setpoint()) is not None:
    print(commands[:, :])