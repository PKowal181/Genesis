import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

# Load the log file
log_filename = "/home/pawel/Desktop/GitLab/Genesis/flight_log_simple_roll-pitch-yaw_12.csv"
df = pd.read_csv(log_filename)

# Convert Time to seconds since start
df["Time"] -= df["Time"].iloc[0]

df["Throttle"] = df["Throttle"].map(lambda x: literal_eval(x))

# Plot reference vs actual for each parameter
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

params = [("Roll", "Roll_Ref", "Roll_Actual"),
          ("Pitch", "Pitch_Ref", "Pitch_Actual"),
          ("Yaw Rate", "Yaw_Rate_Ref", "Yaw_Rate_Actual"),
        #   ("Throttle", "Throttle", "Throttle")
        ]

for i, (title, ref, actual) in enumerate(params):
    axes[i].plot(df["Time"], df[ref], label=f"{title} Reference", linestyle="dashed")
    axes[i].plot(df["Time"], df[actual], label=f"{title} Actual")
    axes[i].set_ylabel(title)
    axes[i].legend()
    axes[i].grid()

axes[len(params)].plot(df["Time"], df["Throttle"].str[0])
axes[len(params)].plot(df["Time"], df["Throttle"].str[1])

axes[-1].set_xlabel("Time (s)")
plt.suptitle("Reference vs Actual Flight Data")
plt.show()
