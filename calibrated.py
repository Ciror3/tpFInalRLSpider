import csv
from pathlib import Path
import numpy as np

ACTIONS = {
    0: (227, "Pivot_Left"),
    1: (228, "Pivot_Right"),
    2: (251, "FwdSteer_Left"),
    3: (252, "Fwd"),
    4: (253, "FwdSteer_Right"),
    5: (256, "BwdSteer_Left"),
    6: (258, "BwdSteer_Right"),
    7: (262, "FastFwd"),
    8: (266, "Bwd"),
    9: (267, "FastBwd"),
    10: (261, "FastFwdSteer_Left"),
    11: (263, "FastFwdSteer_Right"),
}

def build_previous_commands(
    calibration_path="tpFInalRLSpider/calibration_results_combinations_full.txt",
):
    motion_to_action = {motion_name: action for action, (_, motion_name) in ACTIONS.items()}

    previous_commands = {}
    with Path(calibration_path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)

        for row in reader:
            source_motion = row["Source_Motion"].strip()
            target_motion = row["Target_Motion"].strip()
            target_id = int(row["Target_ID"])

            delta_fwd = float(row["Avg_Delta_Fwd(m)"])
            delta_lat = float(row["Avg_Delta_Lat(m)"])
            delta_theta = np.deg2rad(float(row["Avg_Delta_Theta(deg)"]))

            action = motion_to_action[target_motion]

            # Formato compatible con commands: action -> (comando, [dx, dy, dtheta])
            # Uso esta conversion porque en tu calibracion el avance aparece en Avg_Delta_Lat.
            movement = [delta_lat, -delta_fwd, delta_theta]

            previous_commands.setdefault(source_motion, {})
            previous_commands[source_motion][action] = (target_id, movement)

    return previous_commands


commands_previous_steps = build_previous_commands()
print(commands_previous_steps)