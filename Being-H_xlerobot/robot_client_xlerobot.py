"""Laptop-side XLerobot client: read sensors, query server, execute actions.

Assumptions (per your setup):
- COM4: left arm + head motors
- COM3: right arm + base motors
- Camera 0: head (center) camera
- Camera 1: left wrist camera
- Camera 2: right wrist camera (unused here)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


LEFT_ARM_JOINTS = [
    "left_arm_shoulder_pan.pos",
    "left_arm_shoulder_lift.pos",
    "left_arm_elbow_flex.pos",
    "left_arm_wrist_flex.pos",
    "left_arm_wrist_roll.pos",
    "left_arm_gripper.pos",
]


def _add_repo_paths() -> None:
    # Make local packages importable without pip-installing the repo.
    import sys

    vla_root = Path(__file__).resolve().parents[1]
    beingh_root = vla_root / "Being-H"
    lerobot_root = vla_root / "lerobot" / "src"

    for p in (str(lerobot_root), str(beingh_root), str(vla_root / "Being-H_xlerobot")):
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> None:
    _add_repo_paths()

    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.xlerobot import XLerobot, XLerobotConfig

    from BeingH.inference.beingh_service import BeingHInferenceClient

    p = argparse.ArgumentParser()
    p.add_argument("--server_host", required=True, help="GPU server IP/hostname running run_server_xlerobot.py")
    p.add_argument("--server_port", type=int, default=5555)
    p.add_argument("--api_token", default=None)

    p.add_argument("--port1", default="COM4", help="Left arm + head motors serial port (COM4).")
    p.add_argument("--port2", default="COM3", help="Right arm + base motors serial port (COM3).")

    p.add_argument("--head_cam", type=int, default=0, help="OpenCV camera index for head camera (0).")
    p.add_argument("--left_wrist_cam", type=int, default=1, help="OpenCV camera index for left wrist camera (1).")

    p.add_argument(
        "--instruction",
        default="Put the box into the robot basket. 将桌子上的盒子放入机器人篮子里。",
        help="Task instruction string sent to the policy.",
    )

    p.add_argument("--hz", type=float, default=10.0, help="Control loop frequency (open-loop between replans).")
    p.add_argument("--replan_every", type=int, default=16, help="Re-query the server every N executed actions.")
    p.add_argument("--steps", type=int, default=160, help="Total control steps to run before exiting.")

    p.add_argument("--calibrate", action="store_true", default=False, help="Run robot calibration on connect.")
    p.add_argument(
        "--max_relative_target",
        type=int,
        default=10,
        help="Safety clamp: max allowed joint delta per send_action call (degrees-ish). Set 0/None to disable.",
    )
    p.add_argument(
        "--use_degrees",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use degrees for arm/head joints (matches dataset stats).",
    )

    args = p.parse_args()

    cameras = {
        "head": OpenCVCameraConfig(
            index_or_path=args.head_cam,
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
        "left_wrist": OpenCVCameraConfig(
            index_or_path=args.left_wrist_cam,
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
    }

    robot_cfg = XLerobotConfig(
        id="xlerobot_laptop",
        port1=args.port1,
        port2=args.port2,
        cameras=cameras,
        left_arm_only=True,
        use_degrees=args.use_degrees,
        max_relative_target=(None if args.max_relative_target in (0, -1) else args.max_relative_target),
    )
    robot = XLerobot(robot_cfg)

    client = BeingHInferenceClient(host=args.server_host, port=args.server_port, api_token=args.api_token)

    # Connect robot (may prompt to restore calibration file).
    robot.connect(calibrate=args.calibrate)

    time_per_step = 1.0 / max(args.hz, 1e-6)
    action_buffer: list[list[float]] = []
    action_idx = 0

    try:
        for step in range(args.steps):
            t0 = time.time()

            obs = robot.get_observation()

            # Build policy observation dict (matches xlerobot_box2basket_left_head DataConfig).
            state_vec = np.array([obs[j] for j in LEFT_ARM_JOINTS], dtype=np.float32)
            req = {
                "video.head": obs["head"],
                "video.left_wrist": obs["left_wrist"],
                "state.left_arm": state_vec,
                "language.instruction": [args.instruction],
            }

            if action_idx == 0 or (step % args.replan_every == 0):
                resp = client.get_action(req)
                action_buffer = resp["action.left_arm"]
                action_idx = 0

            act = np.array(action_buffer[action_idx], dtype=np.float32)
            action_idx = (action_idx + 1) % len(action_buffer)

            action_dict = {LEFT_ARM_JOINTS[i]: float(act[i]) for i in range(min(len(act), len(LEFT_ARM_JOINTS)))}
            robot.send_action(action_dict)

            dt = time.time() - t0
            if dt < time_per_step:
                time.sleep(time_per_step - dt)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
