"""Laptop-side XLerobot client: read sensors, query Being-H server, execute actions.

Typical wiring (per your setup):
- COM4: left arm + head motors (bus1 / port1)
- COM3: right arm + base motors (bus2 / port2)
- Camera 0: head (center) camera
- Camera 1: left wrist camera
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


def _ensure_time1_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure an RGB uint8 image has a leading time dimension of 1: (1, H, W, 3)."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray image, got: {type(img)}")
    if img.ndim == 3:
        return img[None, ...]
    if img.ndim == 4:
        return img
    raise ValueError(f"Expected HxWx3 or 1xHxWx3, got shape: {img.shape}")


def main() -> None:
    _add_repo_paths()

    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.xlerobot import XLerobot, XLerobotConfig

    from BeingH.inference.beingh_service import BeingHInferenceClient

    p = argparse.ArgumentParser()
    p.add_argument("--server_host", required=True, help="GPU server IP/hostname running Being-H inference server")
    p.add_argument("--server_port", type=int, default=5555)
    p.add_argument("--api_token", default=None)

    p.add_argument("--port1", default="COM4", help="Left arm + head motors serial port (COM4).")
    p.add_argument("--port2", default="COM3", help="Right arm + base motors serial port (COM3). Ignored if bus2 is disabled.")
    p.add_argument(
        "--connect-bus2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Connect bus2 (right arm + base). Use --no-connect-bus2 to disable.",
    )

    p.add_argument("--head_cam", type=int, default=0, help="OpenCV camera index for head camera (0).")
    p.add_argument("--left_wrist_cam", type=int, default=1, help="OpenCV camera index for left wrist camera (1).")

    p.add_argument(
        "--instruction",
        # Keep this ASCII-only by default to avoid Windows terminal encoding issues.
        default="Put the box into the robot basket.",
        help="Task instruction string sent to the policy.",
    )

    p.add_argument("--hz", type=float, default=10.0, help="Control loop frequency (open-loop between replans).")
    p.add_argument("--replan_every", type=int, default=16, help="Re-query the server every N executed actions.")
    p.add_argument("--steps", type=int, default=160, help="Total control steps to run before exiting.")

    # Default to calibrate=True so first-time users don't crash with "has no calibration registered".
    # If you know you already have a valid calibration file and want to skip any prompts, pass --no-calibrate.
    p.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calibrate the robot on connect if needed (recommended).",
    )
    p.add_argument(
        "--max_relative_target",
        type=int,
        default=10,
        help="Safety clamp: max allowed joint delta per send_action call (degrees-ish). Set 0/-1 to disable.",
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
        port2=(args.port2 if args.connect_bus2 else None),
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
    act_chunk: np.ndarray | None = None  # (chunk, 6)
    action_idx = 0

    try:
        for step in range(args.steps):
            t0 = time.time()

            obs = robot.get_observation()

            # Build policy observation dict (matches xlerobot_box2basket_left_head DataConfig).
            head = _ensure_time1_rgb(obs["head"])
            left_wrist = _ensure_time1_rgb(obs["left_wrist"])
            state6 = np.asarray([obs[j] for j in LEFT_ARM_JOINTS], dtype=np.float32)[None, :]  # (1, 6)

            req = {
                "video.head": head,
                "video.left_wrist": left_wrist,
                "state.left_arm": state6,
                "language.instruction": [args.instruction],
            }

            if act_chunk is None or (step % args.replan_every == 0) or (action_idx >= act_chunk.shape[0]):
                resp = client.get_action(req)
                a = np.asarray(resp["action.left_arm"], dtype=np.float32)
                # Server may return (chunk, 6) or (1, chunk, 6) depending on batching detection.
                if a.ndim == 3:
                    a = a[0]
                if a.ndim != 2 or a.shape[-1] != len(LEFT_ARM_JOINTS):
                    raise ValueError(f"Unexpected action.left_arm shape: {a.shape}")
                act_chunk = a
                action_idx = 0

            act = act_chunk[action_idx]
            action_idx += 1

            action_dict = {LEFT_ARM_JOINTS[i]: float(act[i]) for i in range(len(LEFT_ARM_JOINTS))}
            robot.send_action(action_dict)

            dt = time.time() - t0
            if dt < time_per_step:
                time.sleep(time_per_step - dt)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
