"""
Record variable-length demonstration episodes for XLeRobot and save them in
LeRobotDataset (v3) format (Parquet + MP4 videos).

Default control mode (recommended for stable collection):
- Hold the RIGHT arm by hand (torque disabled / leader)
- LEFT arm follows the RIGHT arm (follower)
- Dataset records ONLY: left-arm joints + head (central) camera

Default setup matches the user's hardware:
- COM4: left arm + head motors
- COM3: right arm + base (base is kept stopped; no base actions are recorded)
- OpenCV cameras:
  - 0: head (central) camera
  - 1: left wrist camera
  - 2: laptop webcam (not used by this script)
  - 3: right wrist camera

Controls during recording:
- RIGHT ARROW: finish episode (save)
- LEFT ARROW: discard episode and re-record
- ESC: stop recording session (discard current episode)

Run (PowerShell):
  cd E:\\VLA\\lerobot
  python .\\examples\\record_xlerobot_keyboard_dataset.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow running without manually setting PYTHONPATH=src.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lerobot.cameras.opencv.configuration_opencv import Cv2Rotation, OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.model.SO101Robot import SO101Kinematics
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.control_utils import init_keyboard_listener


class OpenCVPreview:
    def __init__(
        self,
        enabled: bool,
        *,
        camera_keys: list[str] | None = None,
        width: int,
        height: int,
        scale: float = 1.0,
        window_name: str = "XLeRobot Cameras",
    ):
        self.enabled = enabled
        self.camera_keys = camera_keys if camera_keys is not None else ["head", "left_wrist", "right_wrist"]
        self.width = int(width)
        self.height = int(height)
        self.scale = float(scale)
        self.window_name = window_name
        self._window_names = {key: f"{self.window_name} - {key}" for key in self.camera_keys}

        self._initialized = False
        self._cv2 = None
        self._np = None
        self._last_update_t: float | None = None
        self._fps_ema: float | None = None

    def _lazy_init(self) -> None:
        if not self.enabled or self._initialized:
            return

        # Ensure the OpenCV MSMF fix is applied on Windows before importing cv2.
        import os
        import platform

        if platform.system() == "Windows":
            os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as e:
            logging.warning("Preview disabled: failed to import cv2/numpy (%s).", e)
            self.enabled = False
            return

        self._cv2 = cv2
        self._np = np
        self._initialized = True

        try:
            for win_name in self._window_names.values():
                try:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                except Exception:
                    # Some OpenCV builds/window managers don't like WINDOW_NORMAL.
                    cv2.namedWindow(win_name)
        except Exception as e:
            logging.warning("Preview disabled: failed to create OpenCV window (%s).", e)
            self.enabled = False
            self._cv2 = None
            self._np = None
            self._initialized = False

    def _ensure_uint8(self, img):
        np = self._np
        if np is None:
            raise RuntimeError("Preview not initialized")
        if img.dtype == np.uint8:
            return img
        if np.issubdtype(img.dtype, np.floating):
            return (img * 255.0).clip(0, 255).astype(np.uint8)
        return img.astype(np.uint8)

    def update(
        self,
        obs: dict,
        *,
        mode: str,
        saved_episodes: int,
        target_episodes: int,
        hint: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        try:
            self._lazy_init()
            cv2 = self._cv2
            np = self._np
            if cv2 is None or np is None:
                return

            def _get_frame(key: str, label: str):
                frame = obs.get(key)
                if frame is None:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    frame = self._ensure_uint8(frame)
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

                # LeRobot cameras default to RGB. Convert to BGR for OpenCV display.
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception:
                    pass

                cv2.putText(
                    frame,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                return frame

            now = time.perf_counter()
            if self._last_update_t is not None:
                dt = max(now - self._last_update_t, 1e-6)
                inst_fps = 1.0 / dt
                self._fps_ema = inst_fps if self._fps_ema is None else (0.9 * self._fps_ema + 0.1 * inst_fps)
            self._last_update_t = now
            overlay_parts = [f"{mode}", f"{saved_episodes}/{target_episodes}"]
            if self._fps_ema is not None:
                overlay_parts.append(f"{self._fps_ema:.1f} fps")
            overlay_parts.append("右方向键:开始/保存 左方向键:丢弃 ESC:停止")
            if hint:
                overlay_parts.append(hint)
            overlay = " | ".join(overlay_parts)

            # Show each camera in its own window.
            for key in self.camera_keys:
                frame = _get_frame(key, key)
                cv2.putText(
                    frame,
                    overlay,
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if self.scale != 1.0:
                    frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

                win_name = self._window_names.get(key, f"{self.window_name} - {key}")

                # If user closed a window, silently disable preview (recording continues).
                try:
                    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                        self.enabled = False
                        return
                except Exception:
                    pass

                cv2.imshow(win_name, frame)
            cv2.waitKey(1)
        except Exception as e:
            logging.warning("Preview disabled due to an error: %s", e)
            self.enabled = False

    def close(self) -> None:
        if not self._initialized or self._cv2 is None:
            return
        try:
            self._cv2.destroyAllWindows()
        except Exception:
            pass


LEFT_KEYMAP: dict[str, str] = {
    "shoulder_pan+": "q",
    "shoulder_pan-": "e",
    "wrist_roll+": "r",
    "wrist_roll-": "f",
    "gripper+": "t",
    "gripper-": "g",
    "x+": "w",
    "x-": "s",
    "y+": "a",
    "y-": "d",
    "pitch+": "z",
    "pitch-": "x",
    "reset": "c",
}

RIGHT_KEYMAP: dict[str, str] = {
    "shoulder_pan+": "7",
    "shoulder_pan-": "9",
    "wrist_roll+": "/",
    "wrist_roll-": "*",
    "gripper+": "+",
    "gripper-": "-",
    "x+": "8",
    "x-": "2",
    "y+": "4",
    "y-": "6",
    "pitch+": "1",
    "pitch-": "3",
    "reset": "0",
}

LEFT_JOINT_MAP: dict[str, str] = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}

RIGHT_JOINT_MAP: dict[str, str] = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


FOLLOW_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _parse_invert_joints(invert_joints: str) -> set[str]:
    raw = [p.strip() for p in (invert_joints or "").split(",") if p.strip()]
    invalid = [p for p in raw if p not in FOLLOW_JOINTS]
    if invalid:
        raise ValueError(f"Unknown joints in --invert-joints: {invalid}. Valid: {list(FOLLOW_JOINTS)}")
    return set(raw)


def _map_leader_value(joint: str, value: float, invert_joints: set[str]) -> float:
    """Map a leader (right arm) joint value into follower (left arm) space."""
    if joint not in FOLLOW_JOINTS:
        raise ValueError(joint)
    if joint not in invert_joints:
        return float(value)
    if joint == "gripper":
        return float(100.0 - value)
    return float(-value)


@dataclass(frozen=True)
class LeaderFollowerCalibration:
    """
    Per-joint, no-scaling mapping:

    left_target = left_zero + (map(right) - right_zero)

    This guarantees that each joint delta on the right is reproduced 1:1 on the left (in the chosen units).
    """

    right_zero: dict[str, float]
    left_zero: dict[str, float]
    invert_joints: set[str]

    @classmethod
    def from_observation(cls, obs: dict[str, float], *, invert_joints: set[str]) -> "LeaderFollowerCalibration":
        right_zero: dict[str, float] = {}
        left_zero: dict[str, float] = {}
        for joint in FOLLOW_JOINTS:
            r_key = f"right_arm_{joint}.pos"
            l_key = f"left_arm_{joint}.pos"
            if r_key not in obs or l_key not in obs:
                raise KeyError(f"Missing required observation keys: {r_key} / {l_key}")
            right_zero[joint] = _map_leader_value(joint, float(obs[r_key]), invert_joints)
            left_zero[joint] = float(obs[l_key])
        return cls(right_zero=right_zero, left_zero=left_zero, invert_joints=set(invert_joints))

    def compute_left_targets(self, obs: dict[str, float]) -> dict[str, float]:
        targets: dict[str, float] = {}
        for joint in FOLLOW_JOINTS:
            r_key = f"right_arm_{joint}.pos"
            if r_key not in obs:
                raise KeyError(f"Missing required observation key: {r_key}")
            r_val = _map_leader_value(joint, float(obs[r_key]), self.invert_joints)
            delta = r_val - self.right_zero[joint]
            targets[joint] = self.left_zero[joint] + delta
        return targets


def _auto_align_left_from_right(
    *,
    robot: XLerobot,
    fps: int,
    seconds: float,
    invert_joints: set[str],
    preview: OpenCVPreview | None,
) -> LeaderFollowerCalibration:
    """
    1) Snapshot right arm pose (leader).
    2) Move left arm to match that pose (mapped).
    3) Return a calibration object based on the post-align observation.
    """
    seconds = float(seconds)
    if seconds <= 0:
        obs_now = robot.get_observation()
        return LeaderFollowerCalibration.from_observation(obs_now, invert_joints=invert_joints)

    print("[ALIGN] 正在读取右臂初始位姿（请尽量保持右臂不动）...")
    obs0 = robot.get_observation()
    right_targets = {
        joint: _map_leader_value(joint, float(obs0[f"right_arm_{joint}.pos"]), invert_joints)
        for joint in FOLLOW_JOINTS
    }
    left_start = {joint: float(obs0[f"left_arm_{joint}.pos"]) for joint in FOLLOW_JOINTS}

    steps = max(1, int(round(seconds * fps)))
    print(f"[ALIGN] 自动校正左臂：读取右臂初始位姿 -> {seconds:.1f}s 内将左臂对齐（{steps} steps）")

    for i in range(steps):
        t = (i + 1) / steps
        left_cmd = {}
        for joint in FOLLOW_JOINTS:
            goal = left_start[joint] + t * (right_targets[joint] - left_start[joint])
            if joint == "gripper":
                goal = _clamp(goal, 0.0, 100.0)
            left_cmd[f"left_arm_{joint}.pos"] = float(goal)

        robot.send_action(left_cmd)

        if preview is not None:
            try:
                obs_vis = robot.get_observation()
                preview.update(
                    obs_vis,
                    mode="ALIGN",
                    saved_episodes=0,
                    target_episodes=0,
                    hint="自动校正中...",
                )
            except Exception:
                pass

        precise_sleep(1.0 / fps)

    obs1 = robot.get_observation()
    print("[ALIGN] 校正完成。开始 1:1 跟随（无缩放）。")
    return LeaderFollowerCalibration.from_observation(obs1, invert_joints=invert_joints)


@dataclass
class ArmTeleop:
    prefix: str
    joint_map: dict[str, str]
    kinematics: SO101Kinematics
    kp: float = 0.81
    degree_step: float = 3.0
    xy_step: float = 0.0081

    def __post_init__(self) -> None:
        self.pitch: float = 0.0
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.target_positions: dict[str, float] = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }

    def sync_from_observation(self, obs: dict[str, float]) -> None:
        """Initialize internal targets from current robot pose to avoid sudden jumps."""
        self.target_positions["shoulder_pan"] = float(obs[f"{self.prefix}_arm_shoulder_pan.pos"])
        self.target_positions["shoulder_lift"] = float(obs[f"{self.prefix}_arm_shoulder_lift.pos"])
        self.target_positions["elbow_flex"] = float(obs[f"{self.prefix}_arm_elbow_flex.pos"])
        self.target_positions["wrist_flex"] = float(obs[f"{self.prefix}_arm_wrist_flex.pos"])
        self.target_positions["wrist_roll"] = float(obs[f"{self.prefix}_arm_wrist_roll.pos"])
        self.target_positions["gripper"] = float(obs[f"{self.prefix}_arm_gripper.pos"])

        # Derive an approximate (x, y) for IK control from current joint angles.
        try:
            self.current_x, self.current_y = self.kinematics.forward_kinematics(
                joint2_deg=self.target_positions["shoulder_lift"],
                joint3_deg=self.target_positions["elbow_flex"],
            )
        except Exception:
            # Fallback values (will be updated on first IK move).
            self.current_x, self.current_y = 0.1629, 0.1131

        # Approximate pitch component (keeps wrist_flex coupling stable).
        self.pitch = float(
            self.target_positions["wrist_flex"]
            + self.target_positions["shoulder_lift"]
            + self.target_positions["elbow_flex"]
        )

    def move_to_zero_position(self) -> None:
        self.pitch = 0.0
        self.current_x, self.current_y = 0.1629, 0.1131
        for k in self.target_positions:
            self.target_positions[k] = 0.0

    def handle_keys(self, key_state: dict[str, bool]) -> None:
        if key_state.get("reset"):
            self.move_to_zero_position()
            return

        if key_state.get("shoulder_pan+"):
            self.target_positions["shoulder_pan"] += self.degree_step
        if key_state.get("shoulder_pan-"):
            self.target_positions["shoulder_pan"] -= self.degree_step
        if key_state.get("wrist_roll+"):
            self.target_positions["wrist_roll"] += self.degree_step
        if key_state.get("wrist_roll-"):
            self.target_positions["wrist_roll"] -= self.degree_step
        if key_state.get("gripper+"):
            self.target_positions["gripper"] = _clamp(
                self.target_positions["gripper"] + self.degree_step, 0.0, 100.0
            )
        if key_state.get("gripper-"):
            self.target_positions["gripper"] = _clamp(
                self.target_positions["gripper"] - self.degree_step, 0.0, 100.0
            )

        if key_state.get("pitch+"):
            self.pitch += self.degree_step
        if key_state.get("pitch-"):
            self.pitch -= self.degree_step

        moved = False
        if key_state.get("x+"):
            self.current_x += self.xy_step
            moved = True
        if key_state.get("x-"):
            self.current_x -= self.xy_step
            moved = True
        if key_state.get("y+"):
            self.current_y += self.xy_step
            moved = True
        if key_state.get("y-"):
            self.current_y -= self.xy_step
            moved = True

        if moved:
            try:
                joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
                self.target_positions["shoulder_lift"] = joint2
                self.target_positions["elbow_flex"] = joint3
            except Exception as e:
                logging.warning(
                    "[%s] IK failed at x=%.4f y=%.4f: %s",
                    self.prefix,
                    self.current_x,
                    self.current_y,
                    e,
                )

        # Wrist flex is coupled to (shoulder_lift, elbow_flex) and pitch.
        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + self.pitch
        )

    def p_control_action_from_observation(self, obs: dict[str, float]) -> dict[str, float]:
        current = {
            "shoulder_pan": float(obs[f"{self.prefix}_arm_shoulder_pan.pos"]),
            "shoulder_lift": float(obs[f"{self.prefix}_arm_shoulder_lift.pos"]),
            "elbow_flex": float(obs[f"{self.prefix}_arm_elbow_flex.pos"]),
            "wrist_flex": float(obs[f"{self.prefix}_arm_wrist_flex.pos"]),
            "wrist_roll": float(obs[f"{self.prefix}_arm_wrist_roll.pos"]),
            "gripper": float(obs[f"{self.prefix}_arm_gripper.pos"]),
        }
        action: dict[str, float] = {}
        for joint, target in self.target_positions.items():
            err = target - current[joint]
            cmd = current[joint] + self.kp * err
            action[f"{self.joint_map[joint]}.pos"] = float(cmd)
        return action


def _print_instructions(*, control_mode: str) -> None:
    print()
    print("=== XLeRobot 示教录制（LeRobotDataset v3）===")
    print("画面预览：")
    print("- 默认会弹出两个窗口：`XLeRobot Cameras - head`（中央）与 `XLeRobot Cameras - left_wrist`（左臂）")
    print("- 如需额外打开右臂相机：--enable-wrist-cams（会再打开 `... - right_wrist`）")
    print("- 窗口太大可用：--display-scale 0.75；不需要预览可用：--no-display")
    print("- 摄像头编号不对可用：--cam-head/--cam-left/--cam-right 重新指定（先运行：lerobot-find-cameras opencv）")
    print()
    if control_mode == "right_leader_left_follower":
        print("控制模式（推荐，稳定采集）：右臂手持（leader）→ 左臂跟随（follower）")
        print("- 程序会自动关闭右臂力矩，你可以手掰右臂关节/夹爪；左臂会实时跟随")
        print("- 数据集只记录：左臂关节 + head(中央)/left_wrist(左臂)相机（右臂仅用于示教，不会写入数据集）")
        print("- 启动时会先读取右臂初始位姿并自动校正左臂（无缩放）：右臂每个关节转多少，左臂关节也转多少")
        print()
    print("录制流程（两段）：RESET 准备 -> REC 录制")
    print("- RESET 阶段：按【右方向键】进入 REC 开始录制")
    print("- REC 阶段：按【右方向键】结束并保存；按【左方向键】丢弃并重录")
    print("- 任意阶段：按 ESC 停止本次录制（当前 demo 不保存）")
    print()
    print("连接/校准：")
    print("- 若提示从校准文件恢复：直接按 ENTER（推荐）；若第一次用或校准不准：输入 c 回车进行手动校准")
    print()
    print("提示：尽量保持头部相机角度固定，画面里同时看到盒子、篮子和手臂操作区域。")
    print()


def _make_default_root(repo_id: str) -> Path:
    # This file: .../VLA/lerobot/examples/record_xlerobot_keyboard_dataset.py
    vla_root = Path(__file__).resolve().parents[2]
    safe_name = repo_id.replace("/", "_")
    return vla_root / "datasets" / safe_name


def _build_dataset_features(
    robot: XLerobot,
    use_videos: bool,
    *,
    record_left_arm_only: bool,
    record_camera_keys: list[str],
) -> dict:
    # Actions: record only what we command (left arm only for leader-follower collection).
    if record_left_arm_only:
        action_hw = {k: v for k, v in robot.action_features.items() if k.endswith(".pos") and k.startswith("left_arm_")}
    else:
        action_hw = {
            k: v
            for k, v in robot.action_features.items()
            if k.endswith(".pos") and (k.startswith("left_arm_") or k.startswith("right_arm_"))
        }

    # Observations: record state + selected cameras.
    if record_left_arm_only:
        obs_state_hw = {
            k: v for k, v in robot.observation_features.items() if k.endswith(".pos") and k.startswith("left_arm_")
        }
    else:
        obs_state_hw = {
            k: v
            for k, v in robot.observation_features.items()
            if k.endswith(".pos")
            and (k.startswith("left_arm_") or k.startswith("right_arm_") or k.startswith("head_motor_"))
        }

    obs_cam_hw = {
        k: v
        for k, v in robot.observation_features.items()
        if isinstance(v, tuple) and (k in set(record_camera_keys))
    }
    obs_hw = {**obs_state_hw, **obs_cam_hw}

    action_features = hw_to_dataset_features(action_hw, ACTION)
    obs_features = hw_to_dataset_features(obs_hw, OBS_STR, use_video=use_videos)
    return {**action_features, **obs_features}


def _keyboard_teleop_step(
    *,
    robot: XLerobot,
    keyboard: KeyboardTeleop,
    left_arm: ArmTeleop,
    right_arm: ArmTeleop,
    fps: int,
    record: bool,
    dataset: LeRobotDataset | None,
    task: str,
    preview: OpenCVPreview | None,
    preview_mode: str,
    saved_episodes: int,
    target_episodes: int,
    preview_hint: str | None = None,
) -> bool:
    """
    Runs one control step. Returns False when caller should stop (e.g. device errors).
    """
    start_t = time.perf_counter()
    try:
        obs = robot.get_observation()
        if preview is not None:
            preview.update(
                obs,
                mode=preview_mode,
                saved_episodes=saved_episodes,
                target_episodes=target_episodes,
                hint=preview_hint,
            )
        pressed_keys = set(keyboard.get_action().keys())

        left_key_state = {action: (key in pressed_keys) for action, key in LEFT_KEYMAP.items()}
        right_key_state = {action: (key in pressed_keys) for action, key in RIGHT_KEYMAP.items()}

        left_arm.handle_keys(left_key_state)
        right_arm.handle_keys(right_key_state)

        action = {}
        action.update(left_arm.p_control_action_from_observation(obs))
        action.update(right_arm.p_control_action_from_observation(obs))

        sent_action = robot.send_action(action)

        if record:
            if dataset is None:
                raise RuntimeError("record=True requires a dataset")
            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            act_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
            dataset.add_frame({**obs_frame, **act_frame, "task": task})

    except Exception as e:
        logging.error("Teleop step failed: %s", e, exc_info=True)
        return False
    finally:
        dt = time.perf_counter() - start_t
        precise_sleep(max(1.0 / fps - dt, 0.0))

    return True


def _leader_follower_step(
    *,
    robot: XLerobot,
    fps: int,
    record: bool,
    dataset: LeRobotDataset | None,
    task: str,
    preview: OpenCVPreview | None,
    preview_mode: str,
    saved_episodes: int,
    target_episodes: int,
    calib: LeaderFollowerCalibration,
    track_kp: float,
    preview_hint: str | None = None,
) -> bool:
    """
    Leader-follower control step. Returns False when caller should stop (e.g. device errors).
    """
    start_t = time.perf_counter()
    try:
        obs = robot.get_observation()
        if preview is not None:
            preview.update(
                obs,
                mode=preview_mode,
                saved_episodes=saved_episodes,
                target_episodes=target_episodes,
                hint=preview_hint,
            )

        targets = calib.compute_left_targets(obs)
        track_kp = _clamp(float(track_kp), 0.0, 1.0)

        action: dict[str, float] = {}
        for joint, target in targets.items():
            l_key = f"left_arm_{joint}.pos"
            if l_key not in obs:
                raise KeyError(f"Missing required observation key: {l_key}")
            current = float(obs[l_key])
            cmd = target if track_kp >= 1.0 else (current + track_kp * (target - current))
            if joint == "gripper":
                cmd = _clamp(cmd, 0.0, 100.0)
            action[l_key] = float(cmd)
        sent_action = robot.send_action(action)

        if record:
            if dataset is None:
                raise RuntimeError("record=True requires a dataset")
            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            act_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
            dataset.add_frame({**obs_frame, **act_frame, "task": task})

    except Exception as e:
        logging.error("Leader-follower step failed: %s", e, exc_info=True)
        return False
    finally:
        dt = time.perf_counter() - start_t
        precise_sleep(max(1.0 / fps - dt, 0.0))

    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="local/xlerobot_box2basket_left_head")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume recording into an existing dataset root.")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--control-mode",
        type=str,
        default="right_leader_left_follower",
        choices=["right_leader_left_follower", "keyboard"],
        help="Control mode for collection. Use 'right_leader_left_follower' for stable demos.",
    )
    parser.add_argument(
        "--follow-kp",
        type=float,
        default=1.0,
        help="Leader-follower smoothing factor in [0,1]. 1.0=track right arm directly; lower=more stable.",
    )
    parser.add_argument(
        "--invert-joints",
        type=str,
        default="",
        help="Comma-separated joints to invert when mapping right->left (e.g. shoulder_pan,wrist_roll).",
    )
    parser.add_argument(
        "--use-degrees",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use degrees for arm joints (recommended for strict 1:1 leader-follower, no scaling).",
    )
    parser.add_argument(
        "--auto-align-left",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On start, read the right arm pose and automatically align the left arm to it.",
    )
    parser.add_argument(
        "--align-seconds",
        type=float,
        default=2.0,
        help="Duration for the initial left-arm auto-alignment motion (seconds).",
    )
    parser.add_argument(
        "--skip-leader-sync",
        action="store_true",
        help="(Deprecated) Same as --no-auto-align-left.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show real-time camera preview window (use --no-display to disable).",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Scale factor for the camera preview window (e.g. 0.75).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Put the box into the robot basket. 将桌子上的盒子放入机器人篮子里。",
    )

    # NOTE: For this setup, port1 controls left arm + head, port2 controls right arm + base.
    parser.add_argument("--port1", type=str, default="COM4")
    parser.add_argument("--port2", type=str, default="COM3")

    parser.add_argument("--cam-head", type=int, default=0)
    parser.add_argument("--cam-left", type=int, default=1)
    parser.add_argument("--cam-right", type=int, default=3)
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument(
        "--enable-wrist-cams",
        action="store_true",
        help="Also connect the right wrist camera (opens an extra preview window).",
    )
    parser.add_argument("--no-videos", action="store_true", help="Store PNG images (no MP4 encoding).")
    # Default to libsvtav1 for widest availability in PyAV wheels (h264 may be missing on some systems).
    parser.add_argument("--vcodec", type=str, default="libsvtav1", choices=["h264", "hevc", "libsvtav1"])

    parser.add_argument("--kp", type=float, default=0.81)
    parser.add_argument("--degree-step", type=float, default=3.0)
    parser.add_argument("--xy-step", type=float, default=0.0081)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.skip_leader_sync:
        logging.warning("--skip-leader-sync is deprecated; using --no-auto-align-left instead.")
        args.auto_align_left = False

    if args.control_mode == "right_leader_left_follower" and not args.use_degrees:
        raise ValueError(
            "Leader-follower mode requires --use-degrees to guarantee 1:1 joint deltas (no scaling). "
            "Remove --no-use-degrees."
        )

    use_videos = not args.no_videos
    root = args.root if args.root is not None else _make_default_root(args.repo_id)
    invert_joints = _parse_invert_joints(args.invert_joints)

    cameras: dict[str, OpenCVCameraConfig] = {
        "head": OpenCVCameraConfig(
            index_or_path=args.cam_head,
            fps=args.cam_fps,
            width=args.cam_width,
            height=args.cam_height,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
        "left_wrist": OpenCVCameraConfig(
            index_or_path=args.cam_left,
            fps=args.cam_fps,
            width=args.cam_width,
            height=args.cam_height,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
    }
    preview_keys = ["head", "left_wrist"]
    if args.enable_wrist_cams:
        cameras["right_wrist"] = OpenCVCameraConfig(
            index_or_path=args.cam_right,
            fps=args.cam_fps,
            width=args.cam_width,
            height=args.cam_height,
            rotation=Cv2Rotation.NO_ROTATION,
        )
        preview_keys.append("right_wrist")

    robot_cfg = XLerobotConfig(
        id="my_xlerobot",
        port1=args.port1,
        port2=args.port2,
        cameras=cameras,
        use_degrees=args.use_degrees,
    )
    robot = XLerobot(robot_cfg)

    dataset: LeRobotDataset | None = None
    keyboard: KeyboardTeleop | None = None
    listener = None
    preview = None

    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            print(
                f"[提示] 当前终端编码是 {sys.stdout.encoding}，如果中文乱码可先运行：chcp 65001 或设置环境变量 PYTHONUTF8=1。\n"
            )

        print(f"Dataset repo_id: {args.repo_id}")
        print(f"Dataset root: {root}")
        print(f"FPS: {args.fps} (camera fps={args.cam_fps})")
        print(f"Task: {args.task}")
        print(f"Ports: port1={args.port1} (左臂+头部), port2={args.port2} (右臂+底盘, 不记录底盘action)")
        cam_msg = f"Cameras(OpenCV): head={args.cam_head}, left={args.cam_left}"
        if args.enable_wrist_cams:
            cam_msg += f", right={args.cam_right}"
        print(cam_msg)
        print(f"Preview: {'ON' if args.display else 'OFF'} (scale={args.display_scale})")
        print(f"Control mode: {args.control_mode}")

        _print_instructions(control_mode=args.control_mode)

        # Dataset (create or resume)
        record_left_only = args.control_mode == "right_leader_left_follower"
        record_cameras = ["head", "left_wrist"] if record_left_only else list(cameras.keys())
        dataset_features = _build_dataset_features(
            robot,
            use_videos=use_videos,
            record_left_arm_only=record_left_only,
            record_camera_keys=record_cameras,
        )

        if args.resume:
            dataset = LeRobotDataset(
                args.repo_id,
                root=root,
                batch_encoding_size=1,
                vcodec=args.vcodec,
            )
            dataset.start_image_writer(num_processes=0, num_threads=2 * len(cameras))
        else:
            if root.exists():
                raise FileExistsError(
                    f"Dataset root already exists: {root}\n"
                    "Use --resume to append, or choose a new --root."
                )
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id,
                fps=args.fps,
                root=root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=use_videos,
                image_writer_processes=0,
                image_writer_threads=2 * len(cameras),
                batch_encoding_size=1,
                vcodec=args.vcodec,
            )

        robot.connect()
        try:
            robot.stop_base()
        except Exception:
            pass

        if args.control_mode == "keyboard":
            keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="keyboard"))
            keyboard.connect()
        else:
            # Leader-follower mode: right arm is manually moved by hand.
            robot.bus2.disable_torque(robot.right_arm_motors)
            print("[INFO] Right arm torque disabled (leader). Left arm will follow right arm.")
            if invert_joints:
                print(f"[INFO] Inverting joints for mapping: {sorted(invert_joints)}")

        listener, events = init_keyboard_listener()
        preview = OpenCVPreview(
            args.display,
            camera_keys=preview_keys,
            width=args.cam_width,
            height=args.cam_height,
            scale=args.display_scale,
        )
        if args.display:
            preview._lazy_init()
            if not preview.enabled:
                print(
                    "[提示] 相机预览窗口初始化失败（常见原因：安装的是 opencv-python-headless）。\n"
                    "      你可以用 --no-display 继续采集，或安装 opencv-python 以启用预览窗口。\n"
                )

        lf_calib: LeaderFollowerCalibration | None = None
        if args.control_mode == "right_leader_left_follower":
            if args.auto_align_left:
                lf_calib = _auto_align_left_from_right(
                    robot=robot,
                    fps=args.fps,
                    seconds=args.align_seconds,
                    invert_joints=invert_joints,
                    preview=preview,
                )
            else:
                obs_now = robot.get_observation()
                lf_calib = LeaderFollowerCalibration.from_observation(obs_now, invert_joints=invert_joints)
                print("[INFO] Auto-align disabled. Using current pose as baseline for 1:1 delta mapping.")

            if lf_calib is None:
                raise RuntimeError("Leader-follower calibration was not initialized.")

        left_arm = None
        right_arm = None
        if args.control_mode == "keyboard":
            # Teleop models (keyboard IK)
            kin_left = SO101Kinematics()
            kin_right = SO101Kinematics()
            left_arm = ArmTeleop(
                prefix="left",
                joint_map=LEFT_JOINT_MAP,
                kinematics=kin_left,
                kp=args.kp,
                degree_step=args.degree_step,
                xy_step=args.xy_step,
            )
            right_arm = ArmTeleop(
                prefix="right",
                joint_map=RIGHT_JOINT_MAP,
                kinematics=kin_right,
                kp=args.kp,
                degree_step=args.degree_step,
                xy_step=args.xy_step,
            )

            # Sync targets to current pose
            initial_obs = robot.get_observation()
            left_arm.sync_from_observation(initial_obs)
            right_arm.sync_from_observation(initial_obs)

        saved_episodes = 0

        with VideoEncodingManager(dataset):
            while saved_episodes < args.num_episodes and not events["stop_recording"]:
                # Reset mode (no recording)
                print(
                    f"\n[RESET] 准备第 {saved_episodes + 1}/{args.num_episodes} 条 demo："
                    "摆好盒子/篮子并把手臂移动到起始姿态。按【右方向键】开始录制。"
                )
                events["exit_early"] = False
                events["rerecord_episode"] = False
                while not events["exit_early"] and not events["stop_recording"]:
                    if args.control_mode == "keyboard":
                        ok = _keyboard_teleop_step(
                            robot=robot,
                            keyboard=keyboard,
                            left_arm=left_arm,
                            right_arm=right_arm,
                            fps=args.fps,
                            record=False,
                            dataset=None,
                            task=args.task,
                            preview=preview,
                            preview_mode="RESET",
                            saved_episodes=saved_episodes,
                            target_episodes=args.num_episodes,
                            preview_hint="按右方向键开始录制",
                        )
                    else:
                        ok = _leader_follower_step(
                            robot=robot,
                            fps=args.fps,
                            record=False,
                            dataset=None,
                            task=args.task,
                            preview=preview,
                            preview_mode="RESET",
                            saved_episodes=saved_episodes,
                            target_episodes=args.num_episodes,
                            calib=lf_calib,
                            track_kp=args.follow_kp,
                            preview_hint="按右方向键开始录制",
                        )
                    if not ok:
                        events["stop_recording"] = True
                        break

                if events["stop_recording"]:
                    break

                # Recording mode
                print(
                    f"[REC] 开始录制：episode={dataset.num_episodes} "
                    "(右方向键=保存/结束, 左方向键=丢弃重录, ESC=停止)"
                )
                events["exit_early"] = False
                events["rerecord_episode"] = False

                while not events["exit_early"] and not events["stop_recording"]:
                    if args.control_mode == "keyboard":
                        ok = _keyboard_teleop_step(
                            robot=robot,
                            keyboard=keyboard,
                            left_arm=left_arm,
                            right_arm=right_arm,
                            fps=args.fps,
                            record=True,
                            dataset=dataset,
                            task=args.task,
                            preview=preview,
                            preview_mode="REC",
                            saved_episodes=saved_episodes,
                            target_episodes=args.num_episodes,
                            preview_hint="按右方向键保存/结束",
                        )
                    else:
                        ok = _leader_follower_step(
                            robot=robot,
                            fps=args.fps,
                            record=True,
                            dataset=dataset,
                            task=args.task,
                            preview=preview,
                            preview_mode="REC",
                            saved_episodes=saved_episodes,
                            target_episodes=args.num_episodes,
                            calib=lf_calib,
                            track_kp=args.follow_kp,
                            preview_hint="按右方向键保存/结束",
                        )
                    if not ok:
                        events["stop_recording"] = True
                        break

                if events["stop_recording"]:
                    print("[停止] 已停止录制：丢弃当前未保存的 demo 缓冲区。")
                    dataset.clear_episode_buffer(delete_images=True)
                    break

                if events["rerecord_episode"]:
                    print("[重录] 已丢弃本条 demo，准备重录...")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer(delete_images=True)
                    continue

                dataset.save_episode()
                saved_episodes += 1
                print(f"[已保存] 已保存 {saved_episodes}/{args.num_episodes}")

        print(f"\n完成。数据集已保存到：{dataset.root}")
        print("下一步（训练 pi05 LoRA）：")
        model_root = Path(__file__).resolve().parents[2] / "model"
        model_root.mkdir(parents=True, exist_ok=True)
        output_dir = model_root / "pi05_xlerobot_lora"
        print(
            "  python .\\src\\lerobot\\scripts\\lerobot_train.py "
            f"--dataset.repo_id={args.repo_id} --dataset.root=\"{dataset.root}\" "
            "--policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base "
            "--policy.push_to_hub=false --peft.method_type=LORA --peft.r=16 "
            f"--output_dir=\"{output_dir}\""
        )
        return 0
    finally:
        try:
            if preview is not None:
                preview.close()
        except Exception:
            pass

        try:
            if listener is not None:
                listener.stop()
        except Exception:
            pass

        try:
            if keyboard is not None and keyboard.is_connected:
                keyboard.disconnect()
        except Exception:
            pass

        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass

        try:
            if dataset is not None:
                dataset.finalize()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
