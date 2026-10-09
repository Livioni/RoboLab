"""Collect synchronized 320x180 RGB-D and robot truth using local/remote OpenPI.

Run from the RoboLab environment; OpenPI itself runs in its own environment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

import cv2  # noqa: F401 -- must precede Isaac Lab
import numpy as np

from policies.pi0_family.droid_dataset import (
    ARM_JOINT_NAMES, EpisodeWriter, pose_matrix, tcp_transforms,
)


class TimedPolicyConnection:
    """OpenPI wire protocol with bounded connect/receive and explicit cleanup."""

    def __init__(self, host, port, timeout):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        self.codec = msgpack_numpy
        self.timeout = timeout
        self.connection = connect(f"ws://{host}:{port}", compression=None, max_size=None, open_timeout=15)
        self.socket = self.connection.__enter__()
        try:
            self.metadata = self.codec.unpackb(self.socket.recv(timeout=timeout))
        except BaseException:
            self.connection.__exit__(None, None, None)
            raise

    def infer(self, observation):
        self.socket.send(self.codec.Packer().pack(observation))
        result = self.socket.recv(timeout=self.timeout)
        if isinstance(result, str):
            raise RuntimeError(f"OpenPI inference failed: {result}")
        return self.codec.unpackb(result)

    def close(self):
        self.connection.__exit__(None, None, None)


class SimSnapshot:
    def __init__(self, env):
        self.robot = env.scene["robot"]
        self.camera = env.scene.sensors["over_shoulder_left_camera"]
        self.joints = [self.robot.data.joint_names.index(name) for name in ARM_JOINT_NAMES]
        self.finger = self.robot.data.joint_names.index("finger_joint")
        self.base = self.robot.data.body_names.index("panda_link0")
        self.gripper = self.robot.data.body_names.index("base_link")

    @staticmethod
    def array(value):
        return value.detach().cpu().numpy().copy()

    def state(self):
        robot, camera = self.robot.data, self.camera.data
        world_base = pose_matrix(self.array(robot.body_pos_w[0, self.base]),
                                 self.array(robot.body_quat_w[0, self.base]))
        world_camera = pose_matrix(self.array(camera.pos_w[0]), self.array(camera.quat_w_ros[0]))
        world_gripper = pose_matrix(self.array(robot.body_pos_w[0, self.gripper]),
                                    self.array(robot.body_quat_w[0, self.gripper]))
        # Apply the full USD-to-DROID tool transform to the measured body pose.
        # This is also used for terminal_state, so all exported TCP poses agree.
        base_camera, base_tcp = tcp_transforms(world_base, world_camera, world_gripper)
        return {
            "joint_position": self.array(robot.joint_pos[0, self.joints]),
            "gripper_closedness": np.clip(float(robot.joint_pos[0, self.finger]) / (np.pi / 4), 0, 1),
            "finger_joint_position": float(robot.joint_pos[0, self.finger]),
            "base_tcp": base_tcp, "base_camera": base_camera,
            "world_base": world_base, "world_gripper": world_gripper,
            "intrinsic": self.array(camera.intrinsic_matrices[0]),
        }

    def frame(self):
        state = self.state()
        data = self.camera.data.output
        if "rgb" not in data or "depth" not in data:
            raise RuntimeError("RGB-D annotators are not ready")
        state["rgb"] = self.array(data["rgb"][0, ..., :3])
        state["depth_m"] = self.array(data["depth"][0]).reshape(state["rgb"].shape[:2])
        if not np.any(np.isfinite(state["depth_m"]) & (state["depth_m"] > 0)):
            raise RuntimeError("Depth has no valid pixels; refusing an initialization placeholder")
        return state


def collect(args, app):
    import torch

    import robolab.constants as constants
    from policies.pi0_family.client import Pi0DroidJointposClient
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env, end_episode
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.registrations.droid.camera_presets import WRIST_LEFT
    from robolab.robots.droid import WristCameraCfg
    from robolab.variations.camera import with_depth

    class CollectionClient(Pi0DroidJointposClient):
        def _connect(self):
            previous = getattr(self, "client", None)
            if previous is not None:
                previous.close()
            return TimedPolicyConnection(self._remote_host, self._remote_port, args.inference_timeout)

        def _unpack_response(self, response):
            actions = super()._unpack_response(response)
            if actions.ndim != 2 or actions.shape[1] != 8 or actions.shape[0] < self.open_loop_horizon:
                raise ValueError(f"Expected jointpos actions [H>=15,8], received {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("Policy returned non-finite actions")
            return actions

        def _build_visualization(self, extracted_obs):
            return None

        def close(self):
            self.client.close()

    constants.RECORD_IMAGE_DATA = False
    cameras = [camera if camera is WristCameraCfg else with_depth(camera) for camera in WRIST_LEFT]
    auto_register_droid_envs(task=[args.task], cameras=cameras, lazy_sensor_update=False)
    names = get_envs(task=[args.task])
    if len(names) != 1:
        raise ValueError(f"Expected exactly one registered task environment, got {names}")

    client, env = None, None
    try:
        client = CollectionClient(remote_host=args.remote_host, remote_port=args.remote_port,
                                  policy_variant="pi05", open_loop_horizon=15)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        for episode in range(args.num_episodes):
            path = Path(args.output_dir).resolve() / f"{args.task}__{run_id}_{episode:04d}"
            writer = EpisodeWriter(path, width=args.width, height=args.height, fps=15.0, metadata={
                "task": args.task, "seed": args.seed + episode, "policy": "pi05_droid_jointpos",
                "policy_endpoint": f"ws://{args.remote_host}:{args.remote_port}",
                "software": {key: version(key) for key in ("isaacsim", "isaaclab", "torch", "openpi-client")},
            })
            status, complete, reason, terminal = "interrupted", False, None, None
            try:
                constants.set_output_dir(str(path))
                if env is None:
                    cfg = parse_env_cfg(names[0], device=args.device or "cuda:0", seed=args.seed, num_envs=1)
                    cfg.scene.egocentric_mirrored_camera = None
                    cfg.observations.viewport_cam = None
                    for sensor_name in ("over_shoulder_left_camera", "wrist_cam"):
                        camera = getattr(cfg.scene, sensor_name)
                        camera.width, camera.height = args.width, args.height
                        camera.update_period = cfg.sim.dt * cfg.decimation
                    third_person_camera = cfg.scene.over_shoulder_left_camera
                    # Approximate local DROID calibrations: fx=fy~131.55 at 320x180.
                    third_person_camera.spawn.focal_length = 2.21
                    third_person_camera.spawn.horizontal_aperture = 5.376
                    third_person_camera.spawn.vertical_aperture = 3.024
                    third_person_camera.update_latest_camera_pose = True
                    cfg.sim.render.antialiasing_mode = "DLAA"
                    cfg.sim.render.enable_dl_denoiser = True
                    cfg.sim.render.samples_per_pixel = 4
                    env, cfg = create_env(cfg, policy="pi05", renderer="realtime", rendering_mode="performance")
                    if not np.isclose(env.step_dt, 1 / 15):
                        raise ValueError("Collector expects 15 Hz control")
                    # Keep scalar/scene HDF5 recording bounded; do not duplicate image tensors.
                    env.recorder_manager.set_flush_interval(100)
                else:
                    env.reset_eval_state()
                    cfg.seed = args.seed + episode
                    cfg.recorders.dataset_export_dir_path = str(path)
                    env.recorder_manager.cfg.dataset_export_dir_path = str(path)
                env.recorder_manager.set_hdf5_file(str(path / "trajectory.hdf5"))
                obs, _ = env.reset(seed=args.seed + episode)
                env.recorder_manager.set_episode_index(0, env_ids=[0])
                # Refresh annotators after reset without advancing physics or adding frames.
                for _ in range(3):
                    env.sim.render()
                for camera in env.scene.sensors.values():
                    if hasattr(camera, "cfg") and hasattr(camera.cfg, "data_types"):
                        camera.reset()
                        camera.update(0.0, force_recompute=True)
                obs = env.observation_manager.compute()
                snapshotter = SimSnapshot(env)
                if list(env.action_manager.get_term("body")._joint_names) != ARM_JOINT_NAMES:
                    raise ValueError("Robot action joint order does not match the DROID jointpos protocol")
                writer.metadata["language_instruction"] = [cfg.instruction]
                writer.metadata["simulation_joint_names"] = list(snapshotter.robot.data.joint_names)
                writer.metadata["physics_dt"] = cfg.sim.dt
                writer.metadata["decimation"] = cfg.decimation
                (path / "env_cfg.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str) + "\n")
                client.reset()
                client.begin_episode(episode)
                limit = min(env.max_episode_length, args.max_steps or env.max_episode_length)
                print(f"[collect] {path} | {args.width}x{args.height} | max {limit} steps", flush=True)
                # Simulation buffers survive across episodes; inference_mode
                # would create tensors that reset cannot update in-place later.
                with torch.no_grad():
                    for step in range(limit):
                        if not app.is_running():
                            raise InterruptedError("Simulation app closed")
                        before = snapshotter.frame()
                        started = time.monotonic()
                        action = np.asarray(client.infer(obs, cfg.instruction)["action"], dtype=np.float32)
                        latency = time.monotonic() - started
                        previous_step = int(env.episode_length_buf[0])
                        obs, _, _, _, _ = env.step(torch.from_numpy(action.copy()).to(env.device)[None])
                        if int(env.episode_length_buf[0]) != previous_step + 1:
                            raise RuntimeError("Automatic reset during step; episode interrupted to avoid mixing trajectories")
                        writer.append(before, action, inference_seconds=latency)
                        terminal = snapshotter.state()
                        if step % 15 == 0:
                            print(f"[collect] frame={step + 1} simulation_time={(step + 1) / 15:.3f}s", flush=True)
                        if env.all_terminated:
                            success = env.get_env_results()[0]["success"]
                            status, complete = ("success" if success else "failure"), True
                            break
                    else:
                        status, reason = "truncated", "Reached --max-steps before task termination"
            except BaseException as error:
                reason = f"{type(error).__name__}: {error}"
                raise
            finally:
                writer.finish(status=status, complete=complete, terminal_state=terminal, reason=reason)
                if env is not None:
                    end_episode(env)
                print(f"[collect] saved {len(writer.rows)} frames; status={status}; {path}", flush=True)
    finally:
        if env is not None:
            env.close()
        if client is not None:
            client.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--remote-port", type=int, default=8000)
    parser.add_argument("--task", default="BananaInBowlTask")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="output/droid_like")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional smoke-test limit; output is marked truncated")
    parser.add_argument("--inference-timeout", type=float, default=300, help="Seconds, including first-inference JIT compilation")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    for name in ("num_episodes", "width", "height", "inference_timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    args.enable_cameras = True
    launcher = AppLauncher(args)
    exit_code = 0
    try:
        collect(args, launcher.app)
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        # Report BEFORE Kit shutdown, which disables its Python logging hooks.
        traceback.print_exc()
        exit_code = 1
    finally:
        launcher.app.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
