"""Small, simulator-independent writer for synchronized DROID-like episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

CAMERA_ID = "third_person"
TCP_DEFINITION_ID = "droid_generate_tcp_v2"
# generate_tcp.closed_tcp_offset(): fixed pad midpoint at finger_joint=0.8.
# Its reference URDF mounts robotiq_arg2f_base_link identically to panda_link8.
DROID_TCP_OFFSET = np.array([-3.086367901917167e-15, 0.0, 0.1442549775197502])
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


def pose_matrix(position, quaternion_wxyz):
    """Isaac Lab wxyz pose -> homogeneous local-to-world transform."""
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = position
    return result


# From assets/robots/franka_robotiq_2f_85_flattened.usd, joint
# /panda/panda_link8/panda_hand_joint, at full authored float precision.
# W_link8 @ local0 = W_usd_gripper @ local1, hence
# T_usd_gripper_droid_gripper = local1 @ inverse(local0).
_MOUNT_LOCAL0 = pose_matrix([0, 0, 0], [0.9238795042037964, 0, 0, -0.3826834559440613])
_MOUNT_LOCAL1 = pose_matrix(
    [-0.018174022436141968, -7.564315146479927e-11, -2.115828579007939e-08],
    [0.2705981731414795, 0.6532813906669617, 0.27059805393218994, 0.6532815098762512],
)
USD_GRIPPER_TO_TCP = _MOUNT_LOCAL1 @ np.linalg.inv(_MOUNT_LOCAL0)
USD_GRIPPER_TO_TCP[:3, 3] += USD_GRIPPER_TO_TCP[:3, :3] @ DROID_TCP_OFFSET
TCP_OFFSET = USD_GRIPPER_TO_TCP[:3, 3].copy()


def tcp_metadata():
    return {
        "tcp_definition_id": TCP_DEFINITION_ID,
        "tcp_definition": "generate_tcp.py: fixed midpoint of inner fingertip contact faces at "
                          "finger_joint=0.8 rad; orientation equals DROID URDF panda_link8",
        "tcp_reference_frame": "DROID URDF robotiq_arg2f_base_link (identical to panda_link8)",
        "tcp_offset_in_robotiq_base_m": DROID_TCP_OFFSET.tolist(),
        "tcp_offset_in_usd_base_link_m": TCP_OFFSET.tolist(),
        "tcp_transform_in_usd_base_link": USD_GRIPPER_TO_TCP.tolist(),
        "tcp_transform_convention": "T_usd_base_link_tcp; maps TCP coordinates into USD base_link",
        "tcp_mount_source": "assets/robots/franka_robotiq_2f_85_flattened.usd:"
                            "/panda/panda_link8/panda_hand_joint",
    }


def tcp_transforms(world_base, world_camera, world_gripper):
    base_camera = np.linalg.solve(world_base, world_camera)
    base_gripper = np.linalg.solve(world_base, world_gripper)
    return base_camera, base_gripper @ USD_GRIPPER_TO_TCP


def xyz_rpy(transform):
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")]


def encode_depth(depth_m):
    """Z-depth in meters -> nearest millimeter; 0 denotes invalid/unrepresentable."""
    depth = np.asarray(depth_m, dtype=np.float64)
    result = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0) & (depth <= 65.535)
    result[valid] = np.rint(depth[valid] * 1000).astype(np.uint16)
    return result


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class EpisodeWriter:
    """Write images immediately; keep only small state/action arrays on the CPU.

    Call append ONLY after the corresponding env.step succeeds. An episode is
    incomplete until finish marks it complete; existing directories are refused.
    """

    def __init__(self, path, *, width, height, fps, metadata):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.width, self.height, self.fps = width, height, fps
        for folder in (f"images/{CAMERA_ID}", f"depths/{CAMERA_ID}", "intrinsic",
                       "extrinsic", "observations", "action", f"TCP/{CAMERA_ID}"):
            (self.path / folder).mkdir(parents=True, exist_ok=True)
        self.metadata = {
            **metadata, "format": "robolab_droid_like_v2", "camera_id": CAMERA_ID,
            "width": width, "height": height, "fps": fps, "frame_count": 0,
            "status": "incomplete", "complete": False,
            "frame_alignment": "RGB/depth/state at t, action applied over [t,t+1)",
            "timestamp_unit": "seconds", "timestamp_clock": "simulation, episode-relative",
            "arm_joint_names": ARM_JOINT_NAMES, "joint_unit": "radian",
            "base_frame": "panda_link0", "extrinsics": "base_to_camera",
            "camera_frame": "OpenCV (+x right, +y down, +z forward)",
            "cartesian_position": "fixed TCP xyz/rpy in robot base; meters/radians",
            "rpy_convention": "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))",
            "gripper_position": "closedness: 0=open, 1=closed; measured finger_joint/(pi/4), clipped",
            "action_semantics": "absolute arm joint targets and binary gripper closedness sent to env.step",
            **tcp_metadata(),
        }
        self.rows = []
        self.intrinsic = self.base_camera = None
        write_json(self.path / "metadata.json", self.metadata)

    def append(self, snapshot, action, *, inference_seconds=0.0):
        rgb, depth = snapshot["rgb"], snapshot["depth_m"]
        action = np.asarray(action, dtype=np.float32)
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError("RGB must be uint8 at the configured resolution")
        if depth.shape != (self.height, self.width):
            raise ValueError("Depth and RGB resolution must match")
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Expected eight finite executed action values")
        if action[-1] not in (0, 1):
            raise ValueError("Executed gripper command must be binary closedness")
        k, base_camera = snapshot["intrinsic"], snapshot["base_camera"]
        if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError("Camera intrinsics are not initialized")
        for transform in (base_camera, snapshot["base_tcp"]):
            if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                    or not np.allclose(transform[3], [0, 0, 0, 1])
                    or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)):
                raise ValueError("Invalid rigid transform")
        joints = np.asarray(snapshot["joint_position"], dtype=np.float32)
        grip = float(snapshot["gripper_closedness"])
        if joints.shape != (7,) or not np.isfinite(joints).all() or not 0 <= grip <= 1:
            raise ValueError("Invalid measured joint/gripper state")
        if self.intrinsic is None:
            self.intrinsic, self.base_camera = k.copy(), base_camera.copy()
            np.save(self.path / "intrinsic" / f"{CAMERA_ID}.npy", k)
            np.save(self.path / "extrinsic" / f"{CAMERA_ID}.npy", np.linalg.inv(base_camera))
            self.metadata["camera_to_base"] = base_camera.tolist()
        elif not (np.allclose(k, self.intrinsic, atol=1e-5, rtol=0)
                  and np.allclose(base_camera, self.base_camera, atol=1e-5, rtol=0)):
            raise ValueError("This format requires a static camera relative to the robot base")
        index = len(self.rows)
        camera_tcp = np.linalg.solve(base_camera, snapshot["base_tcp"])
        row = {"joint": joints.copy(), "gripper": grip,
               "cartesian": xyz_rpy(snapshot["base_tcp"]),
               "tcp": np.r_[xyz_rpy(camera_tcp), 1 - grip],
               "action": action.copy(), "inference_seconds": float(inference_seconds)}
        # Do not leave an unmatched image if writing its companion fails.
        rgb_path = self.path / "images" / CAMERA_ID / f"{index:06d}.png"
        depth_path = self.path / "depths" / CAMERA_ID / f"{index:06d}.png"
        try:
            Image.fromarray(rgb).save(rgb_path)
            Image.fromarray(encode_depth(depth)).save(depth_path)
        except BaseException:
            rgb_path.unlink(missing_ok=True)
            depth_path.unlink(missing_ok=True)
            raise
        self.rows.append(row)

    def finish(self, *, status, complete, terminal_state=None, reason=None):
        n = len(self.rows)

        def values(key, columns):
            return np.asarray([row[key] for row in self.rows], dtype=np.float32).reshape(n, columns)

        np.save(self.path / "observations/joint_position.npy", values("joint", 7))
        np.save(self.path / "observations/gripper_position.npy", values("gripper", 1))
        np.save(self.path / "observations/cartesian_position.npy", values("cartesian", 6))
        actions = values("action", 8)
        np.save(self.path / "action/joint_position.npy", actions[:, :7])
        np.save(self.path / "action/gripper_position.npy", actions[:, 7:])
        np.save(self.path / "TCP" / CAMERA_ID / "state.npy", values("tcp", 7))
        np.save(self.path / "timestamps.npy", np.arange(n, dtype=np.float64) / self.fps)
        np.save(self.path / "inference_seconds.npy", values("inference_seconds", 1)[:, 0])
        if terminal_state is not None:
            np.savez(self.path / "terminal_state.npz", **terminal_state, timestamp=n / self.fps)
        write_json(self.path / "depths/metadata.json", {
            "units": "millimeters", "dtype": "uint16", "depth_type": "distance_to_image_plane",
            "invalid_value": 0, "frame_alignment": "index",
            "cameras": {CAMERA_ID: {"frames": n, "width": self.width, "height": self.height}},
        })
        write_json(self.path / "TCP" / CAMERA_ID / "metadata.json", {
            "shape": [n, 7], "columns": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_open"],
            "position_unit": "meter", "rotation_unit": "radian",
            "coordinate_frame": self.metadata["camera_frame"],
            "rpy_convention": self.metadata["rpy_convention"],
            **tcp_metadata(),
        })
        self.metadata.update(frame_count=n, status=status, complete=complete, reason=reason)
        write_json(self.path / "metadata.json", self.metadata)
