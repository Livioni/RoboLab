"""Validate a collected episode without Isaac Sim; optionally plot TCP and depth."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from policies.pi0_family.droid_dataset import (
    ARM_JOINT_NAMES, CAMERA_ID, TCP_DEFINITION_ID, TCP_OFFSET,
    USD_GRIPPER_TO_TCP, tcp_transforms,
)


def validate(path):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    for metadata_path in (path / "metadata.json", path / "TCP" / CAMERA_ID / "metadata.json"):
        tcp_meta = json.loads(metadata_path.read_text())
        if tcp_meta.get("tcp_definition_id") != TCP_DEFINITION_ID:
            raise ValueError("Legacy or unknown TCP definition; recollect or explicitly migrate labels "
                             "before validating against generate_tcp.py")
        np.testing.assert_allclose(tcp_meta["tcp_transform_in_usd_base_link"],
                                   USD_GRIPPER_TO_TCP, atol=1e-12, rtol=0)
    n, h, w = meta["frame_count"], meta["height"], meta["width"]
    if n < 1:
        raise ValueError("No committed frames")
    arrays = {}
    for name, shape in {
        "timestamps": (n,), "inference_seconds": (n,),
        "observations/joint_position": (n, 7), "observations/gripper_position": (n, 1),
        "observations/cartesian_position": (n, 6), "action/joint_position": (n, 7),
        "action/gripper_position": (n, 1), f"TCP/{CAMERA_ID}/state": (n, 7),
    }.items():
        array = np.load(path / f"{name}.npy", allow_pickle=False)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid array: {name}")
        arrays[name] = array
    np.testing.assert_allclose(arrays["timestamps"], np.arange(n) / meta["fps"], atol=1e-9)
    k = np.load(path / "intrinsic" / f"{CAMERA_ID}.npy")
    ext = np.load(path / "extrinsic" / f"{CAMERA_ID}.npy")
    if k.shape != (3, 3) or min(k[0, 0], k[1, 1]) <= 0 or not np.isfinite(k).all():
        raise ValueError("Invalid intrinsics")
    np.testing.assert_allclose(ext @ np.asarray(meta["camera_to_base"]), np.eye(4), atol=1e-6)
    np.testing.assert_allclose(ext[:3, :3].T @ ext[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(ext[:3, :3]), 1, atol=1e-6)
    tcp = arrays[f"TCP/{CAMERA_ID}/state"]
    base = arrays["observations/cartesian_position"]
    predicted_xyz = base[:, :3] @ ext[:3, :3].T + ext[:3, 3]
    np.testing.assert_allclose(predicted_xyz, tcp[:, :3], atol=1e-6)
    np.testing.assert_allclose(ext[:3, :3] @ Rotation.from_euler("xyz", base[:, 3:]).as_matrix(),
                               Rotation.from_euler("xyz", tcp[:, 3:6]).as_matrix(), atol=1e-6)
    np.testing.assert_allclose(tcp[:, 6], 1 - arrays["observations/gripper_position"][:, 0], atol=1e-6)
    if not np.isin(arrays["action/gripper_position"], [0, 1]).all():
        raise ValueError("Non-binary executed gripper command")
    valid_depth = []
    for folder in ("images", "depths"):
        files = sorted((path / folder / CAMERA_ID).glob("*.png"))
        if [f.name for f in files] != [f"{i:06d}.png" for i in range(n)]:
            raise ValueError(f"Missing or extra {folder} frames")
        for file in files:
            a = np.asarray(Image.open(file))
            expected = (h, w, 3) if folder == "images" else (h, w)
            if a.shape != expected or a.dtype != (np.uint8 if folder == "images" else np.uint16):
                raise ValueError(f"Invalid PNG: {file}")
            if folder == "depths":
                valid_depth.append(float((a > 0).mean()))
    # Independent simulator recorder: actions have the same index; its states
    # are POST-step, so compare them with our next frame, never the same frame.
    with h5py.File(path / "trajectory.hdf5", "r") as file:
        demo = file["data/demo_0"]
        expected_actions = np.c_[arrays["action/joint_position"], arrays["action/gripper_position"]]
        np.testing.assert_allclose(demo["actions"][:], expected_actions, atol=1e-6)
        sim_names = meta.get("simulation_joint_names", ARM_JOINT_NAMES)
        indices = [sim_names.index(name) for name in ARM_JOINT_NAMES]
        post = demo["states/articulation/robot/joint_position"][:][:, indices]
        np.testing.assert_allclose(post[:-1], arrays["observations/joint_position"][1:], atol=1e-6)
        np.testing.assert_allclose(demo["initial_state/articulation/robot/joint_position"][0, indices],
                                   arrays["observations/joint_position"][0], atol=1e-6)
        # Cross-check TCP against recorded gripper body pose, including its offset.
        pos, quat = demo["ee_pose/position"][:], demo["ee_pose/orientation"][:]
        if n > 1:
            gripper_rotation = Rotation.from_quat(quat[:-1, [1, 2, 3, 0]])
            derived = pos[:-1] + gripper_rotation.apply(TCP_OFFSET)
            np.testing.assert_allclose(derived, base[1:, :3], atol=2e-6)
            np.testing.assert_allclose(gripper_rotation.as_matrix() @ USD_GRIPPER_TO_TCP[:3, :3],
                                       Rotation.from_euler("xyz", base[1:, 3:]).as_matrix(), atol=2e-6)
        terminal = np.load(path / "terminal_state.npz")
        np.testing.assert_allclose(post[-1], terminal["joint_position"], atol=1e-6)
        last_rotation = Rotation.from_quat(quat[-1, [1, 2, 3, 0]])
        np.testing.assert_allclose(pos[-1] + last_rotation.apply(TCP_OFFSET),
                                   terminal["base_tcp"][:3, 3], atol=2e-6)
        np.testing.assert_allclose(last_rotation.as_matrix() @ USD_GRIPPER_TO_TCP[:3, :3],
                                   terminal["base_tcp"][:3, :3], atol=2e-6)
        _, expected_terminal = tcp_transforms(terminal["world_base"], terminal["world_base"],
                                              terminal["world_gripper"])
        np.testing.assert_allclose(terminal["base_tcp"], expected_terminal, atol=2e-6)
    return {"frame_count": n, "resolution": [w, h], "status": meta["status"],
            "complete": meta["complete"], "duration_seconds": n / meta["fps"],
            "mean_valid_depth_fraction": float(np.mean(valid_depth)),
            "hdf5_action_and_state_alignment": "passed", "tcp_geometry": "passed"}


def preview(path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    tcp = np.load(path / "TCP" / CAMERA_ID / "state.npy")
    k = np.load(path / "intrinsic" / f"{CAMERA_ID}.npy")
    fig, axes = plt.subplots(2, 4, figsize=(14, 5), layout="constrained")
    for column, index in enumerate(np.linspace(0, len(tcp) - 1, 4, dtype=int)):
        rgb = np.asarray(Image.open(path / "images" / CAMERA_ID / f"{index:06d}.png"))
        depth = np.asarray(Image.open(path / "depths" / CAMERA_ID / f"{index:06d}.png")) / 1000
        ax = axes[0, column]
        ax.imshow(rgb)
        pixel = k @ tcp[index, :3]
        if pixel[2] > 0:
            ax.scatter(*(pixel[:2] / pixel[2]), c="lime", marker="+", s=90)
        ax.set_title(f"Frame {index}: TCP (+)")
        axes[1, column].imshow(np.ma.masked_equal(depth, 0), vmin=0, vmax=1.5, cmap="viridis")
        axes[1, column].set_title("Z-depth (m); white = invalid")
        for ax in axes[:, column]:
            ax.set_xlim(0, rgb.shape[1])
            ax.set_ylim(rgb.shape[0], 0)
    fig.savefig(output, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.episode), indent=2))
    if args.preview:
        preview(args.episode, args.preview)
