"""CPU-only tests: python -m unittest discover -s tests -p test_droid_dataset.py."""

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from policies.pi0_family.droid_dataset import (
    EpisodeWriter, encode_depth, pose_matrix, tcp_transforms, TCP_DEFINITION_ID, USD_GRIPPER_TO_TCP,
)
from policies.pi0_family.validate_dataset import validate


class DroidDatasetTests(unittest.TestCase):
    def test_tcp_matches_droid_reference_frame(self):
        _, base_tcp = tcp_transforms(np.eye(4), np.eye(4), np.eye(4))
        np.testing.assert_allclose(base_tcp[:3, 3], [.126080955, 0, 0], atol=3e-8)
        np.testing.assert_allclose(base_tcp[:3, :3], [[0, 0, 1], [0, -1, 0], [1, 0, 0]], atol=3e-7)

    def test_recorded_usd_pose_against_generate_tcp_urdf_fk(self):
        # Independent fixture: simulation post-step 79 of seed-1 BananaInBowl.
        # Expected pose was computed from that frame's measured joints using
        # generate_tcp.py and franka-panda-robotiq-2f85/panda_robotiq_2f85.urdf.
        # This detects both the old 18 mm translation error and missing rotation.
        gripper = pose_matrix(
            [.4998984634876251, -.08280409872531891, .1557585448026657],
            [.5248026847839355, .4303947687149048, .4627738893032074, -.5702481269836426],
        )
        _, tcp = tcp_transforms(np.eye(4), np.eye(4), gripper)
        np.testing.assert_allclose(tcp[:3, 3],
                                   [.48997780199602703, -.1080439537573115, .03262865058425826],
                                   atol=2e-6, rtol=0)
        expected_rotation = [
            [-.00513372889243377, -.996886271518474, -.07868549094753273],
            [-.9795363364777477, .02084523685817495, -.20018501847061165],
            [.20120191437312795, .07604760192354415, -.9765933400828956],
        ]
        np.testing.assert_allclose(tcp[:3, :3], expected_rotation, atol=2e-6, rtol=0)

    def test_nonidentity_base_and_camera_projection(self):
        def transform(xyz, angles):
            xyzw = Rotation.from_euler("xyz", angles).as_quat()
            return pose_matrix(xyz, xyzw[[3, 0, 1, 2]])

        world_base = transform([3, -2, 1], [0.2, -0.3, 0.7])
        base_camera = transform([0.2, 0.1, 0.3], [0.1, 0.2, -0.4])
        camera_tcp = transform([0.1, -0.1, 1], [0.3, 0.1, 0.2])
        world_gripper = world_base @ base_camera @ camera_tcp @ np.linalg.inv(USD_GRIPPER_TO_TCP)
        actual_camera, actual_tcp = tcp_transforms(world_base, world_base @ base_camera, world_gripper)
        np.testing.assert_allclose(actual_camera, base_camera, atol=1e-12)
        actual_camera_tcp = np.linalg.solve(actual_camera, actual_tcp)
        np.testing.assert_allclose(actual_camera_tcp, camera_tcp, atol=1e-12)
        k = np.array([[100, 0, 160], [0, 100, 90], [0, 0, 1]])
        pixel = k @ actual_camera_tcp[:3, 3]
        np.testing.assert_allclose(pixel[:2] / pixel[2], [170, 80])
        reconstructed_base = actual_camera @ np.r_[np.linalg.solve(k, [170, 80, 1]), 1]
        np.testing.assert_allclose(reconstructed_base[:3], actual_tcp[:3, 3], atol=1e-12)

    def test_depth_rounding_invalid_and_png(self):
        source = np.array([[1.0004, 1.0006, np.inf, np.nan, -1, 65.536]])
        encoded = encode_depth(source)
        np.testing.assert_array_equal(encoded, [[1000, 1001, 0, 0, 0, 0]])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "depth.png"
            Image.fromarray(encoded).save(path)
            np.testing.assert_array_equal(np.asarray(Image.open(path)), encoded)

    @staticmethod
    def snapshot():
        tcp = np.eye(4)
        tcp[2, 3] = 1
        return {"rgb": np.full((2, 3, 3), [255, 30, 10], dtype=np.uint8),
                "depth_m": np.ones((2, 3)), "intrinsic": np.array([[2., 0, 1], [0, 2, 1], [0, 0, 1]]),
                "base_camera": np.eye(4), "base_tcp": tcp, "joint_position": np.arange(7),
                "gripper_closedness": 0.25}

    def test_alignment_gripper_metadata_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "episode"
            writer = EpisodeWriter(path, width=3, height=2, fps=15, metadata={})
            with self.assertRaises(FileExistsError):
                EpisodeWriter(path, width=3, height=2, fps=15, metadata={})
            for t in range(3):
                snapshot = self.snapshot()
                snapshot["joint_position"] = np.arange(7) + t
                writer.append(snapshot, np.r_[np.arange(7) + t + 0.5, 1])
            writer.finish(status="failure", complete=True, terminal_state={"joint_position": np.arange(7) + 3})
            np.testing.assert_allclose(np.load(path / "timestamps.npy"), np.arange(3) / 15)
            measured = np.load(path / "observations/joint_position.npy")
            executed = np.load(path / "action/joint_position.npy")
            np.testing.assert_allclose(executed - measured, 0.5)
            np.testing.assert_allclose(np.load(path / "TCP/third_person/state.npy")[:, 6], 0.75)
            np.testing.assert_array_equal(np.asarray(Image.open(path / "images/third_person/000000.png")), self.snapshot()["rgb"])
            meta = json.loads((path / "metadata.json").read_text())
            self.assertEqual(meta["frame_count"], 3)
            self.assertTrue(meta["complete"])
            self.assertEqual(meta["tcp_definition_id"], TCP_DEFINITION_ID)
            tcp_meta = json.loads((path / "TCP/third_person/metadata.json").read_text())
            self.assertEqual(tcp_meta["tcp_definition_id"], TCP_DEFINITION_ID)
            np.testing.assert_allclose(tcp_meta["tcp_transform_in_usd_base_link"], USD_GRIPPER_TO_TCP)
            self.assertEqual(len(list((path / "depths/third_person").glob("*.png"))), 3)
            self.assertAlmostEqual(float(np.load(path / "terminal_state.npz")["timestamp"]), 3 / 15)

    def test_incomplete_and_camera_motion_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EpisodeWriter(Path(root) / "episode", width=3, height=2, fps=15, metadata={})
            writer.append(self.snapshot(), np.zeros(8))
            moved = self.snapshot()
            moved["base_camera"][0, 3] = 0.1
            with self.assertRaises(ValueError):
                writer.append(moved, np.zeros(8))
            writer.finish(status="interrupted", complete=False, reason="test reset")
            self.assertEqual(len(writer.rows), 1)
            self.assertFalse(json.loads((writer.path / "metadata.json").read_text())["complete"])

    def test_validator_checks_rotation_terminal_and_legacy_definition(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "episode"
            writer = EpisodeWriter(path, width=3, height=2, fps=15, metadata={})
            states, poses, quats = [], [], []
            for t in range(3):
                quat = Rotation.from_euler("xyz", [.2 * t, -.3, .1]).as_quat()[[3, 0, 1, 2]]
                gripper = pose_matrix([.1 * t, .2, 1], quat)
                _, tcp = tcp_transforms(np.eye(4), np.eye(4), gripper)
                state = self.snapshot()
                state.update(base_tcp=tcp, world_base=np.eye(4), world_gripper=gripper,
                             joint_position=np.arange(7) + t)
                states.append(state)
                poses.append(gripper[:3, 3])
                quats.append(quat)
                if t < 2:
                    writer.append(state, np.zeros(8))
            terminal = {k: v for k, v in states[-1].items() if k not in ("rgb", "depth_m")}
            writer.finish(status="truncated", complete=False, terminal_state=terminal)
            with h5py.File(path / "trajectory.hdf5", "w") as file:
                demo = file.create_group("data/demo_0")
                for name, value in {
                    "actions": np.zeros((2, 8)),
                    "states/articulation/robot/joint_position": [s["joint_position"] for s in states[1:]],
                    "initial_state/articulation/robot/joint_position": [states[0]["joint_position"]],
                    "ee_pose/position": poses[1:], "ee_pose/orientation": quats[1:],
                }.items():
                    demo.create_dataset(name, data=value)
            self.assertEqual(validate(path)["tcp_geometry"], "passed")
            # Corrupt both camera/base RPY consistently: the HDF5 rotation
            # comparison must still reject a geometrically self-consistent label.
            cartesian_path = path / "observations/cartesian_position.npy"
            tcp_path = path / "TCP/third_person/state.npy"
            cartesian, tcp = np.load(cartesian_path), np.load(tcp_path)
            cartesian[1, 3] += .2
            tcp[1, 3] += .2
            np.save(cartesian_path, cartesian)
            np.save(tcp_path, tcp)
            with self.assertRaises(AssertionError):
                validate(path)
            meta = json.loads((path / "metadata.json").read_text())
            meta.pop("tcp_definition_id")
            (path / "metadata.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Legacy or unknown TCP"):
                validate(path)


if __name__ == "__main__":
    unittest.main()
