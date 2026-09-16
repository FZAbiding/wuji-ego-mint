from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mint.das_ego import (
    DEFAULT_FOV,
    CameraCalibration,
    _nearest_indices,
    _write_camera_translation_normalization,
    build_parser,
    double_sphere_remap,
)
from mint.das_ego_hawor import (
    _best_detections,
    _contiguous_chunks,
    _normalize_rotation_6d,
    _smooth_chunk,
)


class DasEgoGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = CameraCalibration(
            camera="camera2",
            topic="/robot0/sensor/camera2/camera_info",
            model="double_sphere",
            width=1600,
            height=1300,
            params=(510.884, 512.746, 801.168, 652.935, -0.00316, 0.57121),
            T_b_c=tuple(tuple(row) for row in np.eye(4)),
            source_mcap="fixture.mcap",
            source_log_time_ns=0,
        )

    def test_rectification_maps_center_and_shape(self) -> None:
        map_x, map_y, K = double_sphere_remap(
            self.calibration, 512, DEFAULT_FOV, DEFAULT_FOV
        )
        self.assertEqual(map_x.shape, (512, 512))
        self.assertEqual(map_y.shape, (512, 512))
        self.assertEqual(K.shape, (3, 3))
        center_error = np.linalg.norm(
            [map_x[256, 256] - self.calibration.cx,
             map_y[256, 256] - self.calibration.cy]
        )
        self.assertLess(center_error, 0.5)

    def test_nearest_timestamp_matching(self) -> None:
        reference = np.asarray([0, 10, 20, 30], dtype=np.int64)
        query = np.asarray([1, 6, 29], dtype=np.int64)
        indices, errors = _nearest_indices(reference, query)
        np.testing.assert_array_equal(indices, [0, 1, 3])
        np.testing.assert_array_equal(errors, [1, 4, 1])

    def test_camera_normalization_metadata(self) -> None:
        frames = 40
        step = np.arange(frames, dtype=np.float32)
        translation = np.stack(
            [step * 0.01, np.sin(step * 0.1) * 0.02, np.cos(step * 0.07) * 0.03],
            axis=1,
        )
        quaternion = np.tile([0, 0, 0, 1], (frames, 1)).astype(np.float32)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
            path = _write_camera_translation_normalization(
                root, translation, quaternion, clip_len=32
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["stats"]["32"]["num_clips"], 9)
            scale = np.asarray(payload["stats"]["32"]["trans_std_m"])
            self.assertEqual(scale.shape, (3,))
            self.assertTrue(np.all(np.isfinite(scale) & (scale > 0)))


class DasEgoCliTest(unittest.TestCase):
    def test_omitted_duration_means_full_sequence(self) -> None:
        args = build_parser().parse_args(
            [
                "prepare",
                "--dataset", "dataset",
                "--master-mcap", "master.mcap",
                "--output", "output",
            ]
        )
        self.assertIsNone(args.duration_sec)

    def test_explicit_demo_duration(self) -> None:
        args = build_parser().parse_args(
            [
                "prepare",
                "--dataset", "dataset",
                "--master-mcap", "master.mcap",
                "--duration-sec", "15",
                "--output", "output",
            ]
        )
        self.assertEqual(args.duration_sec, 15.0)


class DasEgoHaworTest(unittest.TestCase):
    def test_missing_detections_split_runs_and_stay_missing(self) -> None:
        tracks = {
            1: [
                {"frame": 0, "det": True, "det_handedness": [0],
                 "det_box": [[0, 0, 10, 10, 0.4]]},
                {"frame": 1, "det": False, "det_handedness": [0],
                 "det_box": [[0, 0, 0, 0, 0.0]]},
                {"frame": 2, "det": True, "det_handedness": [0],
                 "det_box": [[2, 2, 12, 12, 0.7]]},
            ],
            2: [
                {"frame": 0, "det": True, "det_handedness": [0],
                 "det_box": [[1, 1, 11, 11, 0.9]]},
                {"frame": 1, "det": True, "det_handedness": [1],
                 "det_box": [[1, 1, 11, 11, 0.8]]},
            ],
        }
        frames, boxes = _best_detections(tracks, side=0)
        np.testing.assert_array_equal(frames, [0, 2])
        self.assertEqual(float(boxes[0, 4]), np.float32(0.9))
        chunks = _contiguous_chunks(frames, boxes)
        self.assertEqual([chunk.tolist() for chunk, _ in chunks], [[0], [2]])

    def test_short_detection_run_is_not_smoothed(self) -> None:
        values = np.asarray([[0.0], [2.0]], dtype=np.float32)
        np.testing.assert_array_equal(_smooth_chunk(values, 1.0), values)

    def test_smoothed_rotation_6d_stays_orthonormal(self) -> None:
        values = np.asarray(
            [[1, 0, 0, 0, 1, 0], [1, 0.1, 0, 0, 1, 0.1], [1, 0, 0, 0, 1, 0]],
            dtype=np.float32,
        )
        result = _smooth_chunk(values, 1.0, rotation_6d=True)
        first, second = result[:, :3], result[:, 3:]
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(second, axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.sum(first * second, axis=1), 0.0, atol=1e-6)

    def test_rotation_projection_accepts_batched_joints(self) -> None:
        identity = np.tile([1, 0, 0, 0, 1, 0], (4, 15, 1)).astype(np.float32)
        np.testing.assert_array_equal(_normalize_rotation_6d(identity), identity)


if __name__ == "__main__":
    unittest.main()
