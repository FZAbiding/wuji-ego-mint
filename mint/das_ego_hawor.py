"""Run HaWoR on DAS rectified video and emit camera-frame MANO pseudo labels.

This file is intentionally executable in the standalone ``HaWoR`` conda
environment. Missing detections are never interpolated: inference is split at
every detection gap and ``hand_kept`` records only actual detector hits.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

IDENTITY_6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _contiguous_chunks(frames: np.ndarray, boxes: np.ndarray):
    if len(frames) == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    return [
        (frame_chunk, box_chunk)
        for frame_chunk, box_chunk in zip(
            np.split(frames, breaks), np.split(boxes, breaks), strict=True
        )
        if len(frame_chunk)
    ]


def _best_detections(tracks: dict, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Select the highest-confidence actual detection for each frame/hand."""
    best: dict[int, tuple[float, np.ndarray]] = {}
    for track in tracks.values():
        for item in track:
            if not bool(item.get("det", False)):
                continue
            handedness = int(np.asarray(item["det_handedness"]).reshape(-1)[0] > 0)
            if handedness != side:
                continue
            box = np.asarray(item["det_box"], dtype=np.float32).reshape(-1, 5)[0]
            frame = int(item["frame"])
            confidence = float(box[4])
            if frame not in best or confidence > best[frame][0]:
                best[frame] = (confidence, box[None])
    frames = np.asarray(sorted(best), dtype=np.int64)
    boxes = (
        np.concatenate([best[int(frame)][1] for frame in frames], axis=0)
        if len(frames)
        else np.empty((0, 5), dtype=np.float32)
    )
    return frames, boxes


def _matrix_to_6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def _normalize_rotation_6d(values: np.ndarray) -> np.ndarray:
    """Project the repository's row-major 6D representation back onto SO(3)."""
    values = np.array(values, dtype=np.float32, copy=True)
    first = values[..., :3]
    second = values[..., 3:]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second -= np.sum(first * second, axis=-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    return np.concatenate([first, second], axis=-1)


def _smooth_chunk(values: np.ndarray, sigma: float, *, rotation_6d: bool = False) -> np.ndarray:
    """Smooth one detection-contiguous chunk without borrowing from either gap."""
    values = np.asarray(values, dtype=np.float32)
    if sigma <= 0.0 or len(values) < 3:
        return values
    smoothed = gaussian_filter1d(values, sigma=float(sigma), axis=0, mode="nearest")
    return _normalize_rotation_6d(smoothed) if rotation_6d else smoothed


def run(args: argparse.Namespace) -> None:
    root = args.hawor_root.expanduser().resolve()
    if not (root / "scripts" / "scripts_test_video").is_dir():
        raise FileNotFoundError(f"not a HaWoR checkout: {root}")
    sys.path.insert(0, str(root))
    os.chdir(root)

    import torch
    from hawor.utils.rotation import (
        angle_axis_to_rotation_matrix,
        rotation_matrix_to_angle_axis,
    )
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_video import load_hawor

    class Options:
        pass

    options = Options()
    options.video_path = str(args.video.expanduser().resolve())
    options.img_focal = float(args.focal)
    options.input_type = "file"
    options.checkpoint = str(
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else root / "weights" / "hawor" / "checkpoints" / "hawor.ckpt"
    )
    if not Path(options.checkpoint).is_file():
        raise FileNotFoundError(options.checkpoint)

    start, end, sequence_dir, image_files = detect_track_video(options)
    if start != 0 or end != args.frames or len(image_files) != args.frames:
        raise RuntimeError(
            f"HaWoR extracted {len(image_files)} frames [{start},{end}), expected {args.frames}"
        )
    track_path = Path(sequence_dir) / f"tracks_{start}_{end}" / "model_tracks.npy"
    tracks = np.load(track_path, allow_pickle=True).item()

    model, _config = load_hawor(options.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    first = __import__("cv2").imread(str(image_files[0]))
    if first is None:
        raise RuntimeError(f"cannot read {image_files[0]}")
    center = [first.shape[1] / 2.0, first.shape[0] / 2.0]

    total = int(args.frames)
    kept = np.zeros((total, 2), dtype=bool)
    result: dict[str, np.ndarray] = {}
    for side_name in ("left", "right"):
        result[f"{side_name}_transl_cam"] = np.zeros((total, 3), dtype=np.float32)
        result[f"{side_name}_orient6d"] = np.tile(IDENTITY_6D, (total, 1))
        result[f"{side_name}_pose6d"] = np.tile(IDENTITY_6D, (total, 15))
        result[f"{side_name}_betas"] = np.zeros((total, 10), dtype=np.float32)

    for side, side_name in enumerate(("left", "right")):
        frames, boxes = _best_detections(tracks, side)
        for frame_chunk, box_chunk in _contiguous_chunks(frames, boxes):
            paths = np.asarray(image_files)[frame_chunk]
            with torch.no_grad():
                prediction = model.inference(
                    paths,
                    box_chunk,
                    img_focal=float(args.focal),
                    img_center=center,
                    do_flip=(side == 0),
                )
            root_matrix = prediction["pred_rotmat"][:, 0]
            pose_matrix = prediction["pred_rotmat"][:, 1:]
            if side == 0:
                root_aa = rotation_matrix_to_angle_axis(root_matrix)
                pose_aa = rotation_matrix_to_angle_axis(pose_matrix)
                root_aa[..., 1:] *= -1
                pose_aa[..., 1:] *= -1
                root_matrix = angle_axis_to_rotation_matrix(root_aa)
                pose_matrix = angle_axis_to_rotation_matrix(pose_aa)

            def array(value):
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                return np.asarray(value, dtype=np.float32)

            frame_chunk = np.asarray(frame_chunk, dtype=np.int64)
            root_np = array(root_matrix).reshape(-1, 3, 3)
            pose_np = array(pose_matrix).reshape(-1, 15, 3, 3)
            transl = array(prediction["pred_trans"]).reshape(len(frame_chunk), -1, 3)[:, 0]
            betas = array(prediction["pred_shape"]).reshape(len(frame_chunk), -1)[:, :10]
            root_6d = _smooth_chunk(
                _matrix_to_6d(root_np), args.smooth_sigma, rotation_6d=True
            )
            pose_6d = _smooth_chunk(
                _matrix_to_6d(pose_np), args.smooth_sigma, rotation_6d=True
            )
            result[f"{side_name}_transl_cam"][frame_chunk] = _smooth_chunk(
                transl, args.smooth_sigma
            )
            result[f"{side_name}_orient6d"][frame_chunk] = root_6d
            result[f"{side_name}_pose6d"][frame_chunk] = pose_6d.reshape(-1, 90)
            result[f"{side_name}_betas"][frame_chunk] = _smooth_chunk(
                betas, args.smooth_sigma
            )
            kept[frame_chunk, side] = True

    result["hand_kept"] = kept
    result["frame_index"] = np.arange(total, dtype=np.int64)
    result["label_kind"] = np.asarray("pseudo_gt")
    result["smoothing_sigma"] = np.asarray(args.smooth_sigma, dtype=np.float32)
    for name, values in result.items():
        if np.issubdtype(np.asarray(values).dtype, np.number) and not np.isfinite(values).all():
            raise RuntimeError(f"HaWoR output {name} contains non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(
        f"[das-ego-hawor] wrote {args.output}; "
        f"left={int(kept[:, 0].sum())}/{total}, right={int(kept[:, 1].sum())}/{total}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hawor-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--focal", type=float, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma applied independently inside each consecutive detection run",
    )
    args = parser.parse_args(argv)
    if args.smooth_sigma < 0.0:
        parser.error("--smooth-sigma must be non-negative")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
