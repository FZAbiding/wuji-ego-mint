"""DAS-Ego V6 preparation, VIO alignment, HaWoR orchestration and export.

The MCAP payloads used here are Foxglove protobuf messages. Only the stable
wire fields needed by this adapter are decoded, so generated DAS protobuf
modules are not required.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import venv
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

VIDEO_KEY = "observation.images.ego"
CAMERA_RE = re.compile(r"camera([0-5])$")
UUID_RE = re.compile(r"_([0-9a-fA-F]{8})\.mcap$")
DEFAULT_FOV = 1.7157
DEFAULT_SIZE = 512
DEFAULT_FPS = 30.0
MANO_COLUMNS = ("transl_cam", "orient6d", "pose6d", "betas")


@dataclass(frozen=True)
class CameraCalibration:
    camera: str
    topic: str
    model: str
    width: int
    height: int
    params: tuple[float, float, float, float, float, float]
    T_b_c: tuple[tuple[float, ...], ...]
    source_mcap: str
    source_log_time_ns: int

    @property
    def fx(self) -> float:
        return self.params[0]

    @property
    def fy(self) -> float:
        return self.params[1]

    @property
    def cx(self) -> float:
        return self.params[2]

    @property
    def cy(self) -> float:
        return self.params[3]

    @property
    def xi(self) -> float:
        return self.params[4]

    @property
    def alpha(self) -> float:
        return self.params[5]


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift >= 70:
            raise ValueError("invalid protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def _protobuf_fields(data: bytes) -> Iterator[tuple[int, int, Any]]:
    """Yield ``(field_number, wire_type, value)`` for a protobuf message."""
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        number, wire = key >> 3, key & 7
        if number <= 0:
            raise ValueError("invalid protobuf field number")
        if wire == 0:
            value, offset = _read_varint(data, offset)
        elif wire == 1:
            if offset + 8 > len(data):
                raise ValueError("truncated protobuf fixed64")
            value = data[offset : offset + 8]
            offset += 8
        elif wire == 2:
            length, offset = _read_varint(data, offset)
            if offset + length > len(data):
                raise ValueError("truncated protobuf bytes")
            value = data[offset : offset + length]
            offset += length
        elif wire == 5:
            if offset + 4 > len(data):
                raise ValueError("truncated protobuf fixed32")
            value = data[offset : offset + 4]
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def _field_map(data: bytes) -> dict[int, list[tuple[int, Any]]]:
    result: dict[int, list[tuple[int, Any]]] = {}
    for number, wire, value in _protobuf_fields(data):
        result.setdefault(number, []).append((wire, value))
    return result


def _packed_doubles(fields: dict[int, list[tuple[int, Any]]], number: int) -> list[float]:
    values: list[float] = []
    for wire, value in fields.get(number, []):
        if wire == 1:
            values.append(struct.unpack("<d", value)[0])
        elif wire == 2:
            if len(value) % 8:
                raise ValueError(f"protobuf field {number} is not packed doubles")
            values.extend(struct.unpack("<" + "d" * (len(value) // 8), value))
        else:
            raise ValueError(f"protobuf field {number} has wire type {wire}, expected double")
    return values


def _scalar_fixed32(fields: dict[int, list[tuple[int, Any]]], number: int) -> int:
    try:
        wire, value = fields[number][-1]
    except KeyError as error:
        raise ValueError(f"missing protobuf field {number}") from error
    if wire == 5:
        return int(struct.unpack("<I", value)[0])
    if wire == 0:
        return int(value)
    raise ValueError(f"protobuf field {number} is not uint32")


def _text_field(fields: dict[int, list[tuple[int, Any]]], number: int) -> str:
    try:
        wire, value = fields[number][-1]
    except KeyError as error:
        raise ValueError(f"missing protobuf field {number}") from error
    if wire != 2:
        raise ValueError(f"protobuf field {number} is not text")
    return value.decode("utf-8")


def _validate_transform(transform: np.ndarray, name: str) -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant is not 1")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")


def _transform_from_tq(values: list[float]) -> np.ndarray:
    if len(values) == 16:
        transform = np.asarray(values, dtype=np.float64).reshape(4, 4)
    elif len(values) == 7:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Rotation.from_quat(values[3:7]).as_matrix()
        transform[:3, 3] = values[:3]
    else:
        raise ValueError(f"T_b_c must have 7 or 16 values, got {len(values)}")
    _validate_transform(transform, "T_b_c")
    return transform


def read_camera_calibration(master_mcap: Path, camera: str) -> CameraCalibration:
    """Read and validate one DAS Double Sphere CameraCalibration message."""
    if CAMERA_RE.fullmatch(camera) is None:
        raise ValueError(f"camera must be camera0..camera5, got {camera!r}")
    try:
        from mcap.reader import make_reader
    except ImportError as error:
        raise RuntimeError(
            "DAS-Ego MCAP support is not installed; reinstall this project so mcap is present"
        ) from error

    master_mcap = Path(master_mcap).expanduser().resolve()
    topic = f"/robot0/sensor/{camera}/camera_info"
    with master_mcap.open("rb") as stream:
        reader = make_reader(stream)
        for schema, _channel, message in reader.iter_messages(
            topics=[topic], log_time_order=False
        ):
            if schema is None or schema.name != "foxglove.CameraCalibration":
                raise ValueError(
                    f"{topic}: expected foxglove.CameraCalibration, got "
                    f"{None if schema is None else schema.name!r}"
                )
            fields = _field_map(bytes(message.data))
            model = _text_field(fields, 4).strip().lower()
            if model not in {"ds", "double_sphere", "double-sphere"}:
                raise ValueError(f"{topic}: expected Double Sphere model, got {model!r}")
            params = _packed_doubles(fields, 5)
            if len(params) != 6:
                raise ValueError(
                    f"{topic}: D must be [fx,fy,cx,cy,xi,alpha], got {len(params)} values"
                )
            if not np.isfinite(params).all() or params[0] <= 0 or params[1] <= 0:
                raise ValueError(f"{topic}: invalid Double Sphere parameters {params}")
            if not 0.0 <= params[5] <= 1.0:
                raise ValueError(f"{topic}: alpha must be in [0,1], got {params[5]}")
            transform = _transform_from_tq(_packed_doubles(fields, 10))
            return CameraCalibration(
                camera=camera,
                topic=topic,
                model="double_sphere",
                width=_scalar_fixed32(fields, 2),
                height=_scalar_fixed32(fields, 3),
                params=tuple(float(value) for value in params),
                T_b_c=tuple(tuple(float(value) for value in row) for row in transform),
                source_mcap=str(master_mcap),
                source_log_time_ns=int(message.log_time),
            )
    raise ValueError(f"{master_mcap}: missing calibration topic {topic}")


def target_intrinsics(size: int, vfov: float, hfov: float) -> np.ndarray:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0.0 < vfov < math.pi or not 0.0 < hfov < math.pi:
        raise ValueError("target FoV must be between 0 and pi radians")
    center = size / 2.0  # MINT's camera codec assumes W/2,H/2.
    return np.array(
        [
            [(size / 2.0) / math.tan(hfov / 2.0), 0.0, center],
            [0.0, (size / 2.0) / math.tan(vfov / 2.0), center],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def double_sphere_remap(
    calibration: CameraCalibration, size: int, vfov: float, hfov: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build OpenCV maps from a pinhole target pixel to a Double Sphere source."""
    K = target_intrinsics(size, vfov, hfov)
    grid_x, grid_y = np.meshgrid(
        np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64)
    )
    x = (grid_x - K[0, 2]) / K[0, 0]
    y = (grid_y - K[1, 2]) / K[1, 1]
    z = np.ones_like(x)
    radius = np.sqrt(x * x + y * y + z * z)
    z_xi = calibration.xi * radius + z
    denom = calibration.alpha * np.sqrt(x * x + y * y + z_xi * z_xi) + (
        1.0 - calibration.alpha
    ) * z_xi
    valid = np.isfinite(denom) & (denom > 1e-12)
    map_x = np.full_like(x, -1.0, dtype=np.float64)
    map_y = np.full_like(y, -1.0, dtype=np.float64)
    map_x[valid] = calibration.fx * x[valid] / denom[valid] + calibration.cx
    map_y[valid] = calibration.fy * y[valid] / denom[valid] + calibration.cy
    valid &= (
        (map_x >= 0)
        & (map_x <= calibration.width - 1)
        & (map_y >= 0)
        & (map_y <= calibration.height - 1)
    )
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x.astype(np.float32), map_y.astype(np.float32), K


def _read_dataset_timeline(dataset: Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps") or DEFAULT_FPS)
    paths = sorted((dataset / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files below {dataset / 'data'}")
    pieces = []
    for path in paths:
        names = set(pq.read_schema(path).names)
        required = {"episode_index", "frame_index", "observation.source_timestamp"}
        if not required <= names:
            raise ValueError(f"{path}: missing columns {sorted(required - names)}")
        pieces.append(pq.read_table(path, columns=sorted(required)))
    table = pa.concat_tables(pieces).sort_by(
        [("episode_index", "ascending"), ("frame_index", "ascending")]
    )
    episodes = np.asarray(table["episode_index"])
    unique = np.unique(episodes)
    if unique.tolist() != [0]:
        raise ValueError(f"adapter requires one episode indexed 0, got {unique.tolist()}")
    frames = np.asarray(table["frame_index"], dtype=np.int64)
    if not np.array_equal(frames, np.arange(len(frames), dtype=np.int64)):
        raise ValueError("source frame_index must be contiguous and start at zero")
    timestamps = np.asarray(table["observation.source_timestamp"], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("observation.source_timestamp must be strictly increasing")
    return timestamps, fps, info


def _source_video(dataset: Path, camera: str) -> Path:
    root = dataset / "videos" / f"observation.images.{camera}"
    videos = sorted(root.rglob("*.mp4"))
    if len(videos) != 1:
        raise ValueError(f"expected exactly one {camera} MP4 below {root}, found {len(videos)}")
    return videos[0]


def _ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError("ffmpeg is required to encode the rectified H.264 video")
    return binary


def _encode_rectified(
    source: Path,
    output: Path,
    *,
    start_frame: int,
    frame_count: int,
    fps: float,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video {source}")
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame)):
        capture.release()
        raise RuntimeError(f"cannot seek {source} to frame {start_frame}")
    size = int(map_x.shape[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    command = [
        _ffmpeg_binary(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size}x{size}",
        "-r", f"{fps:.12g}", "-i", "pipe:0", "-an",
        "-frames:v", str(frame_count), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-g", str(max(1, int(round(fps)))), str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for local_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"{source}: decode stopped at source frame {start_frame + local_index}"
                )
            rectified = cv2.remap(
                frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            process.stdin.write(rectified.tobytes())
        process.stdin.close()
        process.stdin = None
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        os.replace(temporary, output)
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot probe output video {path}")
    result = {
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    capture.release()
    return result


def prepare(
    dataset: Path,
    master_mcap: Path,
    output: Path,
    *,
    camera: str = "camera2",
    start_sec: float = 3.0,
    duration_sec: float | None = 15.0,
    size: int = DEFAULT_SIZE,
    vfov: float = DEFAULT_FOV,
    hfov: float = DEFAULT_FOV,
    overwrite: bool = False,
) -> Path:
    dataset = Path(dataset).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    calibration = read_camera_calibration(master_mcap, camera)
    timestamps, fps, source_info = _read_dataset_timeline(dataset)
    source = _source_video(dataset, camera)
    start_frame = int(round(float(start_sec) * fps))
    if start_frame < 0 or start_frame >= len(timestamps):
        raise ValueError(f"start frame {start_frame} is outside 0..{len(timestamps) - 1}")
    if duration_sec is None:
        frame_count = len(timestamps) - start_frame
    else:
        if duration_sec <= 0:
            raise ValueError("duration-sec must be positive when provided")
        frame_count = min(
            int(round(float(duration_sec) * fps)), len(timestamps) - start_frame
        )
    if frame_count <= 0:
        raise ValueError("selected source range contains no frames")

    prepared = output / "prepared"
    video = prepared / f"{camera}_rectified.mp4"
    manifest_path = prepared / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite")
    prepared.mkdir(parents=True, exist_ok=True)
    map_x, map_y, K = double_sphere_remap(calibration, size, vfov, hfov)
    center = size // 2
    center_error = float(
        np.linalg.norm(
            [map_x[center, center] - calibration.cx, map_y[center, center] - calibration.cy]
        )
    )
    if center_error >= 0.5:
        raise RuntimeError(f"rectification center mapping error {center_error:.6f}px >= 0.5px")
    _encode_rectified(
        source, video, start_frame=start_frame, frame_count=frame_count,
        fps=fps, map_x=map_x, map_y=map_y,
    )
    probe = _probe_video(video)
    if probe["width"] != size or probe["height"] != size or probe["frames"] != frame_count:
        raise RuntimeError(f"rectified video verification failed: {probe}")
    if abs(probe["fps"] - fps) > 1e-3:
        raise RuntimeError(f"rectified video fps {probe['fps']} != source fps {fps}")

    selected_timestamps = timestamps[start_frame : start_frame + frame_count]
    np.save(prepared / "timestamps_ns.npy", selected_timestamps)
    calibration_payload = asdict(calibration)
    calibration_payload.update(
        {
            "target_size": [size, size],
            "target_fov_hw_rad": [float(vfov), float(hfov)],
            "target_K": K.tolist(),
            "center_mapping_error_px": center_error,
        }
    )
    (prepared / "calibration.json").write_text(
        json.dumps(calibration_payload, indent=2), encoding="utf-8"
    )
    manifest = {
        "format": "mint.das_ego.prepared.v1",
        "dataset": str(dataset),
        "master_mcap": str(Path(master_mcap).expanduser().resolve()),
        "camera": camera,
        "source_video": str(source),
        "video": str(video),
        "timestamps_ns": str(prepared / "timestamps_ns.npy"),
        "calibration": str(prepared / "calibration.json"),
        "fps": fps,
        "start_sec": float(start_sec),
        "duration_sec": None if duration_sec is None else float(duration_sec),
        "source_frame_start": start_frame,
        "source_frame_end_exclusive": start_frame + frame_count,
        "frame_count": frame_count,
        "target_size": [size, size],
        "target_fov_hw_rad": [float(vfov), float(hfov)],
        "target_K": K.tolist(),
        "source_total_frames": int(source_info.get("total_frames", len(timestamps))),
        "video_probe": probe,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return prepared


def _nested_double(message: bytes, number: int, *, default: float = 0.0) -> float:
    entries = _field_map(message).get(number)
    if not entries:
        return default
    wire, value = entries[-1]
    if wire != 1:
        raise ValueError(f"protobuf field {number} is not double")
    return float(struct.unpack("<d", value)[0])


def _decode_vector3(data: bytes | None) -> np.ndarray:
    if not data:
        return np.zeros(3, dtype=np.float64)
    return np.array([_nested_double(data, index) for index in (1, 2, 3)], dtype=np.float64)


def _decode_quaternion(data: bytes | None) -> np.ndarray:
    if not data:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return np.array([_nested_double(data, index) for index in (1, 2, 3, 4)], dtype=np.float64)


def _last_bytes(fields: dict[int, list[tuple[int, Any]]], number: int) -> bytes | None:
    entries = fields.get(number)
    if not entries:
        return None
    wire, value = entries[-1]
    if wire != 2:
        raise ValueError(f"protobuf field {number} is not a nested message")
    return bytes(value)


def _decode_pose_in_frame(data: bytes) -> np.ndarray:
    fields = _field_map(data)
    pose_fields = _field_map(_last_bytes(fields, 3) or b"")
    position = _decode_vector3(_last_bytes(pose_fields, 1))
    quaternion = _decode_quaternion(_last_bytes(pose_fields, 2))
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("VIO pose contains an invalid quaternion")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    transform[:3, 3] = position
    return transform


def read_vio_poses(vio_mcap: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from mcap.reader import make_reader
    except ImportError as error:
        raise RuntimeError("mcap is required to read DAS VIO output") from error
    topic = "/robot0/vio/eef_pose"
    timestamps: list[int] = []
    poses: list[np.ndarray] = []
    with Path(vio_mcap).expanduser().resolve().open("rb") as stream:
        for schema, _channel, message in make_reader(stream).iter_messages(
            topics=[topic], log_time_order=True
        ):
            if schema is None or schema.name != "foxglove.PoseInFrame":
                raise ValueError(f"{topic}: expected foxglove.PoseInFrame")
            timestamps.append(int(message.log_time))
            poses.append(_decode_pose_in_frame(bytes(message.data)))
    if not poses:
        raise ValueError(f"{vio_mcap}: missing {topic}")
    times = np.asarray(timestamps, dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("VIO timestamps must be strictly increasing")
    return times, np.stack(poses)


def _nearest_indices(reference_ns: np.ndarray, query_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    after = np.searchsorted(reference_ns, query_ns, side="left")
    right = np.clip(after, 0, len(reference_ns) - 1)
    left = np.clip(after - 1, 0, len(reference_ns) - 1)
    use_left = np.abs(reference_ns[left] - query_ns) <= np.abs(reference_ns[right] - query_ns)
    indices = np.where(use_left, left, right)
    errors = np.abs(reference_ns[indices] - query_ns)
    return indices.astype(np.int64), errors.astype(np.int64)


def _validate_camera_track(
    cam_c2w: np.ndarray, cam_trans: np.ndarray, cam_quat: np.ndarray
) -> None:
    if not np.allclose(cam_c2w[0], np.eye(4), atol=1e-6):
        raise RuntimeError("first relative camera pose is not identity")
    if not np.isfinite(cam_c2w).all() or not np.isfinite(cam_trans).all():
        raise RuntimeError("camera track contains non-finite values")
    rotations = cam_c2w[:, :3, :3]
    orthogonal = np.matmul(np.swapaxes(rotations, 1, 2), rotations)
    if not np.allclose(orthogonal, np.eye(3)[None], atol=1e-5):
        raise RuntimeError("camera track contains non-orthogonal rotations")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5):
        raise RuntimeError("camera track rotation determinant is not 1")
    if not np.allclose(np.linalg.norm(cam_quat, axis=1), 1.0, atol=1e-5):
        raise RuntimeError("camera quaternions are not normalized")
    if len(cam_quat) > 1 and np.any(np.sum(cam_quat[:-1] * cam_quat[1:], axis=1) < 0.0):
        raise RuntimeError("camera quaternion track contains sign jumps")


def align_vio(
    prepared_dir: Path,
    vio_mcap: Path,
    *,
    max_error_ns: int | None = None,
) -> Path:
    prepared_dir = Path(prepared_dir).resolve()
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    calibration_json = json.loads(Path(manifest["calibration"]).read_text(encoding="utf-8"))
    frame_times = np.load(manifest["timestamps_ns"]).astype(np.int64)
    vio_times, T_w_b = read_vio_poses(vio_mcap)
    indices, errors = _nearest_indices(vio_times, frame_times)
    allowed = int(
        max_error_ns if max_error_ns is not None else round(1e9 / manifest["fps"] / 2)
    )
    if np.any(errors > allowed):
        worst = int(np.argmax(errors))
        raise RuntimeError(
            f"VIO alignment failed at frame {worst}: {errors[worst] / 1e6:.3f}ms "
            f"> {allowed / 1e6:.3f}ms"
        )
    T_b_c = np.asarray(calibration_json["T_b_c"], dtype=np.float64).reshape(4, 4)
    T_w_c = T_w_b[indices] @ T_b_c
    T_c0_c = np.linalg.inv(T_w_c[0]) @ T_w_c
    # MINT's compact camera encoding stores OpenCV world-to-camera [R|t].
    T_w2c = np.linalg.inv(T_c0_c)
    translations = T_w2c[:, :3, 3]
    quaternions = Rotation.from_matrix(T_w2c[:, :3, :3]).as_quat()
    for index in range(1, len(quaternions)):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
            quaternions[index] *= -1.0
    _validate_camera_track(T_c0_c, translations, quaternions)
    output = prepared_dir.parent / "vio_aligned.npz"
    np.savez_compressed(
        output,
        source_vio_mcap=np.asarray(str(Path(vio_mcap).expanduser().resolve())),
        timestamps_ns=frame_times,
        vio_timestamps_ns=vio_times[indices],
        alignment_error_ns=errors,
        cam_c2w=T_c0_c,
        cam_trans=translations.astype(np.float32),
        cam_quat=quaternions.astype(np.float32),
        cam_fov=np.tile(
            np.asarray(manifest["target_fov_hw_rad"], dtype=np.float32),
            (len(frame_times), 1),
        ),
    )
    return output


def _uuid(path: Path) -> str:
    match = UUID_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot extract trailing UUID from {path.name}")
    return match.group(1).lower()


def _find_triplet(master_mcap: Path) -> tuple[Path, Path, Path, str]:
    master = Path(master_mcap).expanduser().resolve()
    uuid = _uuid(master)
    if "_master_" not in master.name:
        raise ValueError(f"master MCAP name does not contain _master_: {master.name}")
    left = sorted(master.parent.glob(f"DAS-Finger_*_sub_left_*_{uuid}.mcap"))
    right = sorted(master.parent.glob(f"DAS-Finger_*_sub_right_*_{uuid}.mcap"))
    if len(left) != 1 or len(right) != 1:
        raise FileNotFoundError(
            f"UUID {uuid}: expected one sub_left and one sub_right beside {master}; "
            f"found {len(left)} and {len(right)}"
        )
    return master, left[0], right[0], uuid


def _ensure_delivery_cli(stack: Path) -> Path:
    stack = stack.resolve()
    wheel = stack / "delivery_pipeline-1.0.4-py3-none-any.whl"
    if not wheel.is_file():
        raise FileNotFoundError(wheel)
    environment = stack / ".venv-delivery"
    executable = environment / "bin" / "delivery-pipeline"
    if not executable.is_file():
        venv.EnvBuilder(with_pip=True, clear=False).create(environment)
        subprocess.run(
            [str(environment / "bin" / "python"), "-m", "pip", "install", str(wheel)],
            check=True,
        )
    return executable


def run_das_pipeline(master_mcap: Path, output_root: Path, stack: Path) -> Path:
    master, _left, _right, uuid = _find_triplet(master_mcap)
    executable = _ensure_delivery_cli(stack)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    registry = "imagepublic.genrobotai.com/genrobot/genimage"
    env = os.environ.copy()
    for step in ("qc", "merge", "vio", "vio_check"):
        env.setdefault(f"ALGO_{step.upper()}_IMAGE", f"{registry}:{step}-v1.0.5")
    subprocess.run(
        [
            str(executable), "--steps", "qc,merge,vio,vio_check",
            "--input-dir", str(master.parent), "--output-dir", str(output_root),
        ],
        check=True,
        env=env,
    )
    result = output_root / uuid / "merged_output_ego_vio.mcap"
    if not result.is_file() or result.stat().st_size == 0:
        raise RuntimeError(f"DAS pipeline did not produce {result}")
    return result


def _run_hawor(
    prepared_dir: Path,
    output: Path,
    *,
    environment: str,
    hawor_root: Path,
    smooth_sigma: float,
) -> Path:
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    K = np.asarray(manifest["target_K"], dtype=np.float64)
    runner = Path(__file__).with_name("das_ego_hawor.py")
    command = [
        "mamba", "run", "-n", environment, "python", str(runner),
        "--hawor-root", str(Path(hawor_root).expanduser().resolve()),
        "--video", str(Path(manifest["video"]).resolve()),
        "--focal", f"{K[0, 0]:.12g}", "--output", str(output.resolve()),
        "--frames", str(int(manifest["frame_count"])),
        "--smooth-sigma", f"{smooth_sigma:.12g}",
    ]
    subprocess.run(command, check=True)
    if not output.is_file():
        raise RuntimeError(f"HaWoR did not produce {output}")
    return output


def _fixed_list(values: np.ndarray, width: int, name: str) -> pa.FixedSizeListArray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape [T,{width}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    flat = pa.array(array.astype(np.float32, copy=False).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _load_hawor(path: Path, length: int) -> dict[str, np.ndarray]:
    required = {
        f"{side}_{column}" for side in ("left", "right") for column in MANO_COLUMNS
    } | {"hand_kept"}
    with np.load(path) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing HaWoR arrays {sorted(missing)}")
        result = {name: np.asarray(data[name]) for name in required}
    expected = {
        "transl_cam": (length, 3), "orient6d": (length, 6),
        "pose6d": (length, 90), "betas": (length, 10),
    }
    kept = np.asarray(result["hand_kept"], dtype=bool)
    if kept.shape != (length, 2):
        raise ValueError(f"hand_kept must have shape {(length, 2)}, got {kept.shape}")
    result["hand_kept"] = kept
    for side in ("left", "right"):
        for column, shape in expected.items():
            name = f"{side}_{column}"
            values = np.asarray(result[name], dtype=np.float32)
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite placeholders")
            result[name] = values
    return result


def _write_camera_translation_normalization(
    dataset: Path,
    cam_trans: np.ndarray,
    cam_quat: np.ndarray,
    *,
    clip_len: int = 32,
    min_std: float = 1e-6,
) -> Path:
    """Write the dataset-local scale file consumed by Viewer camera losses."""
    if len(cam_trans) < clip_len:
        raise ValueError(
            f"camera normalization needs at least {clip_len} frames, got {len(cam_trans)}"
        )
    rotations = Rotation.from_quat(cam_quat).as_matrix()
    starts = np.arange(len(cam_trans) - clip_len + 1, dtype=np.int64)
    rows = starts[:, None] + np.arange(clip_len, dtype=np.int64)[None]
    relative_rotation = rotations[rows] @ np.swapaxes(
        rotations[starts], -1, -2
    )[:, None]
    rebased = cam_trans[rows] - (
        relative_rotation @ cam_trans[starts, None, :, None]
    )[..., 0]
    raw_std = np.std(rebased.reshape(-1, 3).astype(np.float64), axis=0)
    scale = np.maximum(raw_std, float(min_std))
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise RuntimeError(f"invalid camera translation normalization scale {scale}")

    info_path = dataset / "meta" / "info.json"
    info_sha256 = hashlib.sha256(info_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "normalization": {
            "type": "scale_only",
            "subtract_mean": False,
            "select_stats_by": "clip_len",
        },
        "target_definition": {"rebase_to_each_clip_first_frame": True},
        "sampling": {
            "temporal_frame_step": 1,
            "complete_dataset_scan": True,
            "max_data_files_per_dataset": None,
        },
        "datasets": [
            {
                "root_relative_to_input_anchor": ".",
                "data_files_read": 1,
                "data_files_total": 1,
                "info_sha256": info_sha256,
            }
        ],
        "stats": {
            str(clip_len): {
                "num_clips": int(len(starts)),
                "trans_std_m": scale.tolist(),
                "trans_std_m_before_floor": raw_std.tolist(),
            }
        },
    }
    path = dataset / "meta" / "camera_translation_normalization.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_lerobot(
    prepared_dir: Path,
    vio_npz: Path,
    hawor_npz: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> Path:
    prepared_dir = Path(prepared_dir).resolve()
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    length = int(manifest["frame_count"])
    timestamps = np.load(manifest["timestamps_ns"]).astype(np.int64)
    with np.load(vio_npz) as data:
        cam_trans = np.asarray(data["cam_trans"], dtype=np.float32)
        cam_quat = np.asarray(data["cam_quat"], dtype=np.float32)
        cam_fov = np.asarray(data["cam_fov"], dtype=np.float32)
        alignment_error = np.asarray(data["alignment_error_ns"], dtype=np.int64)
        source_vio_mcap = (
            str(data["source_vio_mcap"].item())
            if "source_vio_mcap" in data.files
            else None
        )
    for name, array, width in (
        ("cam_trans", cam_trans, 3), ("cam_quat", cam_quat, 4),
        ("cam_fov", cam_fov, 2),
    ):
        if array.shape != (length, width) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite with shape {(length, width)}, got {array.shape}")
    hands = _load_hawor(Path(hawor_npz), length)
    with np.load(hawor_npz) as data:
        smoothing_sigma = (
            float(np.asarray(data["smoothing_sigma"]).item())
            if "smoothing_sigma" in data.files
            else None
        )
    output = Path(output).expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    data_dir = output / "data" / "chunk-000"
    episodes_dir = output / "meta" / "episodes" / "chunk-000"
    video_dir = output / "videos" / VIDEO_KEY / "chunk-000"
    data_dir.mkdir(parents=True)
    episodes_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)

    state_mask = np.tile(hands["hand_kept"].any(axis=0), (length, 1))
    columns: dict[str, Any] = {
        "frame_index": pa.array(np.arange(length, dtype=np.int64)),
        "episode_index": pa.array(np.zeros(length, dtype=np.int64)),
        "index": pa.array(np.arange(length, dtype=np.int64)),
        "task_index": pa.array(np.zeros(length, dtype=np.int64)),
        "timestamp": pa.array(np.arange(length, dtype=np.float64) / float(manifest["fps"])),
        "source_timestamp_ns": pa.array(timestamps),
        "state_mask": pa.FixedSizeListArray.from_arrays(pa.array(state_mask.reshape(-1)), 2),
        "hand_kept": pa.FixedSizeListArray.from_arrays(
            pa.array(hands["hand_kept"].reshape(-1)), 2
        ),
        "cam_trans": _fixed_list(cam_trans, 3, "cam_trans"),
        "cam_quat": _fixed_list(cam_quat, 4, "cam_quat"),
        "cam_fov": _fixed_list(cam_fov, 2, "cam_fov"),
    }
    widths = {"transl_cam": 3, "orient6d": 6, "pose6d": 90, "betas": 10}
    for side in ("left", "right"):
        for column, width in widths.items():
            name = f"{side}_mano_{column}"
            columns[name] = _fixed_list(hands[f"{side}_{column}"], width, name)
    pq.write_table(pa.table(columns), data_dir / "file-000.parquet", compression="zstd")
    shutil.copy2(manifest["video"], video_dir / "file-000.mp4")

    task = f"DAS-Ego {manifest['camera']} hand interaction"
    task_table = pa.Table.from_pylist([{"task_index": 0, "task": task}])
    pq.write_table(task_table, output / "meta" / "tasks.parquet")
    episode = pa.table(
        {
            "episode_index": pa.array([0], type=pa.int64()),
            "tasks": pa.array([[task]], type=pa.list_(pa.string())),
            "length": pa.array([length], type=pa.int64()),
            "data/chunk_index": pa.array([0], type=pa.int64()),
            "data/file_index": pa.array([0], type=pa.int64()),
            "dataset_from_index": pa.array([0], type=pa.int64()),
            "dataset_to_index": pa.array([length], type=pa.int64()),
            f"videos/{VIDEO_KEY}/chunk_index": pa.array([0], type=pa.int64()),
            f"videos/{VIDEO_KEY}/file_index": pa.array([0], type=pa.int64()),
            f"videos/{VIDEO_KEY}/from_timestamp": pa.array([0.0], type=pa.float64()),
            f"videos/{VIDEO_KEY}/to_timestamp": pa.array(
                [length / float(manifest["fps"])], type=pa.float64()
            ),
        }
    )
    pq.write_table(episode, episodes_dir / "file-000.parquet", compression="zstd")

    video_info = {
        "video.fps": float(manifest["fps"]), "video.height": int(manifest["target_size"][0]),
        "video.width": int(manifest["target_size"][1]), "video.channel": 3,
        "video.codec": "h264", "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False, "has_audio": False,
    }
    features = {
        "state_mask": {"dtype": "bool", "shape": [2], "names": ["left", "right"]},
        "hand_kept": {"dtype": "bool", "shape": [2], "names": ["left", "right"]},
        "cam_trans": {"dtype": "float32", "shape": [3], "names": None},
        "cam_quat": {"dtype": "float32", "shape": [4], "names": ["x", "y", "z", "w"]},
        "cam_fov": {"dtype": "float32", "shape": [2], "names": ["height", "width"]},
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [int(manifest["target_size"][0]), int(manifest["target_size"][1]), 3],
            "names": ["height", "width", "channel"], "info": video_info,
        },
    }
    for side in ("left", "right"):
        for column, width in widths.items():
            features[f"{side}_mano_{column}"] = {
                "dtype": "float32", "shape": [width], "names": None,
            }
    for name in ("frame_index", "episode_index", "index", "task_index"):
        features[name] = {"dtype": "int64", "shape": [1], "names": None}
    features["timestamp"] = {"dtype": "float64", "shape": [1], "names": None}
    features["source_timestamp_ns"] = {"dtype": "int64", "shape": [1], "names": None}
    info = {
        "codebase_version": "v3.0", "robot_type": "das_ego_v6",
        "total_episodes": 1, "total_frames": length, "total_tasks": 1, "total_videos": 1,
        "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 500,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "fps": float(manifest["fps"]), "splits": {"train": "0:1"}, "features": features,
        "hand_frame": "camera", "label_kind": "pseudo_gt",
        "provenance": {
            "adapter": "mint.das_ego.v1", "dataset": manifest["dataset"],
            "master_mcap": manifest["master_mcap"],
            "vio_mcap": source_vio_mcap,
            "vio_alignment": str(Path(vio_npz).resolve()),
            "hawor_result": str(Path(hawor_npz).resolve()),
            "hawor_smoothing_sigma": smoothing_sigma,
            "hawor_valid_frames": {
                "left": int(hands["hand_kept"][:, 0].sum()),
                "right": int(hands["hand_kept"][:, 1].sum()),
            },
            "camera": manifest["camera"],
            "source_frame_start": manifest["source_frame_start"],
            "source_frame_end_exclusive": manifest["source_frame_end_exclusive"],
            "max_vio_alignment_error_ms": float(alignment_error.max(initial=0)) / 1e6,
            "calibration": manifest["calibration"],
        },
    }
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _write_camera_translation_normalization(output, cam_trans, cam_quat)
    validate_lerobot(output, expected_frames=length)
    return output


def validate_lerobot(dataset: Path, *, expected_frames: int | None = None) -> None:
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    length = int(info["total_frames"])
    if expected_frames is not None and length != expected_frames:
        raise RuntimeError(f"dataset has {length} frames, expected {expected_frames}")
    table = pq.read_table(dataset / "data" / "chunk-000" / "file-000.parquet")
    if table.num_rows != length:
        raise RuntimeError(f"Parquet has {table.num_rows} rows, expected {length}")
    required = {"state_mask", "hand_kept", "cam_trans", "cam_quat", "cam_fov"} | {
        f"{side}_mano_{column}"
        for side in ("left", "right") for column in MANO_COLUMNS
    }
    missing = required - set(table.column_names)
    if missing:
        raise RuntimeError(f"export is missing training columns {sorted(missing)}")
    kept = np.asarray(table["hand_kept"].to_pylist(), dtype=bool)
    if kept.shape != (length, 2):
        raise RuntimeError(f"hand_kept has shape {kept.shape}, expected {(length, 2)}")
    for side in ("left", "right"):
        for column in MANO_COLUMNS:
            values = np.asarray(table[f"{side}_mano_{column}"].to_pylist())
            if not np.isfinite(values).all():
                raise RuntimeError(f"{side}_{column} contains non-finite placeholders")
    probe = _probe_video(dataset / "videos" / VIDEO_KEY / "chunk-000" / "file-000.mp4")
    if probe["frames"] != length:
        raise RuntimeError(f"export video has {probe['frames']} frames, expected {length}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--master-mcap", type=Path, required=True)
    parser.add_argument("--camera", choices=[f"camera{i}" for i in range(6)], default="camera2")
    parser.add_argument("--start-sec", type=float, default=3.0)
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="clip duration in seconds; omit to process through the final frame",
    )
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--vfov", type=float, default=DEFAULT_FOV)
    parser.add_argument("--hfov", type=float, default=DEFAULT_FOV)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mint das-ego")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="rectify a DAS camera video")
    _add_common(prepare_parser)
    all_parser = subparsers.add_parser(
        "all", help="prepare, align DAS VIO, run HaWoR and export"
    )
    _add_common(all_parser)
    all_parser.add_argument("--vio-mcap", type=Path, default=None)
    all_parser.add_argument("--das-stack", type=Path, default=Path("../DAS/das-ego-stack"))
    all_parser.add_argument("--hawor-result", type=Path, default=None)
    all_parser.add_argument("--hawor-env", default="HaWoR")
    all_parser.add_argument("--hawor-root", type=Path, default=Path("../HaWoR"))
    all_parser.add_argument("--hawor-smooth-sigma", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepared_dir = prepare(
            args.dataset, args.master_mcap, args.output, camera=args.camera,
            start_sec=args.start_sec, duration_sec=args.duration_sec, size=args.size,
            vfov=args.vfov, hfov=args.hfov, overwrite=args.overwrite,
        )
        print(f"[das-ego] prepared: {prepared_dir}", flush=True)
        if args.subcommand == "prepare":
            return 0
        vio_mcap = (
            args.vio_mcap.expanduser().resolve()
            if args.vio_mcap is not None
            else run_das_pipeline(args.master_mcap, args.output / "das_pipeline", args.das_stack)
        )
        vio_npz = align_vio(prepared_dir, vio_mcap)
        print(f"[das-ego] VIO aligned: {vio_npz}", flush=True)
        hawor_npz = (
            args.hawor_result.expanduser().resolve()
            if args.hawor_result is not None
            else _run_hawor(
                prepared_dir, args.output / "hawor_camera_mano.npz",
                environment=args.hawor_env, hawor_root=args.hawor_root,
                smooth_sigma=args.hawor_smooth_sigma,
            )
        )
        lerobot = export_lerobot(
            prepared_dir, vio_npz, hawor_npz, args.output / "lerobot_v3",
            overwrite=args.overwrite,
        )
        print(f"[das-ego] LeRobot v3 pseudo-GT: {lerobot}", flush=True)
        return 0
    except Exception as error:
        print(f"[das-ego] error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
