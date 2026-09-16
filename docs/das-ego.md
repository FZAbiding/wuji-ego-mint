# DAS-Ego adapter

The DAS-Ego adapter rectifies a calibrated DAS camera into MINT's pinhole
input and can optionally combine DAS VIO with camera-frame HaWoR estimates to
produce a training-ready LeRobot v3 episode. HaWoR output is a pseudo-label,
not ground truth; the Viewer consequently labels it `Pseudo-GT`.

## Installation

Create the full MINT environment from the repository root:

```bash
bash scripts/create_env.sh full
conda activate mint
python -m mint doctor --profile full --strict
```

`prepare` requires FFmpeg with H.264 encoding support. The complete `all`
workflow additionally requires a matching master/`sub_left`/`sub_right` MCAP
triplet, the DAS delivery wheel and four v1.0.5 Docker images, plus a separately
installed HaWoR checkout, environment, and licensed weights. HaWoR assets are
not distributed by this repository.

## Prediction-only demo

Run the adapter from the repository root. The default camera is `camera2`.
This example selects source frames 90 through 539 (3–18 seconds at 30 fps):

```bash
python -m mint das-ego prepare \
  --dataset /path/to/DAS-Ego_UUID_lerobot \
  --master-mcap /path/to/DAS-Ego_UUID.mcap \
  --camera camera2 \
  --start-sec 3 \
  --duration-sec 15 \
  --output output/das_ego/UUID
```

The command reads the Double Sphere calibration and `T_b_c` from
`camera2/camera_info`, writes a 512×512 H.264/YUV420p video, and preserves the
selected absolute timestamps and calibration provenance. Omit
`--duration-sec` to process every remaining frame.

Open the generated directory and select `camera2_rectified.mp4`:

```bash
python -m mint viewer \
  --input output/das_ego/UUID/prepared \
  --ckpt checkpoints/model.safetensors \
  --fps 30 --max-frames 450 --window 32 --full-max-frames 32 \
  --devices 0 --mode mesh_skel --host 127.0.0.1
```

A plain MP4 is prediction-only. The Viewer keeps the GT column as an empty
placeholder, renders only PRED in the right column, and does not compute loss.
This is the recommended mode for DAS-Ego recordings without real ground truth.

## Optional VIO and HaWoR pseudo-label export

This section is optional and is not part of the no-GT prediction demo. `all`
runs preparation, the DAS `qc,merge,vio,vio_check` chain, timestamp
alignment, HaWoR, and LeRobot export:

```bash
python -m mint das-ego all \
  --dataset /path/to/DAS-Ego_UUID_lerobot \
  --master-mcap /path/to/DAS-Ego_UUID.mcap \
  --camera camera2 --start-sec 3 --duration-sec 15 \
  --das-stack /path/to/das-ego-stack \
  --hawor-root /path/to/HaWoR --hawor-env HaWoR \
  --hawor-smooth-sigma 1.0 \
  --output output/das_ego/UUID
```

The master, `sub_left`, and `sub_right` MCAPs must share the same trailing
UUID and be in one directory. The DAS stack wheel and four v1.0.5 Docker
images must be available locally. HaWoR and its licensed weights stay in a
separate environment. Existing VIO or HaWoR output can be supplied with
`--vio-mcap` or `--hawor-result`.

VIO is matched to each selected camera timestamp with a half-frame maximum
error. Export stops on a missing timestamp, calibration mismatch, invalid
rotation, or failed VIO check. Missing hand detections remain finite identity
placeholders with `hand_kept=false`; they are not used in MANO loss. Temporal
smoothing is bounded to each consecutive run of real detections, so values are
never propagated across a missing-detection gap.

Open the completed comparison dataset with:

```bash
python -m mint viewer \
  --input output/das_ego/UUID/lerobot_v3 \
  --ckpt checkpoints/model.safetensors \
  --max-frames 450 --window 32 --full-max-frames 32 \
  --devices 0 --mode mesh_skel --host 127.0.0.1
```

The exported `meta/info.json` records `label_kind: pseudo_gt`,
`hand_frame: camera`, source frames, calibration, DAS VIO, alignment, and
HaWoR provenance.

## Outputs and data contract

The main outputs are:

```text
output/das_ego/UUID/prepared/camera2_rectified.mp4
output/das_ego/UUID/prepared/manifest.json
output/das_ego/UUID/vio_aligned.npz
output/das_ego/UUID/hawor_camera_mano.npz
output/das_ego/UUID/lerobot_v3
```

The LeRobot episode contains `observation.images.ego`, `cam_trans`,
`cam_quat`, `cam_fov`, `hand_kept`, and each hand's `mano_transl_cam`,
`mano_orient6d`, `mano_pose6d`, and `mano_betas`. The adapter also writes
camera-translation normalization statistics used by Viewer camera losses.
All loss values for this dataset measure agreement with DAS VIO/HaWoR
pseudo-labels; they are not absolute accuracy against motion-capture ground
truth.

## Troubleshooting and constraints

- A missing calibration topic means the master MCAP must be checked for
  `/robot0/sensor/<camera>/camera_info` and a Double Sphere model.
- An encoding failure usually means `ffmpeg` is absent from `PATH` or lacks
  `libx264` support.
- Triplet discovery requires all three MCAPs in one directory with the same
  trailing eight-character UUID and the expected master/sub names.
- A VIO alignment failure indicates inconsistent absolute clocks or an error
  above half a frame; the adapter does not silently interpolate it.
- HaWoR launch failures should be checked against the `mamba` executable,
  environment name, source root, and checkpoint path.
- Existing output is preserved by default; pass `--overwrite` only after
  confirming the destination.
