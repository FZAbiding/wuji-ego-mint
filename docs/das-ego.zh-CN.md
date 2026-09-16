# DAS-Ego 适配器

DAS-Ego 适配器把已标定的 DAS 相机视频校正为 MINT 可直接使用的针孔视频；可选的完整流程还会将 DAS VIO 与相机坐标系 HaWoR 结果对齐并导出为 LeRobot v3。HaWoR 输出属于伪标签，不是真值，因此 Viewer 会将它标为 `Pseudo-GT`。

## 安装

从仓库根目录创建并启用完整环境：

```bash
bash scripts/create_env.sh full
conda activate mint
python -m mint doctor --profile full --strict
```

`prepare` 需要 FFmpeg（支持 H.264 编码）以及项目运行时依赖。完整的 `all` 流程还需要：

- 相互匹配的 master、`sub_left` 与 `sub_right` MCAP；
- DAS stack 的 `delivery_pipeline-1.0.4-py3-none-any.whl` 和本地已有的四个 v1.0.5 Docker 镜像；
- 独立安装的 HaWoR 环境、源码与已授权权重。HaWoR 受其上游许可证约束，不随本仓库分发。

## 无真值预测（推荐）

默认使用 `camera2`。下面的命令从第 3 秒开始处理 15 秒；省略 `--duration-sec` 会一直处理到末帧：

```bash
python -m mint das-ego prepare \
  --dataset /path/to/DAS-Ego_UUID_lerobot \
  --master-mcap /path/to/DAS-Ego_UUID.mcap \
  --camera camera2 \
  --start-sec 3 \
  --duration-sec 15 \
  --output output/das_ego/UUID
```

命令会从 `camera2/camera_info` 读取 Double Sphere 参数与 `T_b_c`，生成 512×512、H.264/YUV420p 的针孔视频，并保留绝对时间戳、目标内参、标定来源与源帧范围。打开输出目录并选择校正后的视频：

```bash
python -m mint viewer \
  --input output/das_ego/UUID/prepared \
  --ckpt checkpoints/model.safetensors \
  --fps 30 --max-frames 450 --window 32 --full-max-frames 32 \
  --devices 0 --mode mesh_skel --host 127.0.0.1
```

普通 MP4 没有真实标签。Viewer 会保留空白参考栏、在右侧只显示 PRED，并且不计算 loss。这是没有真实真值的 DAS-Ego 记录的推荐用法。

## 可选的 VIO 与 HaWoR 伪标签

这一步不是无真值预测演示的必要部分。`all` 会依次完成视频准备、DAS `qc,merge,vio,vio_check`、时间戳对齐、HaWoR 和 LeRobot v3 导出：

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

master、`sub_left` 与 `sub_right` MCAP 必须位于同一目录并带相同的末尾 UUID。也可以通过 `--vio-mcap` 或 `--hawor-result` 复用现有结果。VIO 与每个相机时间戳按半帧最大误差匹配；时间戳缺失、标定不一致、旋转无效或 VIO 检查失败都会中止导出。

HaWoR 没有检测到手的帧会写入有限的单位占位值并设置 `hand_kept=false`，这些帧不参与 MANO loss。时序平滑只在连续真实检测段内部进行，不会跨检测缺口传播参数，也不会用 MINT 预测回填伪标签。

打开导出的伪标签对比数据集：

```bash
python -m mint viewer \
  --input output/das_ego/UUID/lerobot_v3 \
  --ckpt checkpoints/model.safetensors \
  --max-frames 450 --window 32 --full-max-frames 32 \
  --devices 0 --mode mesh_skel --host 127.0.0.1
```

Viewer 会显示 `Pseudo-GT`，而不是将这些数据误称为 `GT`。

## 输入、输出与数据约定

主要输出如下：

```text
output/das_ego/UUID/prepared/camera2_rectified.mp4
output/das_ego/UUID/prepared/manifest.json
output/das_ego/UUID/vio_aligned.npz
output/das_ego/UUID/hawor_camera_mano.npz
output/das_ego/UUID/lerobot_v3
```

最终 LeRobot v3 数据包含 `observation.images.ego`、`cam_trans`、`cam_quat`、`cam_fov`、`hand_kept`，以及左右手的 `mano_transl_cam`、`mano_orient6d`、`mano_pose6d` 和 `mano_betas`。

`meta/info.json` 写入 `hand_frame: camera`、`label_kind: pseudo_gt`，以及源数据、相机标定、DAS VIO、时间对齐、HaWoR 平滑和有效检测数等来源信息。适配器还会写入 Viewer camera loss 使用的相机平移归一化统计。

## 故障排查与约束

- 找不到标定主题：确认 master MCAP 包含所选相机的 `/robot0/sensor/<camera>/camera_info`，且模型为 Double Sphere。
- 无法编码输出：确认 `ffmpeg` 在 `PATH` 中并支持 `libx264`。
- 找不到 DAS 三件套：确认文件位于同一目录、文件名包含 `master`/`sub_left`/`sub_right`，并具有相同的 8 位末尾 UUID。
- VIO 对齐失败：检查 MCAP 的绝对时钟来源；适配器不会对超出半帧的轨迹做静默插值。
- HaWoR 启动失败：确认 `mamba`、环境名、源码根目录与 checkpoint 路径有效。
- 已存在输出默认不会覆盖；确认目标后显式传入 `--overwrite`。

## 实现与验证附录

### 数据转换

原始 LeRobot 数据包含六路 1600×1300 Double Sphere 鱼眼视频和 IMU，但缺少 MINT 所需的 `observation.images.ego`、相机轨迹、MANO 参数及左右手有效性标签，因此新增了 DAS-Ego 专用适配器。

该数据没有真实 GT。正式展示采用 prediction-only：GT 栏保留为空，右侧只显示 PRED，不计算 loss。DAS VIO 和 HaWoR 仅作为可选伪标签分析产物，不再占用 GT 槽位。

处理流程：

```text
LeRobot + master/sub_left/sub_right MCAP
→ camera2 去畸变
→ DAS qc/merge/vio/vio_check
→ VIO 时间对齐
→ HaWoR MANO 伪标签
→ LeRobot v3
→ MINT 推理与 Viewer
```

#### 相机预处理

- 从 master MCAP 的 `/robot0/sensor/camera2/camera_info` 读取 Double Sphere 参数 `[fx, fy, cx, cy, xi, alpha]` 和 `T_b_c`。
- 将 camera2 映射成 512×512 针孔视频，水平/垂直 FoV 均为 1.7157 rad。
- 截取源帧 90–539，即 3–18 秒，共 450 帧、30 FPS。
- 输出 H.264/YUV420p，并保存绝对时间戳、目标内参、标定来源和源帧范围。
- 中心映射误差为 0.00002675 px。

#### DAS VIO

对 UUID `43082562` 的三件套运行 `qc → merge → vio → vio_check`，从 `/robot0/vio/eef_pose` 读取机身轨迹，并按绝对时间戳对齐 camera2。最大误差为 0.291819 ms。

坐标转换：

```text
T_w_c(t) = T_w_b(t) · T_b_c
T_c0_c(t) = inverse(T_w_c(0)) · T_w_c(t)
```

以首帧相机为世界原点，导出 MINT 使用的 `cam_trans [T,3]`、`cam_quat [T,4]` 和 `cam_fov [T,2]`。同时验证首帧单位位姿、旋转正交性、行列式、四元数归一化和连续性。

#### HaWoR 伪标签

在独立 HaWoR 环境中运行手部检测和 MANO 推理，输出左右手：

```text
mano_transl_cam [T,3]
mano_orient6d   [T,6]
mano_pose6d     [T,90]
mano_betas      [T,10]
```

只保留真实 detector 命中的帧；缺失帧设置 `hand_kept=false` 并写有限占位值；只在连续有效检测段内平滑，不跨缺口插值，也不使用 MINT 预测补标签。有效检测为左手 29/450、右手 76/450。

#### 最终 LeRobot v3

导出的主要字段为：

```text
observation.images.ego
cam_trans, cam_quat, cam_fov
hand_kept
left/right_mano_transl_cam
left/right_mano_orient6d
left/right_mano_pose6d
left/right_mano_betas
```

`meta/info.json` 写入 `hand_frame: camera`、`label_kind: pseudo_gt` 及完整来源信息，因此 Viewer 显示 `Pseudo-GT` 而非 `GT`。另生成相机平移归一化统计，供训练和 Viewer loss 使用。

### MINT 推理

使用 RTX 4090 对 450 帧完成推理，参数为 `window=32`、`full-max-frames=32`、`max_chunked`、`smooth` 和 `mesh_skel`。共 19 个推理窗口，总 forward 约 30.56 秒，平均约 67.9 ms/帧。

### Viewer 功能

#### Combined · 2D

将 Pseudo-GT 和 PRED 的 MANO mesh/骨架分别使用各自内参投影到 ego RGB，支持左右并排和叠加。本次已生成 450 帧完整对比视频。

#### Fixed World · 3D

将 HaWoR/MINT 的相机系手部通过各自相机轨迹转换到固定世界，显示左右手轨迹、当前手部、相机轨迹、相机姿态、坐标轴和 Z-up 地面。本次已生成 450 帧完整视频。

#### Current Camera · 3D

以当前相机光心为原点显示左右手，便于检查深度、左右翻转、尺度、手腕朝向，以及区分手部局部误差和 VIO 长期漂移。

#### MuJoCo · Simulation

将 MANO mesh、相机姿态和世界运动放入 MuJoCo 场景，使用固定第三人称相机和参考地面。这是几何仿真可视化，不代表从 Finger MCAP 恢复了真实接触力。面板按需启动。

#### Wuji Hand · Retargeting

把 MANO 21 点通过左右手 retarget 配置转换成 Wuji 机器人手 qpos，再在 MJCF 场景渲染。Wuji 与 MuJoCo 共用世界坐标、观察相机和地面口径。面板按需启动。

#### Per-Frame Values (World)

逐帧对比 Pseudo-GT/PRED 的相机世界位置、欧拉角、FoV，以及左右手世界系手腕位置、朝向和 MANO betas；同时显示整段平均 FoV 和 betas。

#### Per-Frame Values (Camera)

逐帧对比相机坐标系左右手腕位置、朝向和 betas。无效 Pseudo-GT 手部帧显示为空，不展示占位参数。

#### Per-Frame Loss

使用 checkpoint 的训练配置和原始 Criterion 计算。450 帧按 `clip_len=32`、`stride=1` 形成 419 个训练窗口，每个窗口把相机轨迹重新锚定到首帧。

根据模型配置和标签可用性显示 Camera、FoV、手部存在性、2D 重投影、MANO 参数、MANO 21 点及 Camera–MANO consistency loss，包括当前值、权重、加权贡献、占比、均值和总 loss。

`hand_kept` 参与掩码：无效手部帧不进入 MANO loss，速度项只在相邻帧同一只手都有效时计算。所有指标表示相对 DAS VIO/HaWoR 伪标签的一致性，不是相对真实 MoCap GT 的绝对精度。

### 输出与验证

```text
output/das_ego/43082562/prepared/camera2_rectified.mp4
output/das_ego/43082562/vio_aligned.npz
output/das_ego/43082562/hawor_camera_mano.npz
output/das_ego/43082562/lerobot_v3
output/das_ego/43082562/pseudo_gt_vs_pred_2d.mp4
output/das_ego/43082562/pseudo_gt_vs_pred_world.mp4
```

- 数据加载器成功读取完整 32 帧窗口。
- 2D 视频：1030×512、450 帧、30 FPS、15 秒。
- Fixed World 视频：960×540、450 帧、30 FPS、15 秒。
- 逐帧指标计算无报错。
- DAS-Ego 适配器 9 项测试通过。

主要实现文件：`mint/das_ego.py`、`mint/das_ego_hawor.py`、`ego_pipeline/tasks/result_to_lerobot.py` 和 Viewer 的 LeRobot/渲染模块。新增入口为 `python -m mint das-ego prepare` 与 `python -m mint das-ego all`。
