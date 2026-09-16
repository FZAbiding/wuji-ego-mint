#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval/lingbotmap 自包含 lerobot v3 读取：定位数据集、枚举 episode、读预算列 + 抽帧。

只支持训练就绪 export（scripts/data_processed/pipeline_lerobot/build_train_lerobot.py 产物）：相机/手部
label 已离线预算为分开列。

LeRobot v3 打包规则（见 wuji-vitra/tools/aggregate_lerobot_v3.py）：data parquet 与 video mp4
按各自 size 预算独立滚动，故「data 文件号 ≠ video 文件号」，且 episode 在 mp4 里的位置由
meta/episodes 的 from_timestamp（秒）决定，**不是** data parquet 的行号。因此本模块以
meta/episodes/**/*.parquet 为准：数据文件走 data/{chunk,file}，视频文件走
videos/{key}/{chunk,file}，抽帧起点 = round(from_timestamp * fps)，data 行按 episode_index
在数据文件内命中。老布局（无 meta/episodes）回退到旧的「行号=帧号」扫描。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from . import geometry as geom

_CHUNK_FILE_RE = re.compile(r"chunk-(\d+)/file-(\d+)\.parquet$")
_MANO_COLS = ["transl_cam", "orient6d", "pose6d", "betas"]
# observation.state 切片：左[51:61] / 右[112:122]（betas，坐标系无关），与训练端一致。
_BETAS_SLICE = {"left": (51, 61), "right": (112, 122)}
# 抽帧解码（env 可调）：
#   VIEWER_DECORD_THREADS —— 单个 VideoReader 的解码线程数（默认 min(8,cpu)，本机 H20 节点实测甜点）。
#   VIEWER_DECODE_CHUNK   —— 分批读的每批帧数（默认 96），越小进度越平滑、开销略增。
# 用**单 reader**（整段 mp4 只索引一次）+ 分批 get_batch：多 reader 并行会各自重索引整段 v3 mp4
# （一个 mp4 打包多条 episode，很长），反而更卡；单 reader 多线程解码对这里够快。
_DECORD_THREADS = int(os.environ.get("VIEWER_DECORD_THREADS") or min(8, (os.cpu_count() or 4)))
_DECODE_CHUNK = int(os.environ.get("VIEWER_DECODE_CHUNK") or 96)
# 26,878 个 5-12 KiB 小 Parquet 的实测峰值在 4 个外层线程；单文件内层线程必须关闭，
# 否则外层线程与 Arrow 线程池叠加后反而更慢。环境变量保留给不同存储机器重新调优。
_EPISODE_READ_WORKERS = max(1, int(os.environ.get("VIEWER_EPISODE_READ_WORKERS") or 4))
_EPISODE_STAT_WORKERS = max(1, int(os.environ.get("VIEWER_EPISODE_STAT_WORKERS") or 32))
_EPISODE_CACHE_VERSION = 1
_REPO_DIR = Path(__file__).resolve().parents[4]
_EPISODE_CACHE_DIR = Path(
    os.environ.get("VIEWER_EPISODE_CACHE_DIR")
    or (_REPO_DIR / "output" / "cache" / "viewer_episode_index")
)


def _raise_if_cancelled(cancel_check, stage: str = "数据加载") -> None:
    if cancel_check is not None and cancel_check():
        from inference.base import InferenceCancelled
        raise InferenceCancelled(f"{stage}已取消")


def _episode_manifest_signature(ds_dir: Path, ep_files: list[Path], info_path: Path,
                                video_key: str) -> dict:
    """Build a cheap immutable-export signature without statting every Parquet."""
    names = hashlib.sha256()
    episodes_dir = ds_dir / "meta" / "episodes"
    for path in ep_files:
        names.update(path.relative_to(episodes_dir).as_posix().encode("utf-8"))
        names.update(b"\0")
    return {
        "version": _EPISODE_CACHE_VERSION,
        "dataset_root": str(ds_dir.resolve()),
        "video_key": video_key,
        "info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
        "episode_file_count": len(ep_files),
        "episode_filename_sha256": names.hexdigest(),
    }


def _episode_cache_path(signature: dict) -> Path:
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    key = hashlib.sha256(encoded).hexdigest()[:24]
    name = Path(signature["dataset_root"]).parent.name
    safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_") or "dataset"
    return _EPISODE_CACHE_DIR / f"{safe_name}.{key}.parquet"


def _load_episode_cache(path: Path, signature: dict, columns: list[str]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        schema = pq.read_schema(path)
        metadata = schema.metadata or {}
        cached_signature = json.loads(metadata[b"viewer_episode_index"].decode("utf-8"))
        if cached_signature != signature or not set(columns) <= set(schema.names):
            return None
        return pq.ParquetFile(path).read(columns=columns, use_threads=False)
    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, pa.ArrowInvalid):
        return None


def _write_episode_cache(path: Path, table, signature: dict) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    metadata = dict(table.schema.metadata or {})
    metadata[b"viewer_episode_index"] = json.dumps(
        signature, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    try:
        pq.write_table(
            table.replace_schema_metadata(metadata), temporary,
            compression="zstd", use_dictionary=False,
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_frames_chunked(vr, indices, on_step=None, cancel_check=None) -> np.ndarray:
    """单 reader 按 _DECODE_CHUNK 分批 get_batch 抽帧、逐批可选回调进度(done,total)，末尾 cat。

    分批读既能给前端**平滑**的确定进度，又保留每批批量解码效率；只索引一次。
    cancel_check()->bool 在每批解码前后检查；单个在途 get_batch 无法强制中断。
    on_step/cancel_check 均为空（如离线 hand_reproj）时一次读完。"""
    indices = list(indices)
    T = len(indices)
    _raise_if_cancelled(cancel_check, "视频解码")
    if T == 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    if (on_step is None and cancel_check is None) or T <= _DECODE_CHUNK:
        arr = vr.get_batch(indices).asnumpy()
        _raise_if_cancelled(cancel_check, "视频解码")
        if on_step:
            try: on_step(T, T)
            except Exception: pass   # noqa: BLE001  进度回调不影响读帧
        return arr
    parts, done = [], 0
    for i in range(0, T, _DECODE_CHUNK):
        _raise_if_cancelled(cancel_check, "视频解码")
        seg = indices[i:i + _DECODE_CHUNK]
        parts.append(vr.get_batch(seg).asnumpy())
        _raise_if_cancelled(cancel_check, "视频解码")
        done += len(seg)
        if on_step is not None:
            try: on_step(done, T)
            except Exception: pass   # noqa: BLE001
    _raise_if_cancelled(cancel_check, "视频解码")
    return np.concatenate(parts, axis=0)


def find_dataset(root: Path) -> Path | None:
    """在 root 下定位 lerobot v3 数据集（含 meta/info.json）。"""
    root = Path(root)
    if (root / 'meta' / 'info.json').exists():
        return root
    if (root / 'lerobot_v3' / 'meta' / 'info.json').exists():
        return root / 'lerobot_v3'
    cands = sorted(
        {p.parent.parent for p in root.rglob('info.json') if p.parent.name == 'meta'},
        key=lambda d: (len(d.relative_to(root).parts), str(d)),
    )
    return cands[0] if cands else None


def video_hw(ds_dir: Path) -> tuple[int, int]:
    """从 info.json 读 ego 视频 (height, width)。"""
    info = json.loads((Path(ds_dir) / 'meta' / 'info.json').read_text(encoding='utf-8'))
    ego = info["features"]["observation.images.ego"]["shape"]   # [H, W, 3]
    return int(ego[0]), int(ego[1])


def discover_episodes(ds_dir: Path, video_key: str = "observation.images.ego",
                      on_step=None, require_video: bool = True) -> list[dict]:
    """以 meta/episodes/**/*.parquet 为准枚举 episode，返回按 episode_index 升序的列表。

    每项: {episode_index, length, parquet, video, video_start, dataset_root}
      · parquet     : 该 episode 的 data 文件（data/{chunk,file}），行按 episode_index 命中；
      · video       : 该 episode 的 mp4（videos/{key}/{chunk,file}），与 data 文件号无关；
      · video_start : mp4 内抽帧起始帧 = round(from_timestamp * fps)，取连续 length 帧。

    on_step(stage, done, total)：可选进度回调，stage∈{read,resolve,build}，供上位机异步枚举时显示进度条。
    meta/episodes 缺失（老布局）时回退到旧的「行号=帧号」扫描 _discover_episodes_legacy。
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq

    def _step(stage, done, total):        # 进度回调不抛（回调异常不该拖垮枚举）
        if on_step:
            try:
                on_step(stage, int(done), int(total))
            except Exception:  # noqa: BLE001
                pass

    ds_dir = Path(ds_dir)
    ep_files = sorted((ds_dir / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        print("[lerobot_io] 未发现 meta/episodes，回退旧扫描（行号=帧号）", flush=True)
        return _discover_episodes_legacy(ds_dir, video_key)

    info_path = ds_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps") or 30.0)
    data_tmpl = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    video_tmpl = info.get(
        "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    vck, vfk, vtk = (f"videos/{video_key}/chunk_index", f"videos/{video_key}/file_index",
                     f"videos/{video_key}/from_timestamp")
    need_cols = ["episode_index", "length", "data/chunk_index", "data/file_index", vck, vfk, vtk]

    t0 = time.time()
    signature = _episode_manifest_signature(ds_dir, ep_files, info_path, video_key)
    cache_path = _episode_cache_path(signature)
    _step("read", 0, len(ep_files))
    t = _load_episode_cache(cache_path, signature, need_cols)
    if t is not None:
        print(
            f"[lerobot_io] 索引缓存命中：{len(ep_files)} 个 meta/episodes <- {cache_path}",
            flush=True,
        )
        _step("read", len(ep_files), len(ep_files))
    else:
        schema_names = set(pq.read_schema(ep_files[0]).names)
        missing = set(need_cols) - schema_names
        if missing:
            raise RuntimeError(
                f"{ds_dir}/meta/episodes 缺少列 {sorted(missing)}；"
                f"video_key={video_key!r} 是否正确？"
            )
        workers = min(_EPISODE_READ_WORKERS, len(ep_files))
        print(
            f"[lerobot_io] 外层并发({workers})读取 {len(ep_files)} 个 meta/episodes parquet"
            "（单文件内层线程关闭）…",
            flush=True,
        )

        def _read_one(path):
            return pq.ParquetFile(path).read(columns=need_cols, use_threads=False)

        # 按文件名顺序合并，确保 cache miss/hit 与旧实现的 episode 顺序完全一致。
        tabs = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, table in enumerate(ex.map(_read_one, ep_files), 1):
                tabs.append(table)
                if i % 4 == 0 or i == len(ep_files):
                    _step("read", i, len(ep_files))
        t = pa.concat_tables(tabs)
        _write_episode_cache(cache_path, t, signature)
        print(f"[lerobot_io] 已写 episode 索引缓存：{cache_path}", flush=True)

    ei = t.column("episode_index").to_numpy()
    ln = t.column("length").to_numpy()
    dci, dfi = t.column("data/chunk_index").to_numpy(), t.column("data/file_index").to_numpy()
    vci, vfi = t.column(vck).to_numpy(), t.column(vfk).to_numpy()
    fts = t.column(vtk).to_numpy()

    # 文件存在性去重：数十万 episode 只对应 ~万个 data/video 文件，每个文件只 stat 一次（并发）。
    def _resolve_exist(items, tmpl, is_video):
        out: dict[tuple, str | None] = {}
        uniq = {(int(c), int(f)) for c, f in items}

        def _one(cf):
            p = ds_dir / (tmpl.format(video_key=video_key, chunk_index=cf[0], file_index=cf[1])
                          if is_video else tmpl.format(chunk_index=cf[0], file_index=cf[1]))
            return cf, (str(p) if p.exists() else None)
        stat_workers = min(_EPISODE_STAT_WORKERS, max(1, len(uniq)))
        with ThreadPoolExecutor(max_workers=stat_workers) as ex:
            for cf, path in ex.map(_one, uniq):
                out[cf] = path
        return out

    _step("resolve", 0, 1)                # 校验 data/video 文件存在（去重后并发 stat，通常很快）
    data_paths = _resolve_exist(zip(dci, dfi), data_tmpl, is_video=False)
    video_paths = _resolve_exist(zip(vci, vfi), video_tmpl, is_video=True)
    _step("resolve", 1, 1)

    n_ep = len(ei)
    _step("build", 0, n_ep)
    eps: list[dict] = []
    n_skip = 0
    dataset_root = str(ds_dir.resolve())
    hand_frame = info.get("hand_frame", "world")
    label_kind = str(info.get("label_kind", "ground_truth")).strip().lower()
    truth_label = "Pseudo-GT" if label_kind == "pseudo_gt" else "GT"
    for k in range(n_ep):
        if (k & 0x3FFF) == 0:             # 每 ~16k 条上报一次（大数据集汇总也占时间）
            _step("build", k, n_ep)
        dp = data_paths.get((int(dci[k]), int(dfi[k])))
        vp = video_paths.get((int(vci[k]), int(vfi[k])))
        # require_video=False：视频缺失/软链断掉也保留该 episode（GT 世界系渲染只用 parquet 里的
        # 手部/相机 label，不需要图像帧）。video 置 None，由 load_episode_raw 的
        # allow_missing_video 决定怎么兜底。
        if dp is None or (vp is None and require_video):
            n_skip += 1
            continue
        eps.append({
            "episode_index": int(ei[k]),
            "length": int(ln[k]),
            "parquet": dp,
            "video": vp,
            "video_start": int(round(float(fts[k]) * fps)),
            "dataset_root": dataset_root,
            "hand_frame": hand_frame,   # 手部落盘坐标系:camera(新)/world(旧)
            "label_kind": label_kind,
            "truth_label": truth_label,
        })
    eps.sort(key=lambda e: e["episode_index"])
    _step("build", n_ep, n_ep)
    print(f"[lerobot_io] 枚举完成：{len(eps)} episode"
          f"{f'（跳过 {n_skip} 个缺文件）' if n_skip else ''}，用时 {time.time() - t0:.1f}s", flush=True)
    return eps


def _discover_episodes_legacy(ds_dir: Path, video_key: str = "observation.images.ego") -> list[dict]:
    """旧扫描：扫 data/chunk-*/file-*.parquet，按 episode_index 连续段聚合。

    仅在数据集没有 meta/episodes 时回退使用；假设「data 第 j 行 ↔ 同名 mp4 第 j 帧」，
    对真·v3（data/video 独立滚动）并不成立，故只作兜底。
    每项: {episode_index, parquet, video, start, length}（start 为文件内绝对行号 = 帧号）。
    """
    import time

    import pyarrow.parquet as pq
    ds_dir = Path(ds_dir)
    eps: list[dict] = []
    files = sorted(glob.glob(str(ds_dir / "data" / "chunk-*" / "file-*.parquet")))
    t0 = time.time()
    print(f"[lerobot_io] 扫描 {len(files)} 个 data parquet 枚举 episode …", flush=True)
    for k, pf in enumerate(files):
        m = _CHUNK_FILE_RE.search(pf.replace(os.sep, "/"))
        if not m:
            continue
        ci, fi = int(m.group(1)), int(m.group(2))
        vpath = ds_dir / "videos" / video_key / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
        if not vpath.exists():
            continue
        ep = pq.read_table(pf, columns=["episode_index"]).column("episode_index").to_numpy()
        n = len(ep)
        if n == 0:
            continue
        # 向量化切段：episode_index 连续相等即一段，边界=值变化处（避免逐行 Python 循环）。
        bnd = np.flatnonzero(ep[1:] != ep[:-1]) + 1
        starts = np.concatenate(([0], bnd))
        ends = np.concatenate((bnd, [n]))
        vp = str(vpath)
        for s, e in zip(starts.tolist(), ends.tolist()):
            eps.append({"episode_index": int(ep[s]), "parquet": pf,
                        "video": vp, "start": s, "length": e - s,
                        "dataset_root": str(ds_dir.resolve())})
        if (k + 1) % 200 == 0 or k + 1 == len(files):
            print(f"[lerobot_io]   {k + 1}/{len(files)} 文件，累计 {len(eps)} episode "
                  f"（{time.time() - t0:.1f}s）", flush=True)
    eps.sort(key=lambda e: e["episode_index"])
    print(f"[lerobot_io] 枚举完成：{len(eps)} episode，用时 {time.time() - t0:.1f}s", flush=True)
    return eps


def _video_hw_from_info(dataset_root, video_key: str = "observation.images.ego") -> tuple[int, int]:
    """从 meta/info.json 的 feature shape 读 (H, W)——视频不可用时反解 K 仍需要分辨率。"""
    if not dataset_root:
        raise RuntimeError("allow_missing_video 需要 ep['dataset_root'] 才能从 meta/info.json 取分辨率。")
    info = json.loads((Path(dataset_root) / "meta" / "info.json").read_text())
    shape = (info.get("features", {}).get(video_key, {}) or {}).get("shape")
    if not shape or len(shape) < 2:
        raise RuntimeError(f"meta/info.json 的 features[{video_key}].shape 缺失，无法在无视频时确定分辨率。")
    return int(shape[0]), int(shape[1])


def load_episode_raw(ep: dict, max_frames: int | None = None, on_step=None,
                     allow_missing_video: bool = False, cancel_check=None) -> dict:
    """读单个 episode 的 GT + 抽帧，自动识别两种 export，统一输出 6D + c2w/K。

    · 预算集（build_train_lerobot 产物）：cam_trans/quat/fov、hand_kept，以及可选的
      {side}_mano_{transl_cam,orient6d,pose6d,betas} / {side}_kpt21；
    · 裸集（原始 export）：extrinsics_w2c/fov、{side}_transl_world、{side}_orient_world(3x3)、{side}_hand_pose(15x3x3)、
      {side}_kept，betas 从 observation.state 切片。
    两者旋转统一转 6D、相机统一转 (cam_c2w, K)（K 按显示帧分辨率，与预测一致）。

    cancel_check()->bool：在 Parquet、视频分块和列转换边界检查取消。
    返回: frames (T,H0,W0,3 uint8 RGB)、cam_c2w (T,4,4)、K (3,3)、kept (T,2 bool)、
          hands={'left'|'right': {transl_cam(T,3), orient6d(T,6), pose6d(T,90), betas(T,10)}}（key 名义 cam;裸集腿值实为 world,坐标系以 hand_frame 为准）。
    """
    import pyarrow.parquet as pq
    from decord import VideoReader

    _raise_if_cancelled(cancel_check)
    length = ep["length"]
    T = length if max_frames is None else min(length, int(max_frames))

    names = set(pq.read_schema(ep["parquet"]).names)
    is_budget = {"cam_trans", "cam_quat", "cam_fov"} <= names

    has_mano_gt = all(
        f"{side}_mano_{column}" in names
        for side in ("left", "right") for column in _MANO_COLS
    )
    has_kpt21_gt = {"left_kpt21", "right_kpt21"} <= names
    if is_budget:
        cols = ["hand_kept", "cam_trans", "cam_quat", "cam_fov"]
        if has_mano_gt:
            cols += [f"{side}_mano_{column}" for side in ("left", "right")
                     for column in _MANO_COLS]
        if has_kpt21_gt:
            cols += ["left_kpt21", "right_kpt21"]
    else:
        need = ["extrinsics_w2c", "fov"] + \
               [f"{s}_{c}" for s in ("left", "right")
                for c in ("transl_world", "orient_world", "hand_pose", "kept")]
        cols = need + (["observation.state"] if "observation.state" in names else []) + \
               (["state_mask"] if "state_mask" in names else [])
    missing = [c for c in cols if c not in names]
    if missing:
        raise RuntimeError(f"{ep['parquet']} 缺少列 {missing}（既非预算集也非可识别的裸集格式）。")

    # data 行定位：新布局直接让 Arrow 按 episode_index 过滤，避免整文件转 Python 后再筛；
    # 老布局（legacy 扫描）仍用文件内绝对行号 start。
    legacy = "start" in ep
    read_cols = cols if (legacy or "episode_index" in cols) else cols + ["episode_index"]
    vstart = int(ep.get("video_start", ep.get("start", 0)))
    vframes = list(range(vstart, vstart + T))

    def _read_table():
        _raise_if_cancelled(cancel_check, "Parquet 读取")
        if legacy:
            table = pq.read_table(ep["parquet"], columns=read_cols)
            rows = np.arange(ep["start"], ep["start"] + T)
        else:
            table = pq.read_table(
                ep["parquet"], columns=read_cols,
                filters=[("episode_index", "=", int(ep["episode_index"]))],
            )
            rows = np.arange(min(T, table.num_rows))
            if len(rows) < T:
                raise RuntimeError(
                    f"{ep['parquet']} 内 episode {ep['episode_index']} 仅 {len(rows)} 行 < 需要 {T}。"
                )
        _raise_if_cancelled(cancel_check, "Parquet 读取")
        return table, rows

    def _read_video():
        _raise_if_cancelled(cancel_check, "视频解码")
        # allow_missing_video：视频缺失/软链断时跳过抽帧（GT 世界系渲染只用 parquet 里的 label，
        # 不需要图像帧；但反解 K 要分辨率，改从 meta/info.json 的 feature shape 取）。
        if allow_missing_video and not (ep.get("video") and os.path.exists(ep["video"])):
            return None
        vr = VideoReader(ep["video"], num_threads=_DECORD_THREADS)
        if T > 0 and vframes[-1] >= len(vr):
            raise RuntimeError(
                f"视频 {ep['video']} 仅 {len(vr)} 帧，取不到帧 {vframes[0]}..{vframes[-1]}"
                f"（episode {ep['episode_index']}；疑似视频截断或 video_key 与数据不匹配）。"
            )
        return _read_frames_chunked(
            vr, vframes, on_step=on_step, cancel_check=cancel_check
        )

    # Parquet 和 H.264 位于不同文件，二者并行可隐藏小表读取/Arrow 解包的等待。
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="episode-load") as pool:
        table_future = pool.submit(_read_table)
        video_future = pool.submit(_read_video)
        t, data_rows = table_future.result()
        frames = video_future.result()   # (T,H0,W0,3) uint8 RGB，分批读带进度
    _raise_if_cancelled(cancel_check)

    def _col(name: str) -> np.ndarray:
        _raise_if_cancelled(cancel_check, "Parquet 列转换")
        values = np.asarray(t.column(name).combine_chunks().to_numpy(zero_copy_only=False))
        if values.dtype == object and len(values):
            try:
                values = np.stack(values)
            except ValueError:
                values = np.asarray(values.tolist())
        values = values[data_rows]
        _raise_if_cancelled(cancel_check, "Parquet 列转换")
        return values

    # frames 为 None（allow_missing_video 且视频不可用）时，分辨率退回 meta/info.json 的 feature shape。
    H, W = (_video_hw_from_info(ep.get("dataset_root")) if frames is None
            else frames.shape[1:3])

    hands = {}
    kpt21_gt = None
    cam_pose_enc = None   # 仅预算集有(逐帧相机 loss 用);裸集为 None → 网页端跳过误差面板
    if is_budget:
        cam_pose_enc = np.concatenate([_col("cam_trans"), _col("cam_quat"), _col("cam_fov")], axis=-1).astype(np.float32)
        cam_c2w, K = geom.decode_camera_pose_enc(cam_pose_enc, H, W)
        kept = _col("hand_kept").astype(bool)        # (T,2) [left, right]
        if has_mano_gt:
            for side in ("left", "right"):
                hands[side] = {
                    column: _col(f"{side}_mano_{column}").astype(np.float32)
                    for column in _MANO_COLS
                }
        if has_kpt21_gt:
            kpt21_gt = np.stack(
                [_col("left_kpt21"), _col("right_kpt21")], axis=1
            ).astype(np.float32).reshape(T, 2, 21, 3)
    else:
        extr = _col("extrinsics_w2c").astype(np.float64).reshape(T, 4, 4)
        cam_c2w = np.linalg.inv(extr)                 # world->cam 求逆得 cam->world
        fov0 = _col("fov").astype(np.float32)[0]
        K = geom.K_from_fov((fov0[0], fov0[1]), W, H)
        kept = np.stack([_col("left_kept"), _col("right_kept")], axis=1).astype(bool)  # (T,2)
        obs = _col("observation.state").astype(np.float32) if "observation.state" in names else None
        for side in ("left", "right"):
            orient_mat = _col(f"{side}_orient_world").astype(np.float32).reshape(T, 3, 3)
            pose_mat = _col(f"{side}_hand_pose").astype(np.float32).reshape(T, 15, 3, 3)
            if obs is not None:
                lo, hi = _BETAS_SLICE[side]
                betas = obs[:, lo:hi]
            else:
                betas = np.zeros((T, 10), dtype=np.float32)
            hands[side] = {
                "transl_cam": _col(f"{side}_transl_world").astype(np.float32).reshape(T, 3),  # 裸集:值为 world,key 名义统一 cam(hand_frame=world)
                "orient6d": geom.mat_to_6d(orient_mat),
                "pose6d": geom.mat_to_6d(pose_mat).reshape(T, 90),
                "betas": betas,
            }

    _raise_if_cancelled(cancel_check)
    return {
        "episode_index": ep["episode_index"],
        "T": T,
        "frames": frames,
        "hw": (int(H), int(W)),   # 显示帧分辨率(K 就按它反解)；frames 为 None 时从 meta/info.json 取
        "cam_c2w": cam_c2w.astype(np.float64),
        "K": K.astype(np.float64),
        "kept": kept,
        "hands": hands,
        "kpt21_gt": kpt21_gt,
        "cam_pose_enc": cam_pose_enc,
        "video": ep["video"],
        "dataset_root": ep.get("dataset_root"),
        "hand_frame": ep.get("hand_frame", "world"),   # camera/world:供 compare 决定是否把手转回 world
    }


def read_video_frames(path: str, fps: float | None = None, max_frames: int | None = None,
                      on_step=None, cancel_check=None) -> np.ndarray:
    """裸视频抽帧（无 GT 分支用）。fps 给定则按目标帧率均匀抽帧，否则取全部帧。

    on_step(done,total)：可选，分批读时逐批上报进度（网页端加载进度条用）；
    cancel_check()->bool：每个解码块前后检查。返回 (T,H0,W0,3) uint8 RGB。"""
    from decord import VideoReader
    _raise_if_cancelled(cancel_check, "视频解码")
    vr = VideoReader(path, num_threads=_DECORD_THREADS)   # 单 reader（索引一次）+ 多线程解码
    n = len(vr)
    if fps and fps > 0:
        src_fps = float(vr.get_avg_fps()) or 30.0
        step = max(1, int(round(src_fps / float(fps))))
        idx = list(range(0, n, step))
    else:
        idx = list(range(n))
    if max_frames is not None:
        idx = idx[: int(max_frames)]
    return _read_frames_chunked(
        vr, idx, on_step=on_step, cancel_check=cancel_check
    )
