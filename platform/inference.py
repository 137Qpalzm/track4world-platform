"""
platform/inference.py — Track4World 推理接口封装

将 demo.py 的核心功能封装为可调用的 Python API，
供 Gradio 前端调用。
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import numpy as np

# Add Track4World to path
T4W_ROOT = Path(__file__).resolve().parent.parent / "Track4World"
sys.path.insert(0, str(T4W_ROOT))

logger = logging.getLogger(__name__)

# Global model singleton
_model = None
_model_args = None


def _make_args(**overrides) -> argparse.Namespace:
    """Create an argparse.Namespace matching demo.py expectations."""
    defaults = dict(
        coordinate="camera_base",
        ckpt_init=str(T4W_ROOT / "checkpoints" / "track4world_moge.pth"),
        config_path=str(T4W_ROOT / "track4world" / "config" / "eval" / "v1.json"),
        use_original_backbone=False,
        mode="2d",
        vis_mode="geometry",
        query_frame=0,
        image_size=320,
        max_frames=30,
        inference_iters=4,
        Ts=-1,
        rate=1,
        conf_thr=0.5,
        bkg_opacity=0.5,
        vstack=False,
        hstack=True,
        save_base_dir=str(T4W_ROOT / "results" / "gradio_output"),
        mp4_path="",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def get_model(force_reload: bool = False) -> torch.nn.Module:
    """Load model as singleton. Returns cached model on subsequent calls."""
    global _model, _model_args
    if _model is not None and not force_reload:
        return _model

    args = _make_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # Import and load
    os.chdir(str(T4W_ROOT))
    from demo import load_model
    _model = load_model(args, config)
    _model_args = args
    logger.info("Model loaded (singleton).")
    return _model


def run_inference(
    video_path: str,
    mode: str = "2d",
    image_size: int = 320,
    max_frames: int = 30,
    output_dir: Optional[str] = None,
    progress_callback=None,
) -> Dict:
    """
    Run Track4World inference on a video.

    Args:
        video_path: Path to input MP4
        mode: "2d", "3d_ff", or "3d_efep"
        image_size: Max dimension for resize
        max_frames: Maximum frames to process
        output_dir: Override output directory
        progress_callback: Optional callable(float, str) for progress updates

    Returns:
        Dict with result paths and metadata
    """
    os.chdir(str(T4W_ROOT))

    model = get_model()

    # Prepare args
    if output_dir is None:
        video_name = Path(video_path).stem
        output_dir = str(T4W_ROOT / "results" / f"gradio_{video_name}_{mode}")

    args = _make_args(
        mode=mode,
        image_size=image_size,
        max_frames=max_frames,
        mp4_path=video_path,
        save_base_dir=output_dir,
    )

    if progress_callback:
        progress_callback(0.1, "读取视频...")

    # Read video
    import cv2
    from demo import read_mp4
    frames, fps = read_mp4(video_path)
    logger.info(f"Read {len(frames)} frames @ {fps} FPS, resolution {frames[0].shape[:2]}")

    # Limit frames
    if len(frames) > max_frames:
        frames = frames[:max_frames]

    if progress_callback:
        progress_callback(0.2, f"预处理 {len(frames)} 帧...")

    # 3D modes need larger resolution (multiple downsamples inside CorrBlock)
    if mode in ("3d_ff", "3d_efep") and image_size < 448:
        logger.info(f"3D mode requires image_size >= 448, auto-adjusting from {image_size}")
        image_size = 448

    # Preprocess: resize + normalize (same logic as demo.py run_demo)
    H_orig, W_orig = frames[0].shape[:2]
    scale = min(image_size / H_orig, image_size / W_orig)
    H, W = int(H_orig * scale), int(W_orig * scale)
    H, W = (H // 64) * 64, (W // 64) * 64
    logger.info(f"Resizing from ({H_orig},{W_orig}) to ({H},{W})")

    resized = [cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in frames]
    rgbs = torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).float()  # (T, 3, H, W)
    rgbs = rgbs.unsqueeze(0).cuda()  # (1, T, C, H, W)

    if progress_callback:
        progress_callback(0.3, f"运行 {mode} 推理...")

    # Run inference
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    results = {}

    if mode == "2d":
        from demo import forward_video
        forward_video(rgbs, fps, model, args)
        # forward_video saves JPGs; ffmpeg may fail on Chinese paths.
        # Fallback: synthesize mp4 from JPGs with cv2.
        import cv2 as _cv2
        vis_dir = os.path.join(output_dir, "2d_output")
        results["vis_video"] = None
        results["frames_dir"] = os.path.join(output_dir, "final_rgb")
        if os.path.exists(vis_dir):
            # First check if ffmpeg already created an mp4
            mp4_files = [f for f in os.listdir(vis_dir) if f.endswith('.mp4')]
            if mp4_files:
                results["vis_video"] = os.path.join(vis_dir, mp4_files[0])
            else:
                # Find JPG frames in temp subdirectory and synthesize mp4
                for d in os.listdir(vis_dir):
                    dpath = os.path.join(vis_dir, d)
                    if os.path.isdir(dpath):
                        jpgs = sorted([f for f in os.listdir(dpath) if f.endswith('.jpg')])
                        if jpgs:
                            out_mp4 = os.path.join(vis_dir, "result_2d.mp4")
                            sample = _cv2.imread(os.path.join(dpath, jpgs[0]))
                            h_out, w_out = sample.shape[:2]
                            # 尝试 H.264 编码（浏览器兼容），失败则回退 mp4v
                            for fourcc_str in ("avc1", "H264", "mp4v"):
                                writer = _cv2.VideoWriter(
                                    out_mp4, _cv2.VideoWriter_fourcc(*fourcc_str),
                                    fps, (w_out, h_out))
                                if writer.isOpened():
                                    logger.info(f"Using VideoWriter fourcc: {fourcc_str}")
                                    break
                                writer.release()
                            for j in jpgs:
                                writer.write(_cv2.imread(os.path.join(dpath, j)))
                            writer.release()
                            results["vis_video"] = out_mp4
                            logger.info(f"Synthesized 2D result video: {out_mp4}")
                            break

    elif mode == "3d_ff":
        from demo import forward_video3d_ff, save_ff
        ff_results, _ = forward_video3d_ff(rgbs, model, args)
        ff_dir = os.path.join(output_dir, "3d_ff_output")
        save_ff(ff_results, ff_dir, rgbs.shape[-1], rgbs.shape[-2])
        results["ply_dir"] = ff_dir
        results["all_points"] = os.path.join(ff_dir, "all_points.npy")
        results["c2w"] = os.path.join(ff_dir, "c2w.npy")

    elif mode == "3d_efep":
        from demo import forward_video3d_pair, save_efep
        efep_results, _ = forward_video3d_pair(rgbs, model, args)
        efep_dir = os.path.join(output_dir, "3d_efep_output")
        # Create zero masks (no segmentation) for save_efep
        T_sel = rgbs.shape[1]
        masks_tensor = torch.zeros(T_sel, rgbs.shape[-2], rgbs.shape[-1])
        save_efep(efep_results, masks_tensor.cuda(), efep_dir,
                  rgbs.shape[-1], rgbs.shape[-2], args.vis_mode, args.coordinate)
        results["ply_dir"] = efep_dir
        results["trajectory"] = os.path.join(efep_dir, "trajectory_all_pointmap.npy")

    elapsed = time.time() - start_time

    if progress_callback:
        progress_callback(1.0, f"完成! 耗时 {elapsed:.1f}s")

    results.update({
        "mode": mode,
        "output_dir": output_dir,
        "elapsed": elapsed,
        "num_frames": len(frames),
        "resolution": f"{rgbs.shape[-1]}x{rgbs.shape[-2]}",
    })

    torch.cuda.empty_cache()
    return results
