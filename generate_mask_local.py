"""
本地生成运动区域 mask（背景减除）

用法：
python generate_mask_local.py
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ========== 配置 ==========
VIDEO_PATH = "embodied_data/human_grasp/human_grasp_video0.mp4"
OUTPUT_DIR = "Track4World/results/human_grasp_3d_efep_masked/mask"
MAX_FRAMES = 500  # 对应100帧输出，设为 None 处理全部帧
# ==========================

def main():
    video_path = Path(VIDEO_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"错误：视频文件不存在 {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if MAX_FRAMES:
        total_frames = min(total_frames, MAX_FRAMES)

    print(f"视频路径: {video_path}")
    print(f"输出目录: {output_dir}")
    print(f"处理帧数: {total_frames}")

    # 改进参数：更低阈值 + 更大 close 核
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=200,
        varThreshold=8,
        detectShadows=False
    )

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    frame_idx = 0
    with tqdm(total=total_frames, desc="生成 mask") as pbar:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            fg_mask = bg_subtractor.apply(frame, learningRate=-1)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel_open)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
            _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

            mask_path = output_dir / f"mask_{frame_idx:04d}.png"
            cv2.imwrite(str(mask_path), fg_mask)

            frame_idx += 1
            pbar.update(1)

    cap.release()

    # 验证效果
    print("\n验证生成的 mask：")
    for i in [10, 50, 100, 200]:
        if i < frame_idx:
            m = cv2.imread(str(output_dir / f"mask_{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
            ratio = (m > 127).sum() / m.size
            print(f"  mask_{i:04d}: dynamic_ratio={ratio:.4f}")

    print(f"\n完成！生成了 {frame_idx} 个 mask 文件")
    print(f"输出目录: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
