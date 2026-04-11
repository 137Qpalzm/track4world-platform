"""
生成运动区域 mask（背景减除）

用法：
python generate_motion_mask.py --video input.mp4 --output_dir masks/
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def generate_motion_masks(video_path: str, output_dir: str, max_frames: int = None):
    """
    使用背景减除算法生成运动区域 mask

    Args:
        video_path: 输入视频路径
        output_dir: 输出 mask 目录
        max_frames: 最大处理帧数（None = 全部）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"视频总帧数: {total_frames}")

    # 创建背景减除器（MOG2 算法）
    # history: 用于建模的帧数
    # varThreshold: 像素与模型的马氏距离阈值
    # detectShadows: 是否检测阴影
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )

    # 形态学操作的核
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    frame_idx = 0
    with tqdm(total=total_frames, desc="生成 mask") as pbar:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 应用背景减除
            fg_mask = bg_subtractor.apply(frame, learningRate=-1)

            # 去除阴影（MOG2 会把阴影标记为 127）
            fg_mask[fg_mask == 127] = 0

            # 形态学操作：去噪 + 填充空洞
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

            # 二值化（确保只有 0 和 255）
            _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

            # 保存 mask（PNG 格式）
            mask_path = output_dir / f"mask_{frame_idx:04d}.png"
            cv2.imwrite(str(mask_path), fg_mask)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"完成！生成了 {frame_idx} 个 mask 文件到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="生成运动区域 mask")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出 mask 目录")
    parser.add_argument("--max_frames", type=int, default=None, help="最大处理帧数")
    args = parser.parse_args()

    generate_motion_masks(args.video, args.output_dir, args.max_frames)


if __name__ == "__main__":
    main()
