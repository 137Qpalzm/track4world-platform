"""
简化版 3D 点云查看器
====================
解决原始 vis_3d_ff.py 的连接/加载问题:
- 自动复制 PLY 到 ASCII 临时目录绕过中文路径
- 降采样以加速 WebGL 渲染
- 支持 3d_ff 和 3d_efep 两种模式
- 退出时自动清理临时文件

用法:
  E:/Conda/envs/track4world/python.exe simple_viewer.py [--mode 3d_ff|3d_efep]
"""
import sys
import os
import shutil
import tempfile
import argparse
import atexit
import signal
from pathlib import Path

import numpy as np

# 参数
parser = argparse.ArgumentParser(description="Track4World 3D 点云查看器")
parser.add_argument("--mode", default="3d_ff", choices=["3d_ff", "3d_efep"])
parser.add_argument("--port", type=int, default=8012)
parser.add_argument("--downsample", type=int, default=4, help="每 N 个点取 1 个 (越大越快)")
parser.add_argument("--ply_dir", type=str, default=None, help="直接指定 PLY 目录 (可选)")
args = parser.parse_args()

# 定位 PLY 目录
if args.ply_dir:
    ply_dir = Path(args.ply_dir)
else:
    RESULTS_DIR = Path(__file__).parent / "stage1_results"
    sub = f"{args.mode}_output"
    ply_dir = RESULTS_DIR / args.mode / sub

if not ply_dir.exists():
    print(f"[错误] 找不到 PLY 目录: {ply_dir}")
    print("请先运行推理生成 3D 输出, 或用 --ply_dir 指定路径")
    sys.exit(1)

# 复制 PLY 到 ASCII 临时目录 (绕过 Windows 中文路径兼容性问题)
tmp_dir = Path(tempfile.mkdtemp(prefix="t4w_viewer_"))
print(f"[1/3] 复制数据到临时目录: {tmp_dir}")


def cleanup_tmp():
    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        print(f"\n已清理临时目录: {tmp_dir}")


atexit.register(cleanup_tmp)

flow_plys = sorted(ply_dir.glob("flow_*.ply"))
frame_plys = sorted(ply_dir.glob("frame_*.ply"))

# 优先用 flow_*.ply (场景流动画), 其次用 frame_*.ply (静态点云)
use_plys = flow_plys if flow_plys else frame_plys
ply_type = "flow" if flow_plys else "frame"

if not use_plys:
    print(f"[错误] {ply_dir} 中无 PLY 文件")
    sys.exit(1)

for f in use_plys:
    shutil.copy2(str(f), str(tmp_dir / f.name))
print(f"  复制了 {len(use_plys)} 个 {ply_type}_*.ply")

# 加载数据
print(f"\n[2/3] 加载点云 (降采样 1/{args.downsample})...")

try:
    import open3d as o3d
except ImportError:
    print("[错误] open3d 未安装, 请用 track4world 环境的 Python 运行本脚本")
    sys.exit(1)

try:
    import viser
except ImportError:
    print("[错误] viser 未安装")
    sys.exit(1)

all_points = []
all_colors = []

for ply_file in sorted(tmp_dir.glob(f"{ply_type}_*.ply")):
    pcd = o3d.io.read_point_cloud(str(ply_file))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    if len(pts) == 0:
        print(f"  {ply_file.name}: 空点云, 跳过")
        continue

    # 降采样
    idx = np.arange(0, len(pts), args.downsample)
    all_points.append(pts[idx] * 100)  # 缩放以便 viser 可视化
    all_colors.append((cols[idx] * 255).astype(np.uint8))
    print(f"  {ply_file.name}: {len(pts):,} -> {len(idx):,} 点")

if not all_points:
    print("[错误] 没有加载到任何有效点云")
    sys.exit(1)

n_frames = len(all_points)
print(f"\n  共 {n_frames} 帧, 每帧约 {len(all_points[0]):,} 点")

# 启动 Viser
print(f"\n[3/3] 启动 Viser 服务器...")
print("=" * 50)
print(f"  请在浏览器中打开: http://localhost:{args.port}")
print(f"  推荐使用 Chrome 浏览器")
print(f"  首次加载可能需要 5-10 秒")
print(f"  按 Ctrl+C 退出")
print("=" * 50)

server = viser.ViserServer(port=args.port)

# 添加所有帧的点云 (只显示第一帧)
nodes = []
for i in range(n_frames):
    node = server.scene.add_point_cloud(
        name=f"/frame_{i:03d}",
        points=all_points[i],
        colors=all_colors[i],
        point_size=0.02,
        point_shape="rounded",
        visible=(i == 0),
    )
    nodes.append(node)

# GUI 控制
with server.gui.add_folder("Controls"):
    gui_frame = server.gui.add_slider(
        "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
    )
    gui_playing = server.gui.add_checkbox("Auto Play", True)
    gui_fps = server.gui.add_slider(
        "FPS", min=1, max=30, step=1, initial_value=8
    )
    gui_point_size = server.gui.add_slider(
        "Point Size", min=0.005, max=0.1, step=0.005, initial_value=0.02
    )

import time

prev_frame = 0
try:
    while True:
        if gui_playing.value:
            gui_frame.value = (gui_frame.value + 1) % n_frames

        cur = gui_frame.value
        if cur != prev_frame:
            nodes[prev_frame].visible = False
            nodes[cur].visible = True
            nodes[cur].point_size = gui_point_size.value
            prev_frame = cur

        time.sleep(1.0 / gui_fps.value)
except KeyboardInterrupt:
    print("\n退出查看器...")
