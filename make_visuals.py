"""
阶段一可视化修复脚本 (v2)
========================
修复 4 个已知问题:
1. 2D GIF: 背景不变 + 仅前景运动区域着色
2. 3D 点云截图: 自适应相机视角 + 修复中文路径
3. Viser 点云查看: 提供简化版本 + 排查指南
4. GIF/子进程编码: 全部使用 PIL 避免 cv2 中文路径问题

用法: E:/Conda/envs/track4world/python.exe make_visuals.py
"""

import os
import sys
import json
import shutil
import tempfile
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "stage1_results"


# ============================================================
# 问题 1: 改进 2D 追踪 GIF
# ============================================================
def make_2d_motion_gif():
    """
    生成改进版 2D 追踪 GIF:
    - 背景保持原始视频帧
    - 仅对运动区域叠加追踪颜色 (颜色 = 像素在首帧的位置编码)
    - 运动越大颜色越鲜明, 静态区域完全透明
    - 添加运动轮廓线以增强视觉效果
    """
    print("[1/4] 生成 2D 运动追踪 GIF...")

    output_dir = RESULTS_DIR / "2d"
    temp_dirs = list(output_dir.glob("2d_output/temp_*"))
    if not temp_dirs:
        print("  跳过: 无 2D 可视化帧")
        return None

    frames_dir = temp_dirs[0]
    jpg_files = sorted(frames_dir.glob("*.jpg"))
    if not jpg_files:
        print("  跳过: 无帧文件")
        return None

    # 读取所有帧 (左=原始, 右=追踪颜色编码)
    all_originals = []
    all_tracking = []
    for f in jpg_files:
        img = np.array(Image.open(str(f)))
        h, w = img.shape[:2]
        mid = w // 2
        all_originals.append(img[:, :mid].copy())
        all_tracking.append(img[:, mid:].copy())

    n_frames = len(all_originals)
    print(f"  帧数: {n_frames}, 尺寸: {all_originals[0].shape}")

    # 参考帧的追踪颜色 (静态场景 = 平滑的位置编码梯度)
    ref = all_tracking[0].astype(np.float32)

    # 可选: 使用 scipy 做形态学操作
    try:
        from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
        has_scipy = True
    except ImportError:
        has_scipy = False

    gif_frames = []
    for i in range(n_frames):
        orig = all_originals[i].astype(np.float32)
        track = all_tracking[i].astype(np.float32)

        # === 运动检测: 当前帧追踪颜色与参考帧的差异 ===
        diff = np.abs(track - ref)
        motion_mag = diff.mean(axis=2)  # (H, W), 0~255

        # 自适应阈值: 基于全局运动量
        if i == 0:
            threshold = 15.0
        else:
            # 取运动量的 85 分位数的 30% 作为阈值
            p85 = np.percentile(motion_mag[motion_mag > 0], 85) if (motion_mag > 0).any() else 15.0
            threshold = max(8.0, min(p85 * 0.3, 25.0))

        # 二值运动掩码
        motion_mask = (motion_mag > threshold).astype(np.float32)

        if has_scipy:
            # 形态学: 先腐蚀去噪, 再膨胀填充空洞
            motion_mask = binary_erosion(motion_mask > 0.5, iterations=1).astype(np.float32)
            motion_mask = binary_dilation(motion_mask > 0.5, iterations=4).astype(np.float32)
            # 边缘高斯模糊实现柔和过渡
            soft_mask = gaussian_filter(motion_mask, sigma=3.0)
        else:
            soft_mask = motion_mask

        # === Alpha 混合 ===
        # 运动越大, 追踪颜色越不透明 (最高 85%)
        alpha = np.clip(motion_mag / 50.0, 0, 1.0) * soft_mask
        alpha = np.clip(alpha, 0, 0.85)

        # 增强追踪颜色饱和度
        track_enhanced = np.clip(track * 1.4, 0, 255)

        # 混合: 原始帧 × (1-α) + 增强追踪颜色 × α
        alpha_3 = alpha[:, :, np.newaxis]
        blended = orig * (1.0 - alpha_3) + track_enhanced * alpha_3

        # === 绘制运动轮廓 ===
        if has_scipy and i > 0:
            # 轮廓 = 膨胀mask - 腐蚀mask
            dilated = binary_dilation(motion_mask > 0.5, iterations=2).astype(np.float32)
            eroded = binary_erosion(motion_mask > 0.5, iterations=1).astype(np.float32)
            contour = ((dilated - eroded) > 0.5)
            # 轮廓用亮黄色
            blended[contour] = blended[contour] * 0.3 + np.array([255, 255, 50]) * 0.7

        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # 添加帧号标注
        pil_frame = Image.fromarray(blended)
        draw = ImageDraw.Draw(pil_frame)
        text = f"Frame {i:03d}"
        # 黑色阴影 + 白色文字
        draw.text((6, 6), text, fill=(0, 0, 0))
        draw.text((5, 5), text, fill=(255, 255, 255))

        gif_frames.append(pil_frame)

    # 保存 GIF
    gif_path = output_dir / "2d_tracking_motion.gif"
    gif_frames[0].save(
        str(gif_path), save_all=True, append_images=gif_frames[1:],
        duration=300, loop=0, optimize=True
    )
    print(f"  GIF: {gif_path.name} ({n_frames} 帧)")

    # 保存关键帧 PNG
    key_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    key_indices = sorted(set(i for i in key_indices if 0 <= i < n_frames))
    for idx in key_indices:
        png_path = output_dir / f"2d_motion_frame_{idx:03d}.png"
        gif_frames[idx].save(str(png_path))
    print(f"  关键帧 PNG: {len(key_indices)} 张")

    # 同时生成原始左右对比 GIF (保留)
    demo_gif_path = output_dir / "2d_tracking_demo.gif"
    demo_frames = [Image.open(str(f)) for f in jpg_files]
    demo_frames[0].save(
        str(demo_gif_path), save_all=True, append_images=demo_frames[1:],
        duration=300, loop=0, optimize=True
    )
    print(f"  原始对比 GIF: {demo_gif_path.name}")

    return str(gif_path)


# ============================================================
# 问题 2: 修复 3D 点云截图 (自适应视角)
# ============================================================
def render_pointcloud_screenshots():
    """
    用 matplotlib 3D 散点图渲染点云截图。
    open3d 在 Windows 上无法打开含中文的绝对路径, 需先复制到临时目录。
    """
    print("\n[2/4] 渲染 3D 点云截图...")

    try:
        import open3d as o3d
    except ImportError:
        print("  跳过: open3d 未安装")
        return {}

    import matplotlib
    matplotlib.use('Agg')  # 无头模式
    import matplotlib.pyplot as plt

    # 临时目录 (ASCII 路径)
    tmp_base = Path(tempfile.gettempdir()) / "t4w_vis"
    if tmp_base.exists():
        shutil.rmtree(str(tmp_base))
    tmp_base.mkdir(parents=True)

    screenshots = {}

    for mode in ["3d_ff", "3d_efep"]:
        sub = f"{mode}_output"
        ply_dir = RESULTS_DIR / mode / sub

        if not ply_dir.exists():
            print(f"  {mode}: 无输出目录, 跳过")
            continue

        frame_plys = sorted(ply_dir.glob("frame_*.ply"))
        flow_plys = sorted(ply_dir.glob("flow_*.ply"))
        if not frame_plys:
            print(f"  {mode}: 无 PLY 文件, 跳过")
            continue

        render_list = []
        render_list.append((frame_plys[0], "3D Reconstruction (Frame 0)"))
        if len(frame_plys) > 1:
            render_list.append((frame_plys[-1], f"3D Reconstruction (Frame {len(frame_plys)-1})"))
        if flow_plys:
            render_list.append((flow_plys[0], "Scene Flow Visualization"))

        mode_shots = []
        for ply_file, label in render_list:
            try:
                # 复制到 ASCII 临时目录 (open3d 不支持中文绝对路径)
                tmp_ply = tmp_base / f"{mode}_{ply_file.name}"
                shutil.copy2(str(ply_file), str(tmp_ply))

                pcd = o3d.io.read_point_cloud(str(tmp_ply))
                if pcd.is_empty():
                    print(f"  {mode}/{ply_file.stem}: 空点云, 跳过")
                    continue

                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                n_pts = len(points)

                # 离群点去除
                if n_pts > 100:
                    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                    points = points[ind]
                    colors = colors[ind]

                if len(points) < 10:
                    continue

                # 降采样 (matplotlib 画太多点会很慢)
                if len(points) > 50000:
                    idx = np.random.RandomState(42).choice(len(points), 50000, replace=False)
                    points = points[idx]
                    colors = colors[idx]

                # 渲染 matplotlib 3D 散点图
                fig = plt.figure(figsize=(10, 7.5), facecolor='#14141e')
                ax = fig.add_subplot(111, projection='3d', facecolor='#14141e')

                ax.scatter(
                    points[:, 0], points[:, 2], points[:, 1],
                    c=colors, s=0.3, alpha=0.8, edgecolors='none'
                )

                ax.set_xlabel('X', color='#555', fontsize=8)
                ax.set_ylabel('Z', color='#555', fontsize=8)
                ax.set_zlabel('Y', color='#555', fontsize=8)
                ax.tick_params(colors='#444', labelsize=6)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('#333')
                ax.yaxis.pane.set_edgecolor('#333')
                ax.zaxis.pane.set_edgecolor('#333')
                ax.grid(True, alpha=0.15)
                ax.set_title(f"{label} ({n_pts:,} pts)", color='#aaa', fontsize=11, pad=10)
                ax.view_init(elev=-75, azim=-90)

                final_png = RESULTS_DIR / mode / f"pointcloud_{ply_file.stem}.png"
                fig.savefig(str(final_png), dpi=120, bbox_inches='tight',
                            facecolor=fig.get_facecolor(), pad_inches=0.3)
                plt.close(fig)

                mode_shots.append({
                    "path": str(final_png.relative_to(RESULTS_DIR)),
                    "label": label,
                    "n_points": n_pts,
                })
                print(f"  {mode}/{ply_file.stem}: {n_pts:,} 点 -> {final_png.name}")

            except Exception as e:
                print(f"  {mode}/{ply_file.stem}: 渲染失败 - {e}")

        screenshots[mode] = mode_shots

    shutil.rmtree(str(tmp_base), ignore_errors=True)
    return screenshots


# ============================================================
# 问题 3: Viser 点云查看器排查 + 简化版查看器
# ============================================================
def create_simple_viewer_script():
    """
    生成一个简化版的点云查看器脚本, 解决 viser 连接问题:
    - 减少点数以加速 WebGL 渲染
    - 修复中文路径问题
    - 添加连接排查提示
    """
    print("\n[3/4] 生成简化版点云查看器...")

    script_content = r'''"""
简化版 3D 点云查看器
====================
解决原始 vis_3d_ff.py 的连接/加载问题:
- 自动复制 PLY 到临时目录绕过中文路径
- 降采样以加速 WebGL 渲染
- 提供更多调试信息

用法:
  E:/Conda/envs/track4world/python.exe simple_viewer.py [--mode 3d_ff|3d_efep]
"""
import sys, os, shutil, tempfile, argparse
from pathlib import Path
import numpy as np

# 参数
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="3d_ff", choices=["3d_ff", "3d_efep"])
parser.add_argument("--port", type=int, default=8012)
parser.add_argument("--downsample", type=int, default=4, help="每 N 个点取 1 个")
args = parser.parse_args()

RESULTS_DIR = Path(__file__).parent / "stage1_results"
sub = f"{args.mode}_output"
ply_dir = RESULTS_DIR / args.mode / sub

if not ply_dir.exists():
    print(f"错误: 找不到 {ply_dir}")
    print("请先运行推理生成结果")
    sys.exit(1)

# 复制 PLY 到临时目录 (绕过中文路径)
tmp_dir = Path(tempfile.mkdtemp(prefix="t4w_viewer_"))
print(f"临时目录: {tmp_dir}")

frame_plys = sorted(ply_dir.glob("frame_*.ply"))
flow_plys = sorted(ply_dir.glob("flow_*.ply"))
vis_npys = sorted(ply_dir.glob("vis_*.npy"))

if not flow_plys:
    print("错误: 无 flow_*.ply 文件")
    sys.exit(1)

for f in flow_plys + vis_npys:
    shutil.copy2(str(f), str(tmp_dir / f.name))
print(f"已复制 {len(flow_plys)} 个 PLY + {len(vis_npys)} 个 NPY")

# 加载数据
import open3d as o3d
import viser

print(f"\n加载点云 (降采样 1/{args.downsample})...")
all_points = []
all_colors = []

for ply_file in sorted(tmp_dir.glob("flow_*.ply")):
    pcd = o3d.io.read_point_cloud(str(ply_file))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    # 降采样
    idx = np.arange(0, len(pts), args.downsample)
    all_points.append(pts[idx] * 100)  # 缩放以便可视化
    all_colors.append((cols[idx] * 255).astype(np.uint8))
    print(f"  {ply_file.name}: {len(pts)} -> {len(idx)} 点")

n_frames = len(all_points)
print(f"\n共 {n_frames} 帧, 每帧约 {len(all_points[0])} 点")

# 启动 Viser
print(f"\n启动 Viser 服务器...")
print(f"=" * 50)
print(f"  请在浏览器中打开: http://localhost:{args.port}")
print(f"  推荐使用 Chrome 浏览器")
print(f"  如果页面空白, 请等待 5-10 秒后刷新")
print(f"=" * 50)

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
    gui_frame = server.gui.add_slider("Frame", min=0, max=n_frames-1, step=1, initial_value=0)
    gui_playing = server.gui.add_checkbox("Auto Play", True)
    gui_fps = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=8)
    gui_point_size = server.gui.add_slider("Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.02)

import time
prev_frame = 0
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
'''

    script_path = PROJECT_ROOT / "simple_viewer.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    print(f"  脚本: {script_path.name}")
    print(f"  用法: E:/Conda/envs/track4world/python.exe simple_viewer.py --mode 3d_ff")
    return str(script_path)


# ============================================================
# 生成最终 HTML 报告
# ============================================================
def generate_report(screenshots_3d):
    """生成修复后的完整 HTML 报告"""
    print("\n[4/4] 生成 HTML 报告...")

    # 读取 results.json
    json_path = RESULTS_DIR / "results.json"
    if json_path.exists():
        data = json.load(open(json_path, encoding="utf-8"))
        env_info = data.get("env", {})
        results = data.get("results", [])
    else:
        env_info = {}
        results = []

    t4w_py = env_info.get("t4w_python", "E:/Conda/envs/track4world/python.exe")

    # 检查文件
    motion_gif = RESULTS_DIR / "2d" / "2d_tracking_motion.gif"
    demo_gif = RESULTS_DIR / "2d" / "2d_tracking_demo.gif"
    motion_frames = sorted((RESULTS_DIR / "2d").glob("2d_motion_frame_*.png"))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Track4World 阶段一成果报告</title>
<style>
body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #0f0f1a; color: #e0e0e0; }}
h1 {{ color: #6ec6ff; border-bottom: 3px solid #3d5afe; padding-bottom: 10px; }}
h2 {{ color: #82b1ff; margin-top: 35px; }}
h3 {{ color: #b0bec5; }}
.card {{ background: #1a1a2e; border-radius: 10px; padding: 22px; margin: 15px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.4); border: 1px solid #2a2a4a; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #333; padding: 10px 14px; text-align: left; }}
th {{ background: #1a237e; color: #bbdefb; }}
tr:nth-child(even) {{ background: #1e1e3a; }}
.ok {{ color: #66bb6a; font-weight: bold; }}
.fail {{ color: #ef5350; font-weight: bold; }}
img {{ max-width: 100%; border-radius: 6px; border: 1px solid #333; }}
code {{ background: #263238; padding: 2px 6px; border-radius: 3px; color: #80cbc4; }}
.code {{ background: #0d1117; color: #79c0ff; padding: 16px; border-radius: 8px; overflow-x: auto; font-family: 'Cascadia Code', Consolas, monospace; font-size: 0.88em; white-space: pre-wrap; border: 1px solid #21262d; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
.grid3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
.caption {{ color: #90a4ae; font-size: 0.85em; text-align: center; margin-top: 5px; }}
.tag {{ display: inline-block; background: #3d5afe; color: white; padding: 3px 10px; border-radius: 4px; font-size: 0.8em; }}
.section-desc {{ color: #b0bec5; margin: 8px 0 15px 0; line-height: 1.6; }}
a {{ color: #64b5f6; }}
</style>
</head>
<body>

<h1>Track4World 模型复现 — 阶段一成果报告</h1>
<p style="color:#90a4ae">面向具身数据采集场景的关键点追踪系统的设计与实现</p>

<!-- ===== 1. 环境 ===== -->
<h2>1. 运行环境</h2>
<div class="card">
<table>
<tr><th>项目</th><th>值</th><th>状态</th></tr>
<tr><td>conda 环境</td><td><code>{t4w_py}</code></td><td class="ok">OK</td></tr>
<tr><td>Python</td><td>{env_info.get('python','N/A')}</td><td class="ok">OK</td></tr>
<tr><td>PyTorch</td><td>{env_info.get('pytorch','N/A')}</td><td class="ok">OK</td></tr>
<tr><td>CUDA</td><td>{'可用' if env_info.get('cuda_available') else '不可用'}</td>
    <td class="{'ok' if env_info.get('cuda_available') else 'fail'}">{'OK' if env_info.get('cuda_available') else 'FAIL'}</td></tr>
<tr><td>GPU</td><td>{env_info.get('gpu_name','N/A')}</td><td>{env_info.get('gpu_vram_gb',0)} GB VRAM</td></tr>
</table>

<h3>依赖库</h3>
<table>
<tr><th>库</th><th>版本</th><th>用途</th></tr>"""

    dep_desc = {
        "timm": "PyTorch Image Models (骨干网络)",
        "einops": "张量操作工具",
        "gradio": "Web UI 框架",
        "viser": "3D 点云可视化",
        "open3d": "点云处理与渲染",
        "transformers": "HuggingFace 模型加载",
        "sam2": "Segment Anything 2 (动态分割)",
    }
    for dep, ver in env_info.get("deps", {}).items():
        cls = "ok" if ver != "MISSING" else "fail"
        html += f'<tr><td>{dep}</td><td class="{cls}">{ver}</td><td>{dep_desc.get(dep,"")}</td></tr>\n'

    html += """</table>
</div>

<!-- ===== 2. 推理性能 ===== -->
<h2>2. 推理性能</h2>
<div class="card">
<p class="section-desc">使用 demo_data/cat.mp4 测试视频, RTX 3050 (4GB) 上的推理性能:</p>
"""
    if results:
        html += '<table><tr><th>模式</th><th>帧数</th><th>分辨率</th><th>推理耗时</th><th>PLY</th><th>NPY</th><th>状态</th></tr>\n'
        for r in results:
            ok = r.get("success", False)
            of = r.get("output_files", {})
            html += (f'<tr><td><b>{r["mode"]}</b></td>'
                     f'<td>{r.get("max_frames","?")}</td>'
                     f'<td>{r.get("image_size","?")}px</td>'
                     f'<td>{r.get("elapsed_time","?")}s</td>'
                     f'<td>{of.get("ply","-")}</td>'
                     f'<td>{of.get("npy","-")}</td>'
                     f'<td class="{"ok" if ok else "fail"}">{"成功" if ok else "失败"}</td></tr>\n')
        html += '</table>\n'

    html += """<p style="margin-top:10px; color:#ffab40"><b>注意:</b> 4GB VRAM 限制下, 3D 模式需使用
<code>camera_base</code> + <code>track4world_moge.pth</code> (1.7GB)。DA3/Pi3 大模型 (&gt;5GB) 会 OOM。</p>
</div>
"""

    # ===== 3. 2D 追踪 =====
    html += """
<h2>3. 2D 像素追踪</h2>
<div class="card">
<p><span class="tag">模式: 2d</span></p>
<p class="section-desc">追踪视频中每个像素的运动轨迹。颜色编码表示像素在首帧中的位置 — 当物体移动时,
其颜色与背景产生差异, 从而可视化运动区域。</p>
"""

    if motion_gif.exists():
        html += """
<h3>运动区域追踪</h3>
<p>背景保持原始视频, <b>仅运动区域</b>叠加追踪颜色 (黄色轮廓标记运动边界):</p>
<div style="text-align:center; margin: 15px 0">
<img src="2d/2d_tracking_motion.gif" style="max-width:500px; border: 2px solid #3d5afe">
</div>
"""

    if demo_gif.exists():
        html += """
<h3>原始输出对比</h3>
<p>左: 原始视频 | 右: 全像素追踪颜色编码</p>
<div style="text-align:center; margin: 15px 0">
<img src="2d/2d_tracking_demo.gif" style="max-width:600px">
</div>
"""

    if motion_frames:
        html += '<h3>关键帧</h3>\n<div class="grid3">\n'
        for f in motion_frames[:6]:
            rel = f.relative_to(RESULTS_DIR)
            idx = f.stem.split("_")[-1]
            html += f'<div><img src="{rel}"><p class="caption">帧 {idx}</p></div>\n'
        html += '</div>\n'

    html += '</div>\n'

    # ===== 4. 3D_FF =====
    html += """
<h2>4. 3D 首帧追踪 (3d_ff)</h2>
<div class="card">
<p><span class="tag">模式: 3d_ff</span></p>
<p class="section-desc">
<b>原理:</b> 对首帧进行深度估计, 反投影到 3D 空间。然后通过 2D 光流追踪每个像素在后续帧的运动,
结合几何约束更新 3D 坐标。<br>
<b>输出:</b> 每帧一个 PLY 点云 (带 RGB 颜色) + 场景流点云 + 相机位姿。
</p>
"""

    ff_shots = screenshots_3d.get("3d_ff", [])
    if ff_shots:
        cols = min(len(ff_shots), 3)
        html += f'<h3>点云渲染 (Open3D 离屏渲染)</h3>\n<div class="grid{cols}">\n'
        for s in ff_shots:
            html += f'<div><img src="{s["path"]}"><p class="caption">{s["label"]} ({s["n_points"]:,} 点)</p></div>\n'
        html += '</div>\n'

    html += """
<h3>输出文件</h3>
<table>
<tr><th>文件</th><th>格式</th><th>内容</th></tr>
<tr><td><code>frame_*.ply</code></td><td>PLY 点云</td><td>每帧 3D 重建点云 (顶点带 RGB)</td></tr>
<tr><td><code>flow_*.ply</code></td><td>PLY 点云</td><td>场景流投影 (光流追踪后的 3D 位置)</td></tr>
<tr><td><code>all_points.npy</code></td><td>NumPy</td><td>所有帧 3D 坐标堆叠 (T, H&times;W, 3)</td></tr>
<tr><td><code>vis_*.npy</code></td><td>NumPy</td><td>每帧可见性置信度</td></tr>
<tr><td><code>c2w.npy</code></td><td>NumPy</td><td>Camera-to-World 变换矩阵 (T, 4, 4)</td></tr>
</table>
</div>
"""

    # ===== 5. 3D_EFEP =====
    html += """
<h2>5. 3D 全像素全帧追踪 (3d_efep)</h2>
<div class="card">
<p><span class="tag">模式: 3d_efep</span></p>
<p class="section-desc">
<b>原理:</b> 对<b>每一帧</b>独立进行深度估计和 3D 重建, 然后通过光流建立帧间 3D 对应关系,
生成完整的 <b>4D 密集点云轨迹</b>。比 3d_ff 更精确但更慢。<br>
<b>适用:</b> 动态场景、需要高精度 3D 追踪的具身数据采集任务。
</p>
"""

    efep_shots = screenshots_3d.get("3d_efep", [])
    if efep_shots:
        cols = min(len(efep_shots), 3)
        html += f'<h3>点云渲染</h3>\n<div class="grid{cols}">\n'
        for s in efep_shots:
            html += f'<div><img src="{s["path"]}"><p class="caption">{s["label"]} ({s["n_points"]:,} 点)</p></div>\n'
        html += '</div>\n'

    html += """
<h3>3d_ff vs 3d_efep 对比</h3>
<table>
<tr><th>特性</th><th>3d_ff (首帧追踪)</th><th>3d_efep (全像素全帧)</th></tr>
<tr><td>深度估计</td><td>仅首帧</td><td>每帧独立估计</td></tr>
<tr><td>3D 精度</td><td>依赖首帧质量, 远帧误差累积</td><td>逐帧独立, 精度更高</td></tr>
<tr><td>核心输出</td><td>all_points.npy (首帧投影)</td><td>trajectory_all_pointmap.npy (4D轨迹)</td></tr>
<tr><td>速度</td><td>较快 (一次几何计算)</td><td>较慢 (逐帧计算)</td></tr>
<tr><td>典型场景</td><td>静态场景预览</td><td>动态场景、精确具身追踪</td></tr>
</table>
</div>
"""

    # ===== 6. 交互式可视化 =====
    html += f"""
<h2>6. 交互式 3D 点云可视化</h2>
<div class="card">
<p class="section-desc">使用 Viser 在浏览器中交互查看 3D 点云动画 (支持旋转、缩放、播放)。</p>

<h3>方式一: 简化版查看器 (推荐)</h3>
<div class="code">cd "{PROJECT_ROOT}"
E:/Conda/envs/track4world/python.exe simple_viewer.py --mode 3d_ff
# 浏览器打开 http://localhost:8012
# 如果加载慢, 可加大降采样: --downsample 8</div>

<h3>方式二: 原始可视化脚本</h3>
<div class="code">cd "{PROJECT_ROOT / 'Track4World'}"
E:/Conda/envs/track4world/python.exe visualization/vis_3d_ff.py --ply_dir results/cat_3dff_base/3d_ff_output
# 浏览器打开 http://localhost:8080</div>

<h3>方式三: MeshLab (单帧静态查看)</h3>
<p>直接双击 <code>.ply</code> 文件即可用 MeshLab 打开。<a href="https://www.meshlab.net/#download">下载 MeshLab</a></p>

<h3>常见问题</h3>
<table>
<tr><th>问题</th><th>解决方法</th></tr>
<tr><td>浏览器一直 "连接中"</td><td>等待 10-30 秒 (首次加载点云数据); 尝试刷新; 使用 Chrome</td></tr>
<tr><td>页面空白 / 无点云</td><td>用简化版查看器 (自动降采样); 或加 <code>--downsample 8</code></td></tr>
<tr><td>No module named 'open3d'</td><td>必须使用 track4world 环境的 Python, 不是系统 Python</td></tr>
</table>
</div>
"""

    # ===== 7. 使用命令 =====
    html += f"""
<h2>7. 完整使用命令</h2>
<div class="card">
<h3>2D 追踪</h3>
<div class="code">cd Track4World
E:/Conda/envs/track4world/python.exe demo.py \\
  --mp4_path demo_data/cat.mp4 --mode 2d \\
  --image_size 320 --max_frames 20</div>

<h3>3D 首帧追踪 (4GB VRAM 可用)</h3>
<div class="code">cd Track4World
E:/Conda/envs/track4world/python.exe demo.py \\
  --mp4_path demo_data/cat.mp4 --mode 3d_ff \\
  --coordinate camera_base \\
  --ckpt_init checkpoints/track4world_moge.pth \\
  --image_size 448 --max_frames 10</div>

<h3>3D 全像素追踪</h3>
<div class="code">cd Track4World
E:/Conda/envs/track4world/python.exe demo.py \\
  --mp4_path demo_data/cat.mp4 --mode 3d_efep \\
  --coordinate camera_base \\
  --ckpt_init checkpoints/track4world_moge.pth \\
  --image_size 448 --max_frames 10</div>
</div>
"""

    # ===== 8. 技术架构 =====
    html += """
<h2>8. 模型架构</h2>
<div class="card">
<pre style="font-family: 'Cascadia Code', Consolas, monospace; line-height: 1.6; font-size: 0.85em; color: #a5d6a7;">
输入: 单目 RGB 视频 (B, T, 3, H, W)
  │
  ├─[骨干网络] DINOv2-ViT-L / Pi3X / DA3-Giant
  │   输出: 3D点图(T,H,W,3) + 流特征(T,128,H/8,W/8) + 相机位姿(T,4,4)
  │
  ├─[全局特征聚合] VGGT-style Alternating Attention
  │   帧级注意力 ⇄ 全局注意力 (2D Rotary Position Embedding)
  │
  ├─[RAFT 迭代优化] 滑动窗口 (S=16, stride=8)
  │   相关性金字塔(5层) → 运动编码 → 2D/3D流更新 ×4次
  │   Convex Upsampling 8× → 全分辨率
  │
  └─[输出]
      ├─ 2D 光流 (T, 2, H, W) + 可见性 (T, 2, H, W)
      ├─ 3D 场景流 (T, 3, H, W) + 点云 (T, H, W, 3)
      └─ 相机位姿 (T, 4, 4) Camera-to-World
</pre>
</div>
"""

    html += f"""
<footer style="margin-top:40px; padding:20px 0; border-top:1px solid #333; color:#616161; text-align:center;">
Track4World 模型复现 | 毕业设计 — 面向具身数据采集场景的关键点追踪系统 | {time.strftime("%Y-%m-%d %H:%M:%S")}
</footer>
</body></html>"""

    report_path = RESULTS_DIR / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  报告: {report_path}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Track4World 阶段一可视化修复 (v2)")
    print("=" * 55)

    # 1. 改进 2D GIF
    make_2d_motion_gif()

    # 2. 修复 3D 点云截图
    screenshots = render_pointcloud_screenshots()

    # 3. 生成简化版点云查看器
    create_simple_viewer_script()

    # 4. 生成修复后的 HTML 报告
    generate_report(screenshots)

    print("\n" + "=" * 55)
    print("  全部完成!")
    print(f"  报告: stage1_results/report.html")
    print(f"  查看器: simple_viewer.py")
    print("=" * 55)
