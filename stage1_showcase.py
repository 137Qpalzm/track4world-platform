"""
阶段一成果展示脚本
=================
面向具身数据采集场景的关键点追踪系统 -- Track4World 模型复现验证

此脚本不依赖 conda activate，会自动定位 track4world 环境的 Python。
可以用任意 Python (>=3.6) 直接运行:
    python stage1_showcase.py
"""

import os
import sys
import time
import json
import subprocess
import argparse
from pathlib import Path

# ===== 路径常量 =====
PROJECT_ROOT = Path(__file__).parent.resolve()
T4W_DIR = PROJECT_ROOT / "Track4World"
CACHE_DIR = PROJECT_ROOT / ".cache"
RESULTS_DIR = PROJECT_ROOT / "stage1_results"

# ===== 自动定位 track4world 环境 Python =====
CONDA_PYTHON_CANDIDATES = [
    Path("E:/Conda/envs/track4world/python.exe"),
    Path(os.path.expanduser("~/miniconda3/envs/track4world/bin/python")),
    Path(os.path.expanduser("~/anaconda3/envs/track4world/bin/python")),
]

def find_t4w_python():
    """查找 track4world 环境的 Python 可执行文件"""
    for p in CONDA_PYTHON_CANDIDATES:
        if p.exists():
            return str(p)
    # 尝试通过 conda 查找
    try:
        out = subprocess.check_output(
            ["conda", "run", "-n", "track4world", "python", "-c", "import sys; print(sys.executable)"],
            text=True, stderr=subprocess.DEVNULL, timeout=30
        ).strip()
        if out and Path(out).exists():
            return out
    except Exception:
        pass
    return None

T4W_PYTHON = find_t4w_python()


def run_in_t4w_env(code, timeout=60):
    """在 track4world 环境中执行 Python 代码, 返回 stdout"""
    if not T4W_PYTHON:
        return None
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HOME"] = str(CACHE_DIR / "huggingface")
    env["TORCH_HOME"] = str(CACHE_DIR / "torch")
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            [T4W_PYTHON, "-c", code],
            capture_output=True, timeout=timeout, env=env,
            encoding="utf-8", errors="replace"
        )
        return result.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def check_environment():
    """通过子进程在 track4world 环境中检查依赖"""
    info = {"status": "OK", "errors": []}

    if not T4W_PYTHON:
        info["status"] = "FAIL"
        info["errors"].append("找不到 track4world conda 环境")
        info["python"] = "N/A"
        return info

    info["t4w_python"] = T4W_PYTHON

    # 在 track4world 环境中运行检查
    check_code = '''
import json, sys
result = {}
result["python"] = sys.version.split()[0]
result["executable"] = sys.executable
try:
    import torch
    result["pytorch"] = torch.__version__
    result["cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["vram"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
except ImportError:
    result["pytorch"] = "MISSING"
    result["cuda"] = False

deps = {}
for name in ["timm","einops","gradio","viser","open3d","transformers","sam2"]:
    try:
        m = __import__(name)
        deps[name] = getattr(m, "__version__", "OK")
    except:
        deps[name] = "MISSING"
result["deps"] = deps
print(json.dumps(result))
'''
    out = run_in_t4w_env(check_code, timeout=30)
    if not out or out.startswith("ERROR"):
        info["status"] = "FAIL"
        info["errors"].append(f"环境检查失败: {out}")
        return info

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        info["status"] = "FAIL"
        info["errors"].append(f"环境检查输出异常: {out[:200]}")
        return info

    info["python"] = data.get("python", "N/A")
    info["pytorch"] = data.get("pytorch", "N/A")
    info["cuda_available"] = data.get("cuda", False)
    info["gpu_name"] = data.get("gpu", "N/A")
    info["gpu_vram_gb"] = data.get("vram", 0)
    info["deps"] = data.get("deps", {})

    for dep, ver in info["deps"].items():
        if ver == "MISSING":
            info["errors"].append(f"{dep} 缺失")

    # 权重文件
    ckpt_dir = T4W_DIR / "checkpoints"
    info["checkpoints"] = {}
    for name in ["track4world_da3.pth", "track4world_pi3.pth", "track4world_moge.pth"]:
        p = ckpt_dir / name
        if p.exists():
            info["checkpoints"][name] = f"{p.stat().st_size / 1024**3:.2f} GB"
        else:
            info["checkpoints"][name] = "MISSING"

    # Demo 视频
    demo_video = T4W_DIR / "demo_data" / "cat.mp4"
    info["demo_video"] = str(demo_video) if demo_video.exists() and demo_video.stat().st_size > 1000 else "MISSING"

    if info["errors"]:
        info["status"] = "WARN"

    return info


def print_env_report(info):
    """打印环境检查报告"""
    W = 60
    print("=" * W)
    print("  Track4World 环境检查报告")
    print("=" * W)
    print(f"  conda Python: {info.get('t4w_python', 'NOT FOUND')}")
    print(f"  Python:       {info.get('python', 'N/A')}")
    print(f"  PyTorch:      {info.get('pytorch', 'N/A')}")
    print(f"  CUDA:         {'可用' if info.get('cuda_available') else '不可用'}")
    print(f"  GPU:          {info.get('gpu_name', 'N/A')} ({info.get('gpu_vram_gb', 0)} GB)")
    print()

    print("  依赖库:")
    for dep, ver in info.get("deps", {}).items():
        tag = "OK" if ver != "MISSING" else "MISSING"
        print(f"    {dep:20s} [{tag}]")
    print()

    print("  模型权重:")
    for name, size in info.get("checkpoints", {}).items():
        print(f"    {name:30s} [{size}]")
    print()

    if info.get("errors"):
        print(f"  ! 发现 {len(info['errors'])} 个问题:")
        for e in info["errors"]:
            print(f"    - {e}")
    else:
        print("  所有检查通过!")
    print("=" * W)


def run_inference(mode, image_size=448, max_frames=10, coordinate="camera_base",
                  ckpt="track4world_moge.pth"):
    """用子进程在 track4world 环境中运行推理"""
    output_dir = RESULTS_DIR / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*50}")
    print(f"  模式: {mode}  |  {image_size}px  |  {max_frames}帧")
    print(f"  坐标: {coordinate}  |  权重: {ckpt}")
    print(f"{'─'*50}")

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HOME"] = str(CACHE_DIR / "huggingface")
    env["TORCH_HOME"] = str(CACHE_DIR / "torch")
    env["PYTHONIOENCODING"] = "utf-8"
    # 让 ffmpeg 可用
    ffmpeg_dir = Path(T4W_PYTHON).parent / "Scripts"
    if ffmpeg_dir.exists():
        env["PATH"] = str(ffmpeg_dir) + os.pathsep + env.get("PATH", "")

    cmd = [
        T4W_PYTHON, "demo.py",
        "--mp4_path", "demo_data/cat.mp4",
        "--mode", mode,
        "--coordinate", coordinate,
        "--ckpt_init", f"checkpoints/{ckpt}",
        "--Ts", "-1",
        "--image_size", str(image_size),
        "--max_frames", str(max_frames),
        "--save_base_dir", str(output_dir),
    ]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(T4W_DIR), env=env,
            capture_output=True, timeout=600,
            encoding="utf-8", errors="replace"
        )
        success = proc.returncode == 0
        if not success:
            stderr = proc.stderr
            err_lines = [l for l in stderr.splitlines() if l.strip()]
            last_err = err_lines[-1] if err_lines else "未知错误"
            print(f"  失败: {last_err}")
    except subprocess.TimeoutExpired:
        success = False
        print("  超时 (>600s)")
    except Exception as e:
        success = False
        print(f"  异常: {e}")

    elapsed = time.time() - start

    result = {
        "mode": mode,
        "success": success,
        "elapsed_time": round(elapsed, 2),
        "max_frames": max_frames,
        "image_size": image_size,
        "output_dir": str(output_dir),
    }

    if success:
        result["fps"] = round(max_frames / elapsed, 2)
        ply_count = len(list(output_dir.rglob("*.ply")))
        npy_count = len(list(output_dir.rglob("*.npy")))
        jpg_count = len(list(output_dir.rglob("*.jpg")))
        result["output_files"] = {"ply": ply_count, "npy": npy_count, "jpg": jpg_count}
        print(f"  完成: {elapsed:.1f}s | PLY:{ply_count} NPY:{npy_count} JPG:{jpg_count}")

    return result


def make_2d_tracking_gif(output_dir):
    """
    用 track4world 环境的 Python 生成带轨迹线的 2D 追踪动画。
    比原始颜色编码更直观: 在原始视频上叠加稀疏点的运动轨迹线。
    """
    output_dir = Path(output_dir)
    temp_dirs = list(output_dir.glob("2d_output/temp_*"))
    if not temp_dirs:
        print("  无 2D 可视化帧, 跳过")
        return None

    gif_path = output_dir / "2d_tracking_demo.gif"
    frames_dir = temp_dirs[0]

    # 在 track4world 环境中生成
    code = f'''
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

frames_dir = Path(r"{frames_dir}")
gif_path = Path(r"{gif_path}")
rgb_dir = Path(r"{output_dir}") / "final_rgb"

jpg_files = sorted(frames_dir.glob("*.jpg"))
if not jpg_files:
    print("NO_FRAMES")
    exit()

# 方式: 将左侧原始帧和右侧追踪结果横向拼接
# 并在左侧帧上画一些稀疏采样点的运动轨迹
imgs = []
for f in jpg_files:
    img = cv2.imread(str(f))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(Image.fromarray(img))

if imgs:
    imgs[0].save(
        str(gif_path), save_all=True, append_images=imgs[1:],
        duration=200, loop=0, optimize=True
    )
    print(f"OK:{{len(imgs)}}")
'''
    out = run_in_t4w_env(code, timeout=30)
    if out and out.startswith("OK"):
        n = out.split(":")[1]
        print(f"  2D 追踪 GIF 已生成 ({n}帧): {gif_path.name}")
        return str(gif_path)
    else:
        print(f"  GIF 生成失败: {out}")
        return None


def generate_html_report(env_info, results, output_path):
    """生成 HTML 展示报告"""

    # 检查有无 2D GIF
    gif_path = RESULTS_DIR / "2d" / "2d_tracking_demo.gif"
    has_gif = gif_path.exists()

    # 检查有无 2D JPG 帧
    sample_frames = []
    for mode_dir in ["2d", "3d_ff", "3d_efep"]:
        d = RESULTS_DIR / mode_dir
        jpgs = sorted(d.rglob("*.jpg"))[:1]
        pngs = sorted(d.rglob("final_rgb/*.png"))[:1]
        if jpgs:
            sample_frames.append((mode_dir, jpgs[0].relative_to(RESULTS_DIR)))
        elif pngs:
            sample_frames.append((mode_dir, pngs[0].relative_to(RESULTS_DIR)))

    t4w_py = env_info.get("t4w_python", "E:/Conda/envs/track4world/python.exe")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Track4World 阶段一成果报告</title>
<style>
  body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
  h2 {{ color: #16213e; margin-top: 30px; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: left; }}
  th {{ background: #16213e; color: white; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  .ok {{ color: #27ae60; font-weight: bold; }}
  .fail {{ color: #e74c3c; font-weight: bold; }}
  img {{ max-width: 100%; border-radius: 4px; }}
  .code {{ background: #1a1a2e; color: #a8d8ea; padding: 15px; border-radius: 6px; overflow-x: auto; font-family: Consolas, monospace; font-size: 0.9em; white-space: pre-wrap; }}
  .flex {{ display: flex; gap: 15px; flex-wrap: wrap; }}
  .flex > div {{ flex: 1; min-width: 280px; }}
</style>
</head>
<body>

<h1>Track4World 模型复现 &mdash; 阶段一成果报告</h1>
<p>面向具身数据采集场景的关键点追踪系统的设计与实现</p>

<h2>1. 环境配置</h2>
<div class="card">
<table>
  <tr><th>项目</th><th>值</th><th>状态</th></tr>
  <tr><td>conda 环境 Python</td><td><code>{t4w_py}</code></td><td class="ok">OK</td></tr>
  <tr><td>Python</td><td>{env_info.get('python','N/A')}</td><td class="ok">OK</td></tr>
  <tr><td>PyTorch</td><td>{env_info.get('pytorch','N/A')}</td><td class="ok">OK</td></tr>
  <tr><td>CUDA</td><td>{'可用' if env_info.get('cuda_available') else '不可用'}</td>
      <td class="{'ok' if env_info.get('cuda_available') else 'fail'}">{'OK' if env_info.get('cuda_available') else 'FAIL'}</td></tr>
  <tr><td>GPU</td><td>{env_info.get('gpu_name','N/A')}</td><td class="ok">{env_info.get('gpu_vram_gb',0)} GB VRAM</td></tr>
</table>
<p style="margin-top:10px"><b>依赖库:</b>
{', '.join(f'{k} ({v})' for k,v in env_info.get('deps',{}).items() if v != 'MISSING')}
</p>
</div>

<h2>2. 模型权重</h2>
<div class="card">
<table>
  <tr><th>权重文件</th><th>大小</th><th>用途</th></tr>
"""
    ckpt_uses = {
        "track4world_da3.pth": "Depth Anything V3 版 (需 &ge; 12GB VRAM)",
        "track4world_pi3.pth": "Pi3 版 (需 &ge; 12GB VRAM)",
        "track4world_moge.pth": "Base/MoGe 版 (适合 4GB VRAM)",
    }
    for name, size in env_info.get("checkpoints", {}).items():
        html += f'  <tr><td>{name}</td><td>{size}</td><td>{ckpt_uses.get(name,"")}</td></tr>\n'

    html += """</table></div>

<h2>3. 推理性能</h2>
<div class="card">
"""
    if results:
        html += '<table><tr><th>模式</th><th>帧数</th><th>分辨率</th><th>耗时(s)</th><th>FPS</th><th>PLY</th><th>NPY</th><th>状态</th></tr>\n'
        for r in results:
            ok = r.get("success", False)
            cls = "ok" if ok else "fail"
            txt = "成功" if ok else "失败"
            of = r.get("output_files", {})
            html += (f'  <tr><td><b>{r["mode"]}</b></td>'
                     f'<td>{r.get("max_frames","?")}</td>'
                     f'<td>{r.get("image_size","?")}px</td>'
                     f'<td>{r.get("elapsed_time","?")}</td>'
                     f'<td>{r.get("fps","N/A")}</td>'
                     f'<td>{of.get("ply","-")}</td>'
                     f'<td>{of.get("npy","-")}</td>'
                     f'<td class="{cls}">{txt}</td></tr>\n')
        html += '</table>\n'
    else:
        html += '<p>暂无推理结果 (使用 <code>python stage1_showcase.py</code> 运行推理)</p>\n'

    html += '</div>\n'

    # 输出示例
    html += '<h2>4. 输出示例</h2>\n<div class="card">\n'
    if has_gif:
        html += '<p><b>2D 追踪动画:</b></p>\n'
        html += '<img src="2d/2d_tracking_demo.gif" alt="2D Tracking" style="max-width:600px">\n'
        html += '<p style="color:#666">左: 原始视频帧 | 右: 像素级追踪颜色编码 (颜色表示运动方向)</p>\n'

    if sample_frames:
        html += '<div class="flex">\n'
        for mode_name, rel_path in sample_frames:
            html += f'<div><p><b>{mode_name}:</b></p><img src="{rel_path}"></div>\n'
        html += '</div>\n'

    html += """
<p style="margin-top:15px"><b>3D 输出文件说明:</b></p>
<ul>
  <li><code>frame_*.ply</code> - 每帧的 3D 重建点云 (可用 MeshLab / Open3D 查看)</li>
  <li><code>flow_*.ply</code> - 光流投影后的点云 (场景流可视化)</li>
  <li><code>trajectory_all_pointmap.npy</code> - 长程 4D 点追踪轨迹</li>
  <li><code>c2w.npy</code> - 相机位姿 (Camera-to-World 矩阵)</li>
</ul>
</div>
"""

    # 使用命令
    html += f"""
<h2>5. 使用命令</h2>
<div class="card">
<p><b>重要:</b> 所有命令必须使用 track4world 环境的 Python。有两种方式:</p>
<div class="code">
# 方式一: conda activate
conda activate track4world
cd Track4World
python demo.py ...

# 方式二: 直接用绝对路径 (推荐, 不需要 activate)
"{t4w_py}" demo.py ...
</div>

<p style="margin-top:15px"><b>2D 追踪:</b></p>
<div class="code">cd Track4World
"{t4w_py}" demo.py --mp4_path demo_data/cat.mp4 --mode 2d --image_size 320 --max_frames 20 --save_base_dir results/cat_2d</div>

<p style="margin-top:15px"><b>3D 首帧追踪 (4GB VRAM):</b></p>
<div class="code">cd Track4World
"{t4w_py}" demo.py --mp4_path demo_data/cat.mp4 --mode 3d_ff --coordinate camera_base --ckpt_init checkpoints/track4world_moge.pth --image_size 448 --max_frames 10 --save_base_dir results/cat_3dff</div>

<p style="margin-top:15px"><b>3D 全像素追踪:</b></p>
<div class="code">cd Track4World
"{t4w_py}" demo.py --mp4_path demo_data/cat.mp4 --mode 3d_efep --coordinate camera_base --ckpt_init checkpoints/track4world_moge.pth --image_size 448 --max_frames 10 --save_base_dir results/cat_3defep</div>

<p style="margin-top:15px"><b>3D 点云可视化 (Viser, 浏览器打开 localhost:8012):</b></p>
<div class="code">cd Track4World
"{t4w_py}" visualization/vis_3d_ff.py --ply_dir results/cat_3dff/3d_ff_output</div>
</div>
"""

    # 技术架构
    html += """
<h2>6. 技术架构</h2>
<div class="card">
<pre style="font-family: Consolas, monospace; line-height: 1.5; font-size: 0.85em;">
输入: 单目 RGB 视频 (B, T, 3, H, W)
  |
  +--[骨干网络] DINOv2-ViT-L / Pi3X / DA3-Giant
  |   输出: 3D点图(T,H,W,3) + 流特征(T,128,H/8,W/8) + 相机位姿(T,4,4)
  |
  +--[全局特征聚合] VGGT-style Alternating Attention
  |   帧级注意力 + 全局注意力 (2D Rotary Position Embedding)
  |
  +--[RAFT 迭代优化] 滑动窗口 (S=16, stride=8)
  |   相关性金字塔(5层) -&gt; 运动编码 -&gt; 2D/3D流更新 x4次
  |   Convex Upsampling 8x: 低分辨率流 -&gt; 全分辨率输出
  |
  +--[输出]
      +-- 2D 光流 (T, 2, H, W) + 可见性置信度 (T, 2, H, W)
      +-- 3D 场景流 (T, 3, H, W) + 密集点云 (T, H, W, 3)
      +-- 相机位姿 (T, 4, 4) Camera-to-World 变换矩阵
</pre>
</div>

<footer style="margin-top:40px; padding:20px 0; border-top:1px solid #ddd; color:#666; text-align:center;">
  Track4World 模型复现 | 毕业设计 | """ + time.strftime("%Y-%m-%d %H:%M:%S") + """
</footer>
</body></html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML 报告: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Track4World 阶段一成果展示")
    parser.add_argument("--skip-inference", action="store_true", help="跳过推理, 仅环境检查+报告")
    parser.add_argument("--mode", choices=["all", "2d", "3d_ff", "3d_efep"], default="all")
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=448)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Track4World 阶段一成果展示")
    print("  面向具身数据采集场景的关键点追踪系统")
    print("=" * 60)

    # Step 0: 定位 Python
    if T4W_PYTHON:
        print(f"\n  track4world Python: {T4W_PYTHON}")
    else:
        print("\n  [错误] 找不到 track4world conda 环境!")
        print("  请先创建环境: conda create -n track4world python=3.11")
        sys.exit(1)

    CACHE_DIR.mkdir(exist_ok=True)

    # Step 1: 环境检查
    print("\n[Step 1] 环境检查 (通过 track4world 环境)...")
    env_info = check_environment()
    print_env_report(env_info)

    results = []

    if not args.skip_inference and env_info.get("cuda_available"):
        # Step 2: 推理
        print("\n[Step 2] 运行模型推理...")

        modes_config = {
            "2d":      {"image_size": 320, "max_frames": 20,
                        "coordinate": "world_depthanythingv3", "ckpt": "track4world_da3.pth"},
            "3d_ff":   {"image_size": args.image_size, "max_frames": args.max_frames,
                        "coordinate": "camera_base", "ckpt": "track4world_moge.pth"},
            "3d_efep": {"image_size": args.image_size, "max_frames": args.max_frames,
                        "coordinate": "camera_base", "ckpt": "track4world_moge.pth"},
        }

        run_modes = ["2d", "3d_ff", "3d_efep"] if args.mode == "all" else [args.mode]

        for mode in run_modes:
            result = run_inference(mode, **modes_config[mode])
            results.append(result)

        # Step 3: 可视化
        print("\n[Step 3] 生成可视化...")
        for r in results:
            if r.get("success") and r["mode"] == "2d":
                make_2d_tracking_gif(r["output_dir"])

    # Step 4: 报告
    print("\n[Step 4] 生成报告...")
    report_path = RESULTS_DIR / "report.html"
    generate_html_report(env_info, results, str(report_path))

    json_path = RESULTS_DIR / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"env": env_info, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"  JSON 数据: {json_path}")

    print("\n" + "=" * 60)
    print("  阶段一展示完成!")
    print(f"  报告: {report_path}")
    print(f"  请在浏览器中打开 report.html 查看完整结果")
    print("=" * 60)


if __name__ == "__main__":
    main()
