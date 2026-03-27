"""
platform/app.py — Gradio 展示平台

面向具身数据采集场景的关键点追踪系统
"""

import os
import sys
import logging
from pathlib import Path

# 确保 ffmpeg/ffprobe 可被 Gradio 发现（static_ffmpeg 提供完整二进制）
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

import gradio as gr

# Setup paths
PLATFORM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLATFORM_DIR.parent
T4W_ROOT = PROJECT_ROOT / "Track4World"
EMBODIED_DATA = PROJECT_ROOT / "embodied_data"

sys.path.insert(0, str(PLATFORM_DIR))
sys.path.insert(0, str(T4W_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Preset video paths
PRESET_VIDEOS = {}
# Human grasp data
_human_mp4 = EMBODIED_DATA / "human_grasp" / "human_grasp_video0.mp4"
if _human_mp4.exists():
    PRESET_VIDEOS["人类抓握数据 (Video_0)"] = str(_human_mp4)
# Demo cat
_cat_mp4 = T4W_ROOT / "demo_data" / "cat.mp4"
if _cat_mp4.exists():
    PRESET_VIDEOS["Demo: 猫 (cat.mp4)"] = str(_cat_mp4)

# Labels path for annotation comparison
LABELS_PATH = EMBODIED_DATA / "human_grasp" / "Video_0" / "labels.json"


def on_preset_select(preset_name):
    """When user selects a preset video, copy to temp dir to avoid Chinese path issues."""
    if preset_name and preset_name in PRESET_VIDEOS:
        src = Path(PRESET_VIDEOS[preset_name])
        if not src.exists():
            logger.warning(f"Preset video not found: {src}")
            return None
        # Copy to temp dir with ASCII-only path to avoid Gradio/Windows issues
        import tempfile, shutil
        tmp_dir = Path(tempfile.gettempdir()) / "gradio_presets"
        tmp_dir.mkdir(exist_ok=True)
        dst = tmp_dir / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
            logger.info(f"Copied preset to: {dst}")
        return str(dst)
    return None


def run_tracking(video_path, mode, image_size, max_frames, progress=gr.Progress()):
    """Main inference function called by Gradio."""
    if not video_path:
        gr.Warning("请先上传或选择视频!")
        return None, None, None, "请先上传或选择视频"

    progress(0.05, desc="加载模型...")

    from inference import run_inference

    def progress_cb(frac, msg):
        progress(frac, desc=msg)

    try:
        results = run_inference(
            video_path=video_path,
            mode=mode,
            image_size=int(image_size),
            max_frames=int(max_frames),
            progress_callback=progress_cb,
        )
    except Exception as e:
        logger.exception("Inference failed")
        gr.Warning(f"推理失败: {e}")
        return None, None, None, f"推理失败: {e}"

    # Prepare outputs
    vis_video = None
    plotly_fig = None
    status_text = (
        f"完成! 模式: {results['mode']}, "
        f"帧数: {results['num_frames']}, "
        f"分辨率: {results['resolution']}, "
        f"耗时: {results['elapsed']:.1f}s"
    )

    if mode == "2d":
        from visualize import find_output_video
        vis_video = find_output_video(results['output_dir'], mode)

    elif mode in ("3d_ff", "3d_efep"):
        from visualize import create_3d_plotly_multi_frame
        ply_dir = results.get('ply_dir')
        if ply_dir and os.path.exists(ply_dir):
            plotly_fig = create_3d_plotly_multi_frame(
                ply_dir, num_frames=min(results['num_frames'], 10)
            )

    return vis_video, plotly_fig, results.get('output_dir'), status_text


def show_single_frame_3d(output_dir, frame_idx, point_type):
    """Show a single frame's 3D point cloud."""
    if not output_dir:
        return None
    from visualize import create_3d_plotly

    # Determine ply_dir
    for subdir in ['3d_ff_output', '3d_efep_output']:
        ply_dir = os.path.join(output_dir, subdir)
        if os.path.exists(ply_dir):
            return create_3d_plotly(ply_dir, int(frame_idx), point_type=point_type)
    return None


def show_annotations():
    """Show GT annotations from labels.json."""
    if not LABELS_PATH.exists():
        gr.Warning("未找到 labels.json 标注文件")
        return None, None
    from visualize import create_annotation_comparison
    fig_2d, fig_3d = create_annotation_comparison(str(LABELS_PATH))
    return fig_2d, fig_3d


def launch_viser(output_dir):
    """Launch Viser 3D viewer as separate process."""
    if not output_dir:
        return "请先运行 3D 推理"

    ply_dir = None
    for subdir in ['3d_ff_output', '3d_efep_output']:
        d = os.path.join(output_dir, subdir)
        if os.path.exists(d):
            ply_dir = d
            break

    if not ply_dir:
        return "未找到 3D 输出目录"

    import subprocess
    vis_script = str(T4W_ROOT / "visualization" / "vis_3d_ff.py")
    python_exe = sys.executable

    subprocess.Popen(
        [python_exe, vis_script, "--ply_dir", ply_dir],
        cwd=str(T4W_ROOT),
    )
    return "Viser 服务器已启动，请在浏览器打开 http://localhost:8080"


# ============================================================
# Gradio UI
# ============================================================

def build_app():
    with gr.Blocks(
        title="具身场景关键点追踪系统",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown("# 面向具身数据采集场景的关键点追踪系统")
        gr.Markdown("基于 Track4World 模型 | 支持 2D 光流追踪 / 3D 点云重建 / 全帧 4D 追踪")

        # State
        output_dir_state = gr.State(None)

        with gr.Row():
            # ============ Left: Controls ============
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 参数配置")

                with gr.Group():
                    video_input = gr.Video(label="上传视频")
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESET_VIDEOS.keys()),
                        label="或选择预置素材",
                        interactive=True,
                    )

                with gr.Group():
                    mode_radio = gr.Radio(
                        choices=["2d", "3d_ff", "3d_efep"],
                        value="2d",
                        label="追踪模式",
                        info="2d: 2D光流 | 3d_ff: 3D首帧 | 3d_efep: 3D全帧",
                    )
                    image_size = gr.Slider(
                        minimum=192, maximum=640, value=320, step=32,
                        label="分辨率 (max dim)",
                        info="越大越精确但越慢，4GB显存建议≤448",
                    )
                    max_frames = gr.Slider(
                        minimum=5, maximum=100, value=20, step=5,
                        label="最大帧数",
                        info="帧数越多耗时越长",
                    )

                run_btn = gr.Button("开始推理", variant="primary", size="lg")
                status_box = gr.Textbox(label="状态", interactive=False)

            # ============ Right: Results ============
            with gr.Column(scale=2):
                with gr.Tabs():
                    # Tab 1: Video preview
                    with gr.Tab("原始视频"):
                        preview_video = gr.Video(label="视频预览")

                    # Tab 2: 2D tracking result
                    with gr.Tab("2D 追踪结果"):
                        result_video = gr.Video(label="2D 追踪可视化")

                    # Tab 3: 3D Plotly
                    with gr.Tab("3D 点云 (Plotly)"):
                        plotly_3d = gr.Plot(label="3D 点云可视化")
                        with gr.Row():
                            frame_slider = gr.Slider(
                                minimum=0, maximum=20, value=0, step=1,
                                label="帧选择",
                            )
                            pt_type = gr.Radio(
                                choices=["frame", "flow"],
                                value="frame",
                                label="点云类型",
                            )
                            refresh_3d_btn = gr.Button("刷新单帧")

                    # Tab 4: Annotation comparison
                    with gr.Tab("标注对比 (GT)"):
                        gr.Markdown("展示 `labels.json` 中的 GT 标注数据")
                        show_anno_btn = gr.Button("加载 GT 标注")
                        anno_2d = gr.Plot(label="2D 边界框轨迹")
                        anno_3d = gr.Plot(label="3D 手部/物体轨迹")

                    # Tab 5: Viser (advanced)
                    with gr.Tab("Viser 3D 查看器"):
                        gr.Markdown("启动独立的 Viser 3D 交互式查看器（实验性）")
                        viser_btn = gr.Button("启动 Viser 服务器")
                        viser_status = gr.Textbox(label="Viser 状态", interactive=False)

        # ============ Event handlers ============

        # Preset selection → load video
        preset_dropdown.change(
            fn=on_preset_select,
            inputs=[preset_dropdown],
            outputs=[video_input],
        )

        # Video upload → preview
        video_input.change(
            fn=lambda v: v,
            inputs=[video_input],
            outputs=[preview_video],
        )

        # Run inference
        run_btn.click(
            fn=run_tracking,
            inputs=[video_input, mode_radio, image_size, max_frames],
            outputs=[result_video, plotly_3d, output_dir_state, status_box],
        )

        # Refresh single frame 3D
        refresh_3d_btn.click(
            fn=show_single_frame_3d,
            inputs=[output_dir_state, frame_slider, pt_type],
            outputs=[plotly_3d],
        )

        # Show annotations
        show_anno_btn.click(
            fn=show_annotations,
            inputs=[],
            outputs=[anno_2d, anno_3d],
        )

        # Launch Viser
        viser_btn.click(
            fn=launch_viser,
            inputs=[output_dir_state],
            outputs=[viser_status],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
