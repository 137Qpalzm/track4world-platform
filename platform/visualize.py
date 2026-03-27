"""
platform/visualize.py — 可视化工具

提供 2D 视频展示、3D Plotly 点云、标注对比等功能。
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go


def load_ply_as_numpy(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a PLY file and return points (N,3) and colors (N,3) as numpy arrays.
    Uses open3d if available, falls back to manual parsing.
    """
    try:
        import open3d as o3d
        # Handle Chinese paths
        if any(ord(c) > 127 for c in ply_path):
            import tempfile, shutil
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
                shutil.copy2(ply_path, tmp.name)
                pcd = o3d.io.read_point_cloud(tmp.name)
            os.unlink(tmp.name)
        else:
            pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)
        return pts, cols
    except Exception:
        return _parse_ply_manual(ply_path)


def _parse_ply_manual(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback PLY parser for simple vertex-only PLY files."""
    with open(ply_path, 'rb') as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break
        # Parse vertex count
        n_verts = 0
        for line in header.decode('ascii', errors='ignore').split('\n'):
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
        # Read binary data
        data = np.frombuffer(f.read(), dtype=np.float32)
    # Assume x,y,z,r,g,b per vertex
    if len(data) >= n_verts * 6:
        data = data[:n_verts * 6].reshape(n_verts, 6)
        pts = data[:, :3]
        cols = data[:, 3:6]
        # Normalize colors if > 1
        if cols.max() > 1.0:
            cols = cols / 255.0
    else:
        data = data[:n_verts * 3].reshape(n_verts, 3)
        pts = data
        cols = np.ones_like(pts) * 0.5
    return pts, cols


def create_3d_plotly(
    ply_dir: str,
    frame_idx: int = 0,
    max_points: int = 50000,
    point_type: str = "frame",
) -> go.Figure:
    """
    Create a Plotly 3D scatter plot from PLY files.

    Args:
        ply_dir: Directory containing frame_xxx.ply / flow_xxx.ply
        frame_idx: Which frame to display
        max_points: Downsample to this many points
        point_type: "frame" for raw geometry, "flow" for tracked points
    """
    prefix = point_type
    ply_file = os.path.join(ply_dir, f"{prefix}_{frame_idx:03d}.ply")
    if not os.path.exists(ply_file):
        # Try without prefix
        ply_file = os.path.join(ply_dir, f"frame_{frame_idx:03d}.ply")
    if not os.path.exists(ply_file):
        fig = go.Figure()
        fig.add_annotation(text="未找到点云文件", showarrow=False, font=dict(size=20))
        return fig

    pts, cols = load_ply_as_numpy(ply_file)

    # Downsample if needed
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]

    # Convert colors to plotly format
    colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in cols]

    fig = go.Figure(data=[go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=colors, opacity=0.8),
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        title=f"3D 点云 — 帧 {frame_idx}",
    )
    return fig


def create_3d_plotly_multi_frame(
    ply_dir: str,
    num_frames: int = 5,
    max_points_per_frame: int = 10000,
) -> go.Figure:
    """Create a multi-frame 3D plot with color-coded frames."""
    import plotly.express as px
    colors = px.colors.qualitative.Set1

    fig = go.Figure()

    for i in range(num_frames):
        ply_file = os.path.join(ply_dir, f"flow_{i:03d}.ply")
        if not os.path.exists(ply_file):
            continue

        pts, cols = load_ply_as_numpy(ply_file)
        if len(pts) > max_points_per_frame:
            idx = np.random.choice(len(pts), max_points_per_frame, replace=False)
            pts = pts[idx]

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=1, color=colors[i % len(colors)], opacity=0.6),
            name=f"帧 {i}",
        ))

    fig.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        title="多帧 3D 追踪点云",
    )
    return fig


def load_labels(labels_path: str) -> List[Dict]:
    """Load labels.json for annotation comparison."""
    with open(labels_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_annotation_comparison(
    labels_path: str,
    max_frames: int = 100,
) -> Tuple[go.Figure, go.Figure]:
    """
    Create annotation visualization from labels.json.

    Returns:
        (fig_2d, fig_3d): 2D bbox trajectory + 3D hand/object trajectory plots
    """
    labels = load_labels(labels_path)[:max_frames]

    # Extract trajectories
    hand_3d = np.array([l['hand_position_3d'] for l in labels])
    obj_3d = np.array([l['club_head_position_3d'] for l in labels])
    frames = list(range(len(labels)))

    # 3D trajectory plot
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter3d(
        x=hand_3d[:, 0], y=hand_3d[:, 1], z=hand_3d[:, 2],
        mode='lines+markers',
        marker=dict(size=3, color=frames, colorscale='Viridis', showscale=True,
                    colorbar=dict(title="帧号")),
        line=dict(color='blue', width=2),
        name="手部轨迹 (GT)",
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=obj_3d[:, 0], y=obj_3d[:, 1], z=obj_3d[:, 2],
        mode='lines+markers',
        marker=dict(size=3, color='red'),
        line=dict(color='red', width=2),
        name="物体轨迹 (GT)",
    ))
    fig_3d.update_layout(
        scene=dict(aspectmode='data',
                   xaxis_title='X(m)', yaxis_title='Y(m)', zaxis_title='Z(m)'),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        title="GT 标注: 手部与物体 3D 轨迹",
    )

    # 2D bbox trajectory
    fig_2d = go.Figure()
    for i, l in enumerate(labels):
        bbox = l['bbox_2d']
        opacity = 0.1 + 0.9 * (i / len(labels))
        fig_2d.add_shape(
            type="rect",
            x0=bbox['x_min'], y0=bbox['y_min'],
            x1=bbox['x_max'], y1=bbox['y_max'],
            line=dict(color=f'rgba(0,100,255,{opacity})', width=1),
        )

    # Add start and end bbox highlighted
    if labels:
        for idx, color, name in [(0, 'green', '起始帧'), (-1, 'red', '结束帧')]:
            b = labels[idx]['bbox_2d']
            fig_2d.add_shape(type="rect",
                x0=b['x_min'], y0=b['y_min'], x1=b['x_max'], y1=b['y_max'],
                line=dict(color=color, width=3))
            fig_2d.add_annotation(x=(b['x_min']+b['x_max'])/2, y=b['y_min']-10,
                text=name, showarrow=False, font=dict(color=color, size=12))

    fig_2d.update_layout(
        xaxis=dict(range=[0, 1440], title="X (px)"),
        yaxis=dict(range=[1080, 0], title="Y (px)", scaleanchor="x"),
        height=500,
        title="GT 标注: 2D 边界框轨迹",
    )

    return fig_2d, fig_3d


def find_output_video(output_dir: str, mode: str) -> Optional[str]:
    """Find the result video in output directory."""
    if mode == "2d":
        vis_dir = os.path.join(output_dir, "2d_output")
        if os.path.exists(vis_dir):
            for f in os.listdir(vis_dir):
                if f.endswith('.mp4') and not f.startswith('temp_'):
                    return os.path.join(vis_dir, f)
            # Check for temp directories with mp4
            for d in os.listdir(vis_dir):
                dpath = os.path.join(vis_dir, d)
                if os.path.isdir(dpath):
                    for f in os.listdir(dpath):
                        if f.endswith('.mp4'):
                            return os.path.join(dpath, f)
    return None


def get_frame_count_in_ply_dir(ply_dir: str) -> int:
    """Count how many frame_xxx.ply files exist."""
    if not os.path.exists(ply_dir):
        return 0
    return len([f for f in os.listdir(ply_dir) if f.startswith('frame_') and f.endswith('.ply')])
