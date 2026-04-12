import time
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import viser
import viser.transforms as tf
from tqdm.auto import tqdm


def _safe_read_ply(filepath: Path) -> o3d.geometry.PointCloud:
    """Read PLY file, handling non-ASCII paths by copying to a temp dir."""
    path_str = str(filepath)
    try:
        path_str.encode('ascii')
        return o3d.io.read_point_cloud(path_str)
    except UnicodeEncodeError:
        pass
    # Non-ASCII path: copy to temp location
    tmp_dir = tempfile.mkdtemp(prefix="viser_ply_")
    tmp_file = Path(tmp_dir) / filepath.name
    shutil.copy2(filepath, tmp_file)
    pcd = o3d.io.read_point_cloud(str(tmp_file))
    shutil.rmtree(tmp_dir)
    return pcd

# ==============================================================================
# Configuration
# ==============================================================================

# Global downsample rate for trajectory lines to improve performance
POINT_DOWNSAMPLE_RATE = 50
# Downsample rate for point cloud display (reduce WebSocket message size)
PC_DOWNSAMPLE_RATE = 4

def main(
    max_frames: int = 400, 
    share: bool = False
) -> None:
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Track4World 3D-FF (First Frame) Visualization")
    parser.add_argument('--ply_dir', type=str, 
                        default='results/cat/3d_ff_output', 
                        help='Directory containing flow_xx.ply and vis_xx.npy files')
    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    if not ply_dir.exists():
        raise FileNotFoundError(f"Directory not found: {ply_dir}")

    # --- Viser Server Setup ---
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    # --- File Discovery ---
    # Sort files by frame index
    ply_files = sorted(ply_dir.glob("flow_*.ply"), key=lambda x: int(x.stem.split("_")[-1]))
    frame_ply_files = sorted(ply_dir.glob("frame_*.ply"), key=lambda x: int(x.stem.split("_")[-1]))
    vis_files = sorted(ply_dir.glob("vis_*.npy"), key=lambda x: int(x.stem.split("_")[-1]))
    dyn_mask_files = sorted(ply_dir.glob("pc_dyn_mask_*.npy"), key=lambda x: int(x.stem.split("_")[-1]))

    # 如果没有 pc_dyn_mask，尝试从上级目录的 mask/ 文件夹加载输入 mask PNG
    input_mask_dir = ply_dir.parent / "mask"
    input_mask_files = sorted(input_mask_dir.glob("mask_*.png"), key=lambda x: int(x.stem.split("_")[-1])) if input_mask_dir.exists() else []
    video_path = ply_dir.parent / "input_copy.mp4"

    num_frames = min(max_frames, len(ply_files))
    if num_frames == 0:
        raise RuntimeError(f"No valid flow_*.ply files found in {ply_dir}")

    # 如果没有 vis 文件，使用全1的可见性（所有点都可见）
    use_dummy_vis = len(vis_files) == 0
    if use_dummy_vis:
        print(f"Warning: No vis_*.npy files found. Using dummy visibility (all points visible).")
    elif len(vis_files) < num_frames:
        print(f"Warning: PLY files={len(ply_files)}, NPY files={len(vis_files)}. Using {len(vis_files)} frames.")
        num_frames = len(vis_files)

    # 检测运动分割 mask 来源（优先 pc_dyn_mask，其次输入 mask PNG）
    if len(dyn_mask_files) >= num_frames:
        use_motion_seg = "npy"
        print(f"Found {len(dyn_mask_files)} pc_dyn_mask files. Using output masks.")
    elif len(input_mask_files) >= num_frames and len(frame_ply_files) >= num_frames and video_path.exists():
        use_motion_seg = "png"
        print(f"Found {len(input_mask_files)} input mask PNGs. Using frame PLY for mask alignment.")
    else:
        use_motion_seg = None
        print(f"No motion mask files found. All points will be treated as dynamic.")

    # --- Data Loading & Processing ---
    # We need to find points that are valid (not outliers) across ALL frames
    # to draw consistent trajectories.

    raw_pcds = []
    raw_vis = []
    raw_dyn_masks = []  # 运动分割 mask
    common_indices = None

    print(f"Loading {num_frames} frames...")

    for i in tqdm(range(num_frames), desc="Processing Frames"):
        # 1. Load Point Cloud
        pcd = _safe_read_ply(ply_files[i])

        # 2. Load Visibility/Confidence Map
        # Shape is usually (N, 1) or (N,), we ensure it's 1D
        if use_dummy_vis:
            # 没有 vis 文件，使用全1可见性
            vis = np.ones(len(pcd.points), dtype=np.float32)
        else:
            vis = np.load(vis_files[i]).reshape(-1)

        # 3. Load Motion Mask (if available)
        if use_motion_seg == "npy":
            dyn_mask = np.load(dyn_mask_files[i]).reshape(-1)
            dyn_mask = (dyn_mask > 0.5).astype(np.float32)
        elif use_motion_seg == "png":
            # 用扫描顺序对齐：PLY 点按 320x448 的扫描顺序保存（跳过被过滤的像素）
            # 我们需要建立 pixel_index → point_index 的映射
            import cv2 as _cv2

            # 读取 frame PLY 来建立映射（frame PLY 和 mask 对齐更准确）
            frame_pcd = _safe_read_ply(frame_ply_files[i])
            n_frame_pts = len(frame_pcd.points)

            # 读取 mask 并 resize 到推理分辨率
            mask_img = _cv2.imread(str(input_mask_files[i]), _cv2.IMREAD_GRAYSCALE)
            mask_resized = _cv2.resize(mask_img, (448, 320), interpolation=_cv2.INTER_NEAREST)
            mask_flat = (mask_resized.flatten() > 127).astype(np.float32)  # (143360,)

            # frame PLY 的点是按扫描顺序保存的，但跳过了被 depth edge 过滤的像素
            # 我们假设前 n_frame_pts 个像素对应 frame PLY 的点（粗略近似）
            # 更精确的方法：用 frame PLY 的颜色匹配找到每个点对应的像素
            frame_cols = (np.asarray(frame_pcd.colors) * 255).astype(np.uint8)

            # 读取对应帧
            cap = _cv2.VideoCapture(str(video_path))
            for _ in range(i): cap.read()
            ret, frame = cap.read()
            cap.release()

            if not ret:
                dyn_mask = np.ones(len(pcd.points), dtype=np.float32)
            else:
                frame_resized = _cv2.resize(frame, (448, 320))
                frame_rgb = _cv2.cvtColor(frame_resized, _cv2.COLOR_BGR2RGB)
                frame_flat_rgb = frame_rgb.reshape(-1, 3)

                # 建立 pixel → has_point 的 boolean mask
                # 用颜色匹配找到哪些像素有对应的点
                from collections import defaultdict
                pixel_has_point = np.zeros(143360, dtype=bool)
                color_to_pixels = defaultdict(list)
                for pix_idx, c in enumerate(frame_flat_rgb):
                    color_to_pixels[tuple(c)].append(pix_idx)

                point_to_pixel = np.zeros(n_frame_pts, dtype=np.int32)
                for pt_idx, c in enumerate(frame_cols):
                    key = tuple(c)
                    if key in color_to_pixels and color_to_pixels[key]:
                        pix_idx = color_to_pixels[key].pop(0)
                        pixel_has_point[pix_idx] = True
                        point_to_pixel[pt_idx] = pix_idx

                # 查询 mask
                frame_dyn_mask = mask_flat[point_to_pixel]

                # 传递给 flow PLY（用颜色匹配）
                flow_cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
                frame_color_to_dyn = {}
                for j, c in enumerate(frame_cols):
                    frame_color_to_dyn[tuple(c)] = frame_dyn_mask[j]

                dyn_mask = np.array([frame_color_to_dyn.get(tuple(c), 0.0) for c in flow_cols])
        else:
            dyn_mask = np.ones(len(pcd.points), dtype=np.float32)

        # 4. Outlier Removal
        # We calculate indices of "good" points for this specific frame
        _, ind_list = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        current_indices = np.array(ind_list)

        # 5. Update Common Intersection
        # We only want to track points that survive filtering in EVERY frame (or the frames processed so far)
        if common_indices is None:
            common_indices = current_indices
        else:
            # Intersection of current valid points and previous common points
            common_indices = np.intersect1d(common_indices, current_indices)

        raw_pcds.append(pcd)
        raw_vis.append(vis)
        raw_dyn_masks.append(dyn_mask)

    if common_indices is None or len(common_indices) == 0:
        raise RuntimeError("No common points found across frames after filtering! Try increasing the outlier radius.")

    print(f"Tracking {len(common_indices)} common points across {num_frames} frames.")

    # --- Final Data Assembly ---
    all_points_list = []
    all_colors_list = []
    all_vis_list = []
    all_dyn_mask_list = []  # 运动分割 mask

    point_nodes: List[viser.PointCloudHandle] = []

    # Process filtered data for visualization
    for i in range(num_frames):
        pcd = raw_pcds[i]
        vis = raw_vis[i]
        dyn_mask = raw_dyn_masks[i]

        # Extract only the common points
        pts = np.asarray(pcd.points)[common_indices] * 100 # Scale up for visualization
        cols = np.asarray(pcd.colors)[common_indices]
        v = vis[common_indices]
        d = dyn_mask[common_indices]

        all_points_list.append(pts)
        all_colors_list.append(cols)
        all_vis_list.append(v)
        all_dyn_mask_list.append(d)

        # Add Point Cloud Node to Viser (Hidden by default)
        # We filter by visibility score > 0.1 for the point cloud display
        mask = v > 0.1
        # Downsample to avoid overwhelming WebSocket
        display_pts = pts[mask][::PC_DOWNSAMPLE_RATE]
        display_cols = cols[mask][::PC_DOWNSAMPLE_RATE]
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud",
                points=display_pts,
                colors=(display_cols * 255).astype(np.uint8),
                point_size=0.01,
                point_shape="rounded",
                visible=(i == 0) # Only show first frame initially
            )
        )

    # Stack into (T, N, 3) arrays
    all_points = np.stack(all_points_list, axis=0)
    all_vis = np.stack(all_vis_list, axis=0)
    all_dyn_masks = np.stack(all_dyn_mask_list, axis=0)  # (T, N)

    # 计算每个点的平均动态分数（用于分离静态/动态）
    avg_dyn_score = np.mean(all_dyn_masks, axis=0)  # (N,)
    is_dynamic = avg_dyn_score > 0.5  # 动态点
    is_static = ~is_dynamic  # 静态点

    print(f"Motion segmentation: {is_dynamic.sum()} dynamic points, {is_static.sum()} static points")

    # Generate consistent colors for trajectories based on the first frame
    # Using HSV colormap for distinct tracking lines
    num_tracks = len(common_indices)
    track_colors_hsv = plt.cm.get_cmap('hsv', num_tracks)
    initial_colors = (track_colors_hsv(np.arange(num_tracks))[:, :3] * 255).astype(np.uint8)

    # 为静态点生成灰色
    static_colors = np.full((num_tracks, 3), 150, dtype=np.uint8)  # 灰色
    # 混合：静态点用灰色，动态点用彩色
    initial_colors_mixed = np.where(is_dynamic[:, None], initial_colors, static_colors)

    # Subsample colors for the downsampled trajectory lines
    initial_colors_sampled = initial_colors_mixed[::POINT_DOWNSAMPLE_RATE]
    is_dynamic_sampled = is_dynamic[::POINT_DOWNSAMPLE_RATE]

    # ===================== GUI Controls =====================
    with server.gui.add_folder("Playback Controls"):
        # --- Playback Logic ---
        # Toggles the animation state (on/off)
        gui_playing = server.gui.add_checkbox("Playing", True)

        # Controls the speed of the temporal update
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=1, initial_value=24
        )

        # Manual scrub control for the current video/sequence frame
        gui_timestep = server.gui.add_slider(
            "Timestep", min=0, max=num_frames - 1, step=1, initial_value=0
        )

        # --- Appearance & Aesthetics ---
        # Controls the radius of individual 3D points
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=10, step=0.001, initial_value=0.01
        )

        # Controls the thickness of tracking lines/trajectories
        gui_line_width = server.gui.add_slider(
            "Line width", min=0.1, max=5.0, step=0.1, initial_value=0.5
        )

        # Sets how many historical frames of motion are visible behind a point
        gui_max_traj_length = server.gui.add_slider(
            "Trail Length (Frames)", min=1, max=50, step=1, initial_value=5
        )

        # --- Visualization Mode ---
        # Switch between viewing static geometry, motion paths, or both
        gui_vis_mode = server.gui.add_button_group(
            "Vis Mode", ("PointCloud", "Tracking", "Both")
        )
        gui_vis_mode.value = "Both"

    # ===================== Motion Segmentation Controls =====================
    with server.gui.add_folder("Motion Segmentation"):
        # 运动分割显示模式
        gui_motion_filter = server.gui.add_button_group(
            "Show", ("All", "Dynamic Only", "Static Only")
        )
        gui_motion_filter.value = "All"

        # 显示统计信息
        server.gui.add_text(
            "Statistics",
            initial_value=f"Dynamic: {is_dynamic.sum()} | Static: {is_static.sum()}",
            disabled=True
        )
        
    # ===================== Trajectory Setup =====================
    # Line node for drawing trails
    line_node = server.scene.add_line_segments(
        name="/trajectories",
        points=np.zeros((0, 2, 3)),
        colors=np.zeros((0, 2, 3), dtype=np.uint8),
        line_width=gui_line_width.value,
        visible=True,
    )

    # History buffers for the trail effect
    all_line_positions = []
    all_line_colors = []

    def update_trajectories(t_curr: int, show_lines: bool):
        """Updates the trajectory lines based on current timestep."""
        if not show_lines:
            line_node.visible = False
            return

        line_node.visible = True
        line_node.line_width = gui_line_width.value

        if t_curr == 0:
            return

        # Get positions for t-1 and t
        # Apply downsampling to reduce rendering load
        prev_pts = all_points[t_curr - 1, ::POINT_DOWNSAMPLE_RATE]
        curr_pts = all_points[t_curr, ::POINT_DOWNSAMPLE_RATE]

        # Check visibility for both frames
        prev_vis = all_vis[t_curr - 1, ::POINT_DOWNSAMPLE_RATE] > 0.1
        curr_vis = all_vis[t_curr, ::POINT_DOWNSAMPLE_RATE] > 0.1
        valid_mask = prev_vis & curr_vis

        # 根据运动分割过滤
        motion_filter = gui_motion_filter.value
        if motion_filter == "Dynamic Only":
            valid_mask = valid_mask & is_dynamic_sampled
        elif motion_filter == "Static Only":
            valid_mask = valid_mask & (~is_dynamic_sampled)
        # "All" 不过滤

        if not np.any(valid_mask):
            return

        # Create line segments
        p1 = prev_pts[valid_mask]
        p2 = curr_pts[valid_mask]
        new_lines = np.stack([p1, p2], axis=1) # (M, 2, 3)

        # Create colors (repeated for start/end of segment)
        c = initial_colors_sampled[valid_mask]
        new_colors = np.stack([c, c], axis=1) # (M, 2, 3)

        # Update history
        all_line_positions.append(new_lines)
        all_line_colors.append(new_colors)

        # Trim history
        MAX_TRAJECTORY_LENGTH = gui_max_traj_length.value
        if len(all_line_positions) > MAX_TRAJECTORY_LENGTH:
            all_line_positions.pop(0)
            all_line_colors.pop(0)

        # Update Viser Node
        if all_line_positions:
            line_node.points = np.concatenate(all_line_positions, axis=0)
            line_node.colors = np.concatenate(all_line_colors, axis=0)

    # Initial update
    update_trajectories(gui_timestep.value, True)

    # ===================== Main Loop =====================
    prev_timestep = gui_timestep.value

    while True:
        # 1. Handle Playback
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        current_timestep = gui_timestep.value

        # Reset trails if looping back to start
        if current_timestep == 0:
            all_line_positions = []
            all_line_colors = []
            line_node.points = np.zeros((0, 2, 3))

        # 2. Determine Visibility based on Mode
        show_points = gui_vis_mode.value in ("PointCloud", "Both")
        show_lines = gui_vis_mode.value in ("Tracking", "Both")

        # 3. Update Point Clouds
        # Only update if timestep changed or visibility toggled
        if current_timestep != prev_timestep or True: # Logic simplified for robustness
            for i, node in enumerate(point_nodes):
                is_current = (i == current_timestep)
                node.visible = is_current and show_points
                if is_current:
                    node.point_size = gui_point_size.value

        # 4. Update Trajectories
        if current_timestep != prev_timestep:
            update_trajectories(current_timestep, show_lines)
            prev_timestep = current_timestep
        else:
            # Just toggle visibility if paused
            line_node.visible = show_lines

        time.sleep(1.0 / gui_framerate.value)

if __name__ == "__main__":
    main()

