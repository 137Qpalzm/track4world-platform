import time
import argparse
import tempfile
import shutil
import gc
from pathlib import Path
from typing import Tuple, List, Optional
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import viser
from tqdm.auto import tqdm


def _safe_read_ply(filepath: Path) -> o3d.geometry.PointCloud:
    """Read PLY file, handling non-ASCII paths by copying to a temp dir."""
    path_str = str(filepath)
    try:
        path_str.encode('ascii')
        return o3d.io.read_point_cloud(path_str)
    except UnicodeEncodeError:
        pass
    tmp_dir = tempfile.mkdtemp(prefix="viser_ply_")
    tmp_file = Path(tmp_dir) / filepath.name
    shutil.copy2(filepath, tmp_file)
    pcd = o3d.io.read_point_cloud(str(tmp_file))
    shutil.rmtree(tmp_dir)
    return pcd
from vis_3d_efep import (
    remove_std_outlier_open3d, 
    process_trajectories, 
    fill_trajectory_gaps,
    smooth_trajectories_temporal
)
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import matplotlib.colors as mcolors
# ==============================================================================
# Global Constants
# ==============================================================================
DEFAULT_POINT_DOWNSAMPLE_RATE = 20  # Downsample rate for trajectories
STATIC_SKIP_FRAMES = 5             # Skip frames for accumulating static points
STATIC_VOXEL_SIZE = 0.005           # Voxel size for downsampling static background (5mm, preserve small objects like balls)
MAX_DISPLACEMENT = 5             # Maximum displacement for trajectory segments
DEFAULT_CAM_POS = (1.2, 2.5, 0.0)
DEFAULT_LOOK_AT = (-1.5, 0.0, 0.0)
DEFAULT_UP = (-0.08, 0.0, 0.4)
DEFAULT_WXYZ = (-0.05, 0.98, -0.17, -0.12)

# ==============================================================================
# Helper Functions
# ==============================================================================

def fade_color_saturation_batch(rgb_uint8, factor):
    """
    Batch process RGB arrays (N, 3)
    factor: 0.0 (grayscale) to 1.0 (original color)
    """
    if factor == 1.0:
        return rgb_uint8
    rgb_float = rgb_uint8 / 255.0
    hsv = mcolors.rgb_to_hsv(rgb_float)
    hsv[..., 1] *= factor  # Only scale the saturation channel
    new_rgb = mcolors.hsv_to_rgb(hsv)
    return (new_rgb * 255).astype(np.uint8)

def fade_color_saturation(rgb_uint8, factor):
    """
    rgb_uint8: (3,) uint8 array
    factor: 0.0 (fade to gray) to 1.0 (original color)
    """
    # Normalize to 0-1 and convert to HSV
    rgb_float = rgb_uint8 / 255.0
    hsv = mcolors.rgb_to_hsv(rgb_float)
    
    # Reduce saturation, maintain brightness
    hsv[1] = hsv[1] * factor 
    
    # Convert back to RGB
    new_rgb = mcolors.hsv_to_rgb(hsv)
    return (new_rgb * 255).astype(np.uint8)

def canonicalize_quaternions(quats):
    """
    Solve the quaternion double cover problem (q and -q represent the same rotation).
    Ensure the dot product of quaternions in adjacent frames is positive to guarantee the shortest interpolation path.
    """
    canon_quats = quats.copy()
    for i in range(1, len(canon_quats)):
        prev = canon_quats[i - 1]
        curr = canon_quats[i]
        # If the dot product is negative, it means taking the "long path", so negate the current quaternion
        if np.dot(prev, curr) < 0:
            canon_quats[i] = -curr
    return canon_quats

def smooth_translation_spline(t, smoothing=0.5):
    """
    Use UnivariateSpline for translation smoothing.
    Compared to Savitzky-Golay, Spline better ensures global physical continuity (C2 continuity).
    smoothing: Smoothing factor, larger means smoother, smaller means closer to original trajectory.
    """
    T = t.shape[0]
    x = np.arange(T)
    t_smooth = np.zeros_like(t)
    
    # Fit x, y, z respectively
    weights = np.ones(T) 
    # (Optional) If confidence is available, weights of some frames can be reduced
    
    for i in range(3):
        # s is the smoothing parameter, needs to be adjusted based on data noise level
        # s=0 interpolates through all points, a large s becomes a straight line
        spl = UnivariateSpline(x, t[:, i], w=weights, s=smoothing)
        t_smooth[:, i] = spl(x)
        
    return t_smooth

def smooth_rotation_savgol(rot_objs, win=21, poly=3):
    """
    Smooth rotations.
    1. Extract quaternions
    2. Canonicalize
    3. Savitzky-Golay filtering
    4. Normalize
    """
    quats = rot_objs.as_quat()
    
    # Key step: resolve sign flips
    quats = canonicalize_quaternions(quats)
    
    # Ensure window is odd
    if win % 2 == 0: win += 1
    
    # Filter
    quats_smooth = savgol_filter(quats, window_length=win, polyorder=poly, axis=0, mode='interp')
    
    # Normalize (norm is not 1 after filtering)
    quats_smooth /= np.linalg.norm(quats_smooth, axis=1, keepdims=True)
    
    return R.from_quat(quats_smooth)

def smooth_c2w(c2w, 
                        trans_smoothing=1.0, # Translation smoothness (Spline s parameter)
                        rot_window=21,       # Rotation window size
                        rot_poly=3):         # Rotation polynomial order
    """
    c2w: (T, 4, 4) or (T, 3, 4)
    """
    T = c2w.shape[0]
    c2w_smooth = c2w.copy()
    
    # 1. Separate rotation and translation
    raw_t = c2w[:, :3, 3]
    raw_R = c2w[:, :3, :3]
    
    # 2. Smooth translation (Spline method)
    # Spline's s parameter depends on the data value range.
    # If data is in meters (scale~1.0), s=0.1~1.0 is suitable.
    # If translation jitter is severe, increase s.
    print(f"Smoothing Translation (Spline, s={trans_smoothing})...")
    smooth_t = smooth_translation_spline(raw_t, smoothing=trans_smoothing)
    
    # 3. Smooth rotation (Savitzky-Golay on Unwrapped Quaternions)
    print(f"Smoothing Rotation (SavGol, win={rot_window}, poly={rot_poly})...")
    rot_objs = R.from_matrix(raw_R)
    smooth_r_objs = smooth_rotation_savgol(rot_objs, win=rot_window, poly=rot_poly)
    smooth_R = smooth_r_objs.as_matrix()
    
    # 4. Recombine
    c2w_smooth[:, :3, :3] = smooth_R
    c2w_smooth[:, :3, 3] = smooth_t
    
    return c2w_smooth

def cam_points_to_world(points_cam, c2w):
    N = points_cam.shape[0]
    if N == 0:
        return points_cam
    points_h = np.concatenate([points_cam, np.ones((N, 1))], axis=1)
    points_world = (c2w @ points_h.T).T[:, :3]
    return points_world

def project_trajectories_to_image(
    trajectories_3d_down, visibility_mask_down, initial_colors,
    deleted_mask, selected_mask, pinned_traj_id,
    c2w, frame_idx, img_w=640, img_h=480
):
    """
    将当前帧的轨迹点反变换回相机坐标系，投影到 2D 图像。
    返回 PIL Image (RGB)，以及每个可见轨迹点的 (px, py, traj_id) 列表。
    """
    from PIL import Image as PILImage, ImageDraw
    img = PILImage.new("RGB", (img_w, img_h), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    c2w_t = c2w[frame_idx]  # (4,4)
    w2c = np.linalg.inv(c2w_t)

    # 估算内参：假设 FOV=60°，图像中心为主点
    fx = img_w / (2 * np.tan(np.radians(30)))
    fy = fx
    cx, cy = img_w / 2, img_h / 2

    vis_frame = visibility_mask_down[frame_idx]  # (N,)
    pts_world = 100 * trajectories_3d_down[frame_idx]  # (N,3)，已是世界坐标×100

    point_list = []  # (px, py, traj_id)

    for i in range(pts_world.shape[0]):
        if not vis_frame[i]:
            continue
        if deleted_mask[i]:
            continue

        # 世界坐标 → 相机坐标（注意 vis_3d_efep_world 里做了 *[1,-1,-1]，这里需还原）
        pw = pts_world[i] / [1, -1, -1]  # 还原翻转
        pw_h = np.array([pw[0], pw[1], pw[2], 1.0])
        pc = (w2c @ pw_h)[:3]

        if pc[2] <= 0:
            continue

        px = int(fx * pc[0] / pc[2] + cx)
        py = int(fy * pc[1] / pc[2] + cy)

        if not (0 <= px < img_w and 0 <= py < img_h):
            continue

        point_list.append((px, py, i))

        # 颜色：钉子轨迹=红，选中=白，普通=原色
        if i == pinned_traj_id:
            color = (255, 50, 50)
            r = 5
        elif selected_mask[i]:
            color = (255, 255, 255)
            r = 4
        else:
            c = initial_colors[i]
            color = (int(c[0]), int(c[1]), int(c[2]))
            r = 2

        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

    # 画十字准星（当前 XY 滑块位置）
    return img, point_list

# ==============================================================================
# Main Execution
# ==============================================================================

def main(
    max_frames: int = 400,
    share: bool = False,
) -> None:
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ply_dir', type=str,
        default='results/cat/3d_efep_output',
        help='Directory containing frame_xx.ply files and masks'
    )
    parser.add_argument(
        '--save_dir', type=str,
        default='recordings/cat',
        help='Directory to save output files'
    )
    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if not ply_dir.exists():
        raise FileNotFoundError(f"{ply_dir} not found")

    # --- Viser Server Setup ---
    server = viser.ViserServer()
    
    # [Fix 1]: Use callback to set initial camera
    # Automatically set camera position when a client connects
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        print(f"New client connected! Setting camera for {client.client_id}")
        client.camera.position = DEFAULT_CAM_POS
        client.camera.look_at = DEFAULT_LOOK_AT
        client.camera.up_direction = DEFAULT_UP
        client.camera.wxyz = DEFAULT_WXYZ

    if share:
        server.request_share_url()

    # --- Load File Paths ---
    ply_files = sorted([f for f in ply_dir.glob("frame_*.ply")], key=lambda x: int(x.stem.split("_")[-1]))
    num_frames = min(max_frames, len(ply_files))
    if num_frames == 0:
        raise RuntimeError(f"No valid frame_*.ply files found in {ply_dir}")

    # --- 1. Load Trajectory Data & Mask ---
    print("Loading trajectory data...")
    traj_path = ply_dir / 'trajectory_all_pointmap.npy'
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    # Use memory mapping for large trajectory files
    print(f"Loading trajectory from {traj_path}...")
    trajectory_all_raw = np.load(str(traj_path), mmap_mode='r')
    print(f"Trajectory shape: {trajectory_all_raw.shape}, size: {trajectory_all_raw.nbytes/1e9:.2f} GB")

    traj_mask_path = ply_dir / 'trajectory_all_pointmap_dyn_mask.npy'
    if not traj_mask_path.exists():
        print(f"Warning: Trajectory mask not found. Assuming all dynamic.")
        traj_dyn_mask_raw = np.ones(
            (trajectory_all_raw.shape[0], trajectory_all_raw.shape[1]),
            dtype=bool
        )
    else:
        print(f"Loading trajectory mask from {traj_mask_path}...")
        # Use memory mapping to avoid loading entire file into RAM
        try:
            traj_dyn_mask_raw = np.load(str(traj_mask_path), mmap_mode='r')
            print(f"Loaded trajectory mask with memory mapping: {traj_dyn_mask_raw.shape}")
            # Convert to regular array only if small enough, otherwise keep as memmap
            if traj_dyn_mask_raw.nbytes < 1e9:  # Less than 1GB
                traj_dyn_mask_raw = np.array(traj_dyn_mask_raw)
            else:
                print(f"Keeping as memory-mapped array ({traj_dyn_mask_raw.nbytes/1e9:.2f} GB)")
        except Exception as e:
            print(f"Error loading trajectory mask: {e}")
            print("Falling back to assuming all dynamic.")
            traj_dyn_mask_raw = np.ones(
                (trajectory_all_raw.shape[0], trajectory_all_raw.shape[1]),
                dtype=bool
            )

        if traj_dyn_mask_raw.ndim == 3:
            traj_dyn_mask_raw = traj_dyn_mask_raw.squeeze(-1)

    # --- 2. Load Camera Poses & Transform Trajectories ---
    c2w_path = ply_dir / 'c2w.npy'
    if not c2w_path.exists():
        raise FileNotFoundError(f"Camera pose file not found: {c2w_path}")
    c2w = np.load(str(c2w_path))
    # c2w = smooth_c2w(c2w, rot_window=c2w.shape[0]-1)
    trajectories_3d = trajectory_all_raw.copy()
    for i in tqdm(range(trajectory_all_raw.shape[0]), desc="Transforming Trajectories"):
        world_pts = cam_points_to_world(trajectory_all_raw[i], c2w[i])
        trajectories_3d[i] = world_pts * [1, -1, -1]
    
    visibility_mask = ~np.isnan(trajectory_all_raw).any(axis=-1)
    # Process trajectories (outlier removal, etc.)
    trajectories_3d, visibility_mask, traj_dyn_mask_raw = process_trajectories(
        trajectories_3d, visibility_mask, traj_dyn_mask_raw, 
        k_consecutive=5, jump_threshold=12, acc_threshold=10
    )
    # trajectories_3d, visibility_mask, traj_dyn_mask_raw = process_trajectories(
    #     trajectories_3d, visibility_mask, traj_dyn_mask_raw, 
    #     k_consecutive=5, jump_threshold=0.3, acc_threshold=0.3
    # )
    print("Filtering trajectories...")

    # =========================================================
    # 1. Global Filtering: Keep only trajectories that are 
    #    "dynamic at all visible moments"
    # =========================================================
    dyn = traj_dyn_mask_raw.astype(bool)
    vis = visibility_mask.astype(bool)

    # Keep trajectories where at least 30% of visible frames are dynamic
    vis_count = vis.sum(axis=0).astype(float)
    dyn_and_vis_count = (dyn & vis).sum(axis=0).astype(float)
    dyn_ratio = np.where(vis_count > 0, dyn_and_vis_count / vis_count, 0)
    keep_pure_dynamic_mask = dyn_ratio >= 0.3  # at least 30% dynamic frames

    # Slice arrays directly to remove invalid trajectories
    trajectories_3d = trajectories_3d[:, keep_pure_dynamic_mask]
    visibility_mask = visibility_mask[:, keep_pure_dynamic_mask]
    traj_dyn_mask_raw = traj_dyn_mask_raw[:, keep_pure_dynamic_mask]
    print(f"Remaining trajectories after dynamic filter: {trajectories_3d.shape[1]}")

    # =========================================================
    # 2. Per-frame Spatial Denoising
    # =========================================================
    # Note: If trajectories are very sparse, this might remove good points.
    # If lines disappear completely, comment out this loop.
    for i in tqdm(range(trajectories_3d.shape[0]), desc="Spatial Outlier Removal"):
        mask = visibility_mask[i]
        if np.sum(mask) == 0:
            continue
            
        pts = trajectories_3d[i][mask]
        
        # Only perform statistical filtering if there are enough points
        if pts.shape[0] > 10: 
            _, ind = remove_std_outlier_open3d(pts)
            
            # Update mask
            valid_indices = np.where(mask)[0]
            new_mask = np.zeros_like(mask, dtype=bool)
            if ind.shape[0] > 0:
                filtered_indices = valid_indices[ind]
                new_mask[filtered_indices] = True
            visibility_mask[i] = new_mask

    # =========================================================
    # 3. [Critical] Repair Breaks (Gap Filling)
    # =========================================================
    # Must run after denoising to fill short-term breaks
    print("Repairing broken trajectories...")
    trajectories_3d, visibility_mask = fill_trajectory_gaps(
        trajectories_3d, 
        visibility_mask, 
        max_gap=5  
    )
    trajectories_3d = smooth_trajectories_temporal(
        trajectories_3d, 
        visibility_mask, 
        sigma=3.0 
    )
    # --- 3. Load Point Clouds (Split Static/Dynamic) ---
    dynamic_point_nodes: List[viser.PointCloudHandle] = []
    dynamic_colors_original: List[np.ndarray] = []  # New: store original color copies
    static_points_accumulator = []
    static_colors_accumulator = []

    print(f"Loading {num_frames} frames and splitting static/dynamic...")

    for i, ply_file in enumerate(tqdm(ply_files[:num_frames], desc="Processing Frames")):
        pcd = _safe_read_ply(ply_file)
        points_local = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        mask_path = ply_dir / f"pc_dyn_mask_{i:03d}.npy"
        if not mask_path.exists():
            mask_path_alt = ply_dir / f"pc_dyn_mask_{i}.npy"
            if mask_path_alt.exists():
                mask_path = mask_path_alt
            else:
                is_dynamic = np.ones(points_local.shape[0], dtype=bool)

        if mask_path.exists():
            is_dynamic = np.load(str(mask_path))
            if is_dynamic.ndim > 1:
                is_dynamic = is_dynamic.squeeze()
            is_dynamic = (is_dynamic > 0)

        min_len = min(len(is_dynamic), len(points_local))
        is_dynamic = is_dynamic[:min_len]
        points_local = points_local[:min_len]
        colors = colors[:min_len]

        points_world = cam_points_to_world(points_local, c2w[i])
        points_world = 100 * points_world * [1, -1, -1]

        pts_dyn = points_world[is_dynamic]
        col_dyn_uint8 = (colors[is_dynamic] * 255).astype(np.uint8) # Convert to uint8
        pts_stat = points_world[~is_dynamic]
        col_stat = colors[~is_dynamic]

        # Add dynamic points to the scene
        node = server.scene.add_point_cloud(
            name=f"/dynamic/t{i}",
            points=pts_dyn,
            colors=col_dyn_uint8,
            point_size=0.01,
            point_shape="rounded",
            visible=False
        )
        dynamic_point_nodes.append(node)
        dynamic_colors_original.append(col_dyn_uint8) # Save copy

        # Accumulate static points
        if i % STATIC_SKIP_FRAMES == 0 and len(pts_stat) > 0:
            static_points_accumulator.append(pts_stat)
            static_colors_accumulator.append(col_stat)

        # Memory management: explicitly delete large temporary arrays
        del pcd, points_local, colors, is_dynamic, points_world, pts_dyn, pts_stat, col_stat

        # Force garbage collection every 10 frames to prevent memory buildup
        if i % 10 == 0:
            gc.collect()

    # --- 4. Process Static Background ---
    print("Merging and downsampling static background...")
    static_node: Optional[viser.PointCloudHandle] = None

    if static_points_accumulator:
        all_static_pts = np.concatenate(static_points_accumulator, axis=0)
        all_static_cols = np.concatenate(static_colors_accumulator, axis=0)

        # Free memory immediately after concatenation
        del static_points_accumulator, static_colors_accumulator
        gc.collect()

        pcd_static = o3d.geometry.PointCloud()
        pcd_static.points = o3d.utility.Vector3dVector(all_static_pts)
        pcd_static.colors = o3d.utility.Vector3dVector(all_static_cols)

        # Free memory after creating point cloud
        del all_static_pts, all_static_cols
        gc.collect()

        # Remove statistical outliers and downsample
        pcd_static, ind = pcd_static.remove_statistical_outlier(
            nb_neighbors=128, std_ratio=4.0
        )
        pcd_static_down = pcd_static.voxel_down_sample(voxel_size=STATIC_VOXEL_SIZE)

        # Free memory after downsampling
        del pcd_static
        gc.collect()

        final_static_pts = np.asarray(pcd_static_down.points)
        final_static_cols = (np.asarray(pcd_static_down.colors) * 255).astype(np.uint8)

        # Free memory after extracting arrays
        del pcd_static_down
        gc.collect()

        print(f"Static background: downsampled to {len(final_static_pts)} points")

        # Add static points to the scene
        static_node = server.scene.add_point_cloud(
            name="/static_background",
            points=final_static_pts,
            colors=final_static_cols,
            point_size=0.01,
            point_shape="rounded",
            visible=True
        )
    else:
        print("Warning: No static points found.")

    # --- 5. Prepare Dynamic Trajectory Colors ---
    trajectories_3d_down = trajectories_3d[:, 3::DEFAULT_POINT_DOWNSAMPLE_RATE]
    visibility_mask_down = visibility_mask[:, 3::DEFAULT_POINT_DOWNSAMPLE_RATE]
    
    N_traj = trajectories_3d_down.shape[1]
    initial_colors = np.zeros((N_traj, 3), dtype=np.uint8)
    
    if N_traj > 0:
        first_visible_idx = np.argmax(visibility_mask_down, axis=0)
        never_visible = ~np.any(visibility_mask_down, axis=0)
        
        first_visible_idx[never_visible] = 0
        indices = np.arange(N_traj)
        first_visible_xyz = trajectories_3d_down[first_visible_idx, indices]
        first_visible_xyz[never_visible] = np.nan
        
        xyz_min = np.nanmin(first_visible_xyz, axis=0)
        xyz_max = np.nanmax(first_visible_xyz, axis=0)
        xyz_norm = (first_visible_xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)
        scalar = np.nansum(xyz_norm, axis=1)
        
        scalar = (scalar - np.nanmin(scalar)) / (np.nanmax(scalar) - np.nanmin(scalar) + 1e-6)
        sort_idx = np.argsort(scalar)
        colors_hsv = plt.cm.hsv(np.linspace(0, 1, 5 * N_traj))[:, :3]
    
        # Assign colors sorted by spatial position
        sorted_hsv = colors_hsv[3 * N_traj + np.argsort(sort_idx)]
        initial_colors = (sorted_hsv * 255).astype(np.uint8)
    else:
        initial_colors = np.zeros((0, 3), dtype=np.uint8)

    # --- 6. GUI Controls (两个Tab布局) ---
    tabs = server.gui.add_tab_group()

    with tabs.add_tab("📺 播放控制"):
        # 追踪模式选择
        gui_tracking_mode = server.gui.add_dropdown(
            "🎯 追踪模式",
            options=["前景追踪", "区域框选", "单点追踪"],
            initial_value="前景追踪"
        )
        server.gui.add_markdown("**前景追踪**：显示动态物体(mask)\n\n**区域框选**：框选区域内轨迹\n\n**单点追踪**：点击选择单条轨迹")

        # 追踪模式对应的交互按钮（根据模式动态显示/隐藏）
        btn_mode_enter_click  = server.gui.add_button("进入选点模式",   visible=False)
        btn_mode_exit_click   = server.gui.add_button("退出选点模式",   visible=False)
        btn_mode_clear_click  = server.gui.add_button("清除选中点",     visible=False)
        btn_mode_enter_rect   = server.gui.add_button("进入区域框选",   visible=False)
        btn_mode_exit_rect    = server.gui.add_button("退出区域框选",   visible=False)
        btn_mode_clear_rect   = server.gui.add_button("清除区域框选",   visible=False)
        gui_mode_status       = server.gui.add_markdown("", visible=False)

        @gui_tracking_mode.on_update
        def _(_):
            mode = gui_tracking_mode.value
            is_click = (mode == "单点追踪")
            is_rect  = (mode == "区域框选")
            btn_mode_enter_click.visible = is_click
            btn_mode_exit_click.visible  = is_click
            btn_mode_clear_click.visible = is_click
            btn_mode_enter_rect.visible  = is_rect
            btn_mode_exit_rect.visible   = is_rect
            btn_mode_clear_rect.visible  = is_rect
            gui_mode_status.visible      = True
            if mode == "前景追踪":
                gui_mode_status.content = "当前：前景追踪模式"
                display_mask[:] = True
            elif mode == "区域框选":
                gui_mode_status.content = "当前：区域框选模式 — 点击「进入区域框选」后拖拽选择"
                display_mask[:] = False
            elif mode == "单点追踪":
                gui_mode_status.content = "当前：单点追踪模式 — 点击「进入选点模式」后单击轨迹"
                display_mask[:] = False

        # 播放控制
        gui_timestep = server.gui.add_slider(
            "时间步", min=0, max=num_frames - 1, step=1, initial_value=0
        )
        gui_playing = server.gui.add_checkbox("播放中", True)
        gui_framerate = server.gui.add_slider(
            "帧率", min=1, max=60, step=0.1, initial_value=24
        )

        # 外观设置
        gui_static_point_size = server.gui.add_slider(
            "静态点大小", min=0.0001, max=10, step=0.0001, initial_value=0.01
        )
        gui_dynamic_point_size = server.gui.add_slider(
            "动态点大小", min=0.0001, max=10, step=0.0001, initial_value=0.01
        )
        gui_line_width = server.gui.add_slider(
            "轨迹线宽", min=0.1, max=5.0, step=0.1, initial_value=0.5
        )
        gui_dyn_saturation = server.gui.add_slider(
            "动态点饱和度", min=0.0, max=1.0, step=0.01, initial_value=1.0
        )
        gui_show_static = server.gui.add_checkbox("显示静态背景", True)
        gui_show_dynamic = server.gui.add_checkbox("显示动态点", True)
        gui_show_traj = server.gui.add_checkbox("显示动态轨迹", True)

        @gui_dyn_saturation.on_update
        def _(_):
            t_curr = gui_timestep.value
            if t_curr < len(dynamic_point_nodes):
                dynamic_point_nodes[t_curr].colors = fade_color_saturation_batch(
                    dynamic_colors_original[t_curr],
                    gui_dyn_saturation.value
                )

        # 轨迹设置
        gui_max_traj_length = server.gui.add_slider(
            "轨迹拖尾长度", min=1, max=50, step=1, initial_value=10
        )
        gui_max_displacement = server.gui.add_slider(
            "最大位移阈值",
            min=0.1, max=20.0, step=0.1,
            initial_value=MAX_DISPLACEMENT
        )

        # 相机控制
        gui_cam_pos = server.gui.add_vector3(
            "相机位置", initial_value=DEFAULT_CAM_POS, step=0.05
        )
        gui_cam_look = server.gui.add_vector3(
            "观察点", initial_value=DEFAULT_LOOK_AT, step=0.05
        )
        gui_cam_up = server.gui.add_vector3(
            "向上方向", initial_value=DEFAULT_UP, step=0.05
        )
        btn_reset_cam = server.gui.add_button("重置相机")
        btn_sync_from_view = server.gui.add_button("同步当前视角")

        @gui_cam_pos.on_update
        def _(_):
            for client in server.get_clients().values():
                client.camera.position = gui_cam_pos.value

        @gui_cam_look.on_update
        def _(_):
            for client in server.get_clients().values():
                client.camera.look_at = gui_cam_look.value

        @gui_cam_up.on_update
        def _(_):
            for client in server.get_clients().values():
                client.camera.up_direction = gui_cam_up.value

        @btn_reset_cam.on_click
        def _(_):
            gui_cam_pos.value = DEFAULT_CAM_POS
            gui_cam_look.value = DEFAULT_LOOK_AT
            gui_cam_up.value = DEFAULT_UP
            for client in server.get_clients().values():
                client.camera.position = DEFAULT_CAM_POS
                client.camera.look_at = DEFAULT_LOOK_AT
                client.camera.up_direction = DEFAULT_UP
                client.camera.wxyz = DEFAULT_WXYZ

        @btn_sync_from_view.on_click
        def _(_):
            clients = server.get_clients()
            if clients:
                client = list(clients.values())[0]
                gui_cam_pos.value = client.camera.position
                gui_cam_look.value = client.camera.look_at
                gui_cam_up.value = client.camera.up_direction
                
    line_node = server.scene.add_line_segments(
        name="/trajectories",
        points=np.zeros((0, 2, 3)),
        colors=np.zeros((0, 2, 3), dtype=np.uint8),
        line_width=gui_line_width.value,
        visible=True,
    )

    # ==========================================================================
    # Core Update Logic (Modified)
    # ==========================================================================
    def update_scene_state(
        t_curr: int, 
        t_prev: int, 
        history_pos: List[np.ndarray], 
        history_col: List[np.ndarray],
        history_ind: List[np.ndarray]  # Stores trajectory IDs for segments
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        
        # Update static point visibility and size
        if static_node is not None:
            static_node.visible = gui_show_static.value
            static_node.point_size = gui_static_point_size.value

        # Update dynamic points visibility
        if t_curr != t_prev:
            if t_prev >= 0 and t_prev < len(dynamic_point_nodes):
                dynamic_point_nodes[t_prev].visible = False

            # 检测加载后节点列表被清空，需要重建
            if len(dynamic_point_nodes) != num_frames:
                print(f"检测到轨迹数据变化，重建场景节点（{len(dynamic_point_nodes)} -> {num_frames} 帧）")
                for node in dynamic_point_nodes:
                    node.remove()
                dynamic_point_nodes.clear()
                for t in range(num_frames):
                    pts = 100 * trajectories_3d_down[t]
                    vis = visibility_mask_down[t]
                    valid_pts = pts[vis]
                    valid_cols = initial_colors[vis]
                    node = server.scene.add_point_cloud(
                        name=f"/dynamic_frame_{t}",
                        points=valid_pts,
                        colors=valid_cols,
                        point_size=gui_dynamic_point_size.value,
                        visible=False,
                    )
                    dynamic_point_nodes.append(node)
                dynamic_colors_original.clear()
                for t in range(num_frames):
                    vis = visibility_mask_down[t]
                    dynamic_colors_original.append(initial_colors[vis].copy())
                print("场景节点重建完成")

            if gui_show_dynamic.value and t_curr < len(dynamic_point_nodes):
                node = dynamic_point_nodes[t_curr]
                node.visible = True
                node.point_size = gui_dynamic_point_size.value

                # --- Key modification: apply slider saturation before displaying each frame ---
                node.colors = fade_color_saturation_batch(
                    dynamic_colors_original[t_curr],
                    gui_dyn_saturation.value
                )
            else:
                dynamic_point_nodes[t_curr].visible = False

        # Update trajectory lines
        if gui_show_traj.value and N_traj > 0:
            line_node.visible = True
            line_node.line_width = gui_line_width.value

            # If looped back to frame 0, clear history
            if t_curr == 0 and t_prev != 0:
                history_pos.clear()
                history_col.clear()
                history_ind.clear() # Clear indices
                line_node.points = np.zeros((0, 2, 3))
                return history_pos, history_col, history_ind

            # --- 1. Calculate new segments for the current frame ---
            current_active_indices = np.array([], dtype=int) 

            if t_curr < num_frames and t_curr > 0:
                pos_curr = 100 * trajectories_3d_down[t_curr - 1]
                pos_next = 100 * trajectories_3d_down[t_curr]
                
                # Get visibility mask
                valid_mask = (
                    visibility_mask_down[t_curr - 1] & 
                    visibility_mask_down[t_curr]
                )
                
                if np.any(valid_mask):
                    # Get original indices (0 to N_traj-1)
                    all_indices = np.arange(N_traj)

                    # Filter out deleted trajectories
                    active_mask = valid_mask & ~deleted_trajectories

                    # 根据追踪模式的 display_mask 筛选（前景模式全显示，单点/区域模式按选中筛选）
                    if gui_tracking_mode.value != "前景追踪":
                        active_mask = active_mask & display_mask

                    # Initial filtering
                    p1 = pos_curr[active_mask]
                    p2 = pos_next[active_mask]
                    curr_inds = all_indices[active_mask]

                    # Calculate displacement and filter jumps
                    dist = np.linalg.norm(p2 - p1, axis=1)
                    jump_mask = dist < gui_max_displacement.value

                    if np.any(jump_mask):
                        final_p1 = p1[jump_mask]
                        final_p2 = p2[jump_mask]
                        segments = np.stack([final_p1, final_p2], axis=1)

                        final_indices = curr_inds[jump_mask]

                        # Use white for selected trajectories, original color otherwise
                        base_cols = initial_colors[active_mask][jump_mask].copy()
                        is_selected = selected_trajectories[final_indices]
                        base_cols[is_selected] = [255, 255, 255]
                        segment_colors = np.stack([base_cols, base_cols], axis=1)

                        history_pos.append(segments)
                        history_col.append(segment_colors)
                        history_ind.append(final_indices)

                        current_active_indices = final_indices

            # Maintain maximum trajectory length
            while len(history_pos) > gui_max_traj_length.value:
                history_pos.pop(0)
                history_col.pop(0)
                history_ind.pop(0)
            
            # --- 2. Rendering Logic (Critical Modification) ---
            # Only historical segments of trajectories present in 
            # current_active_indices will be displayed
            if history_pos and len(current_active_indices) > 0:
                
                # Create a boolean lookup table to quickly check if historical 
                # segments belong to currently active trajectories
                active_lookup = np.zeros(N_traj, dtype=bool)
                active_lookup[current_active_indices] = True
                
                render_pos_list = []
                render_col_list = []
                num_history_steps = len(history_pos)
                # Iterate through history, keeping only segments of active trajectories
                for i, (h_pos, h_col, h_ind) in enumerate(zip(history_pos, history_col, history_ind)):
                    keep_mask = active_lookup[h_ind]
                    
                    if np.any(keep_mask):
                        # 1. Calculate Alpha factor (from 0.0 to 1.0)
                        # The smaller i (older), the lower alpha, making lines more transparent
                        alpha_factor = 1 - i / num_history_steps
                        
                        # 2. Get original RGB
                        rgb_cols = h_col[keep_mask] # Shape (M, 2, 3)
                        if rgb_cols.shape[0] > 1:
                            faded_col = fade_color_saturation(rgb_cols, alpha_factor)
                        else:
                            faded_col = rgb_cols
                        # Concatenate to (M, 2, 4)
                        rgba_cols = faded_col
                        
                        render_pos_list.append(h_pos[keep_mask])
                        render_col_list.append(rgba_cols)
                
                if render_pos_list:
                    line_node.points = np.concatenate(render_pos_list, axis=0)
                    line_node.colors = np.concatenate(render_col_list, axis=0)
                else:
                    line_node.points = np.zeros((0, 2, 3))
            else:
                # If no active trajectories or history is empty, do not show lines
                line_node.points = np.zeros((0, 2, 3))
        else:
            line_node.visible = False
            
        return history_pos, history_col, history_ind

    # ==========================================================================
    # Tab 2: 编辑与导出
    # ==========================================================================
    deleted_trajectories  = np.zeros(N_traj, dtype=bool)
    selected_trajectories = np.zeros(N_traj, dtype=bool)
    display_mask          = np.ones(N_traj, dtype=bool)   # 追踪模式显示筛选，默认全显示

    # 加载mask数据用于前景模式
    mask_data_available = False
    pc_dyn_masks = []
    mask_dir = ply_dir
    for i in range(num_frames):
        mask_path = mask_dir / f"pc_dyn_mask_{i:03d}.npy"
        if not mask_path.exists():
            mask_path = mask_dir / f"pc_dyn_mask_{i}.npy"
        if mask_path.exists():
            mask_data_available = True
            pc_dyn_masks.append(np.load(str(mask_path)))
        else:
            pc_dyn_masks.append(None)

    with tabs.add_tab("✏️ 编辑与导出"):

        # 交互模式选择
        with server.gui.add_folder("🖱️ 交互模式"):
            btn_rect_select = server.gui.add_button("进入框选模式")
            btn_click_select = server.gui.add_button("进入选点模式")
            btn_exit_pointer = server.gui.add_button("退出交互模式")
            gui_pointer_status = server.gui.add_markdown("当前模式：正常旋转")

        # 轨迹修正
        with server.gui.add_folder("🔧 轨迹修正"):
            gui_pinned_id  = server.gui.add_markdown("选中轨迹：无")
            gui_locate_id  = server.gui.add_number("按ID定位", initial_value=0, min=0, max=max(N_traj-1,0), step=1)
            btn_locate_id  = server.gui.add_button("定位该轨迹")

            gui_fix_frame  = server.gui.add_slider("修正帧", min=0, max=num_frames - 1, step=1, initial_value=0)

            # 新增：点击选择修正位置模式
            btn_pick_position = server.gui.add_button("🎯 点击选择修正位置")
            gui_fix_xyz    = server.gui.add_vector3("目标位置 (XYZ)", initial_value=(0.0, 0.0, 0.0), step=0.5)
            server.gui.add_markdown("💡 提示：点击上方按钮后，在场景中点击目标位置")

            btn_add_pin    = server.gui.add_button("添加修正点")
            btn_apply_fix  = server.gui.add_button("应用插值修正")
            btn_clear_pins = server.gui.add_button("清除修正点")
            gui_pins_info  = server.gui.add_markdown("待应用的修正点：无")

        # 区域管理
        with server.gui.add_folder("📦 区域管理"):
            btn_delete       = server.gui.add_button("删除选中轨迹")
            btn_keep_only    = server.gui.add_button("只保留选中轨迹")
            btn_restore      = server.gui.add_button("恢复全部轨迹")
            gui_stats = server.gui.add_markdown(
                f"**统计**：总计 {N_traj} | 已选 0 | 已删 0 | 剩余 {N_traj}"
            )

        # 导出
        with server.gui.add_folder("💾 导出"):
            btn_export   = server.gui.add_button("导出剩余轨迹 (.npy)")
            btn_save_vis = server.gui.add_button("保存 .viser 文件")


        # ---- 状态变量 ----
        pinned_traj_id = [-1]
        pin_keyframes  = []
        picking_position = [False]  # 是否处于选择修正位置模式

        # 红点：显示目标修正位置
        target_dot = server.scene.add_icosphere(
            name="/target_dot",
            radius=0.3,
            color=(255, 0, 0),
            position=(0.0, 0.0, 0.0),
            visible=False,
        )

        def _update_target_dot():
            if pinned_traj_id[0] >= 0:
                xyz = gui_fix_xyz.value
                target_dot.position = tuple(float(v) for v in xyz)
                target_dot.visible = True
            else:
                target_dot.visible = False

        @gui_fix_xyz.on_update
        def _(_): _update_target_dot()

        @gui_fix_frame.on_update
        def _(_):
            tid = pinned_traj_id[0]
            if tid >= 0:
                f = gui_fix_frame.value
                vis = visibility_mask_down[f]
                if vis[tid]:
                    pos = 100 * trajectories_3d_down[f, tid]
                    gui_fix_xyz.value = tuple(float(v) for v in pos)

        @btn_locate_id.on_click
        def _(_):
            tid = int(gui_locate_id.value)
            if tid < 0 or tid >= N_traj:
                print(f"轨迹 ID {tid} 超出范围")
                return
            pinned_traj_id[0] = tid
            selected_trajectories[:] = False
            selected_trajectories[tid] = True
            f = gui_timestep.value
            vis = visibility_mask_down[f]
            if vis[tid]:
                pos = 100 * trajectories_3d_down[f, tid]
                gui_fix_xyz.value = tuple(float(v) for v in pos)
            gui_pinned_id.content = f"选中轨迹 ID：**{tid}**"
            gui_fix_frame.value = f
            _update_target_dot()
            _update_stats()
            print(f"已定位轨迹 {tid}")

        def _ray_to_nearest_traj(ray_origin, ray_direction, frame_idx, max_dist=5.0):
            """射线与当前帧轨迹点做最近距离匹配，返回轨迹 ID"""
            o = np.array(ray_origin)
            d = np.array(ray_direction)
            d = d / (np.linalg.norm(d) + 1e-9)
            vis = visibility_mask_down[frame_idx]
            pts = 100 * trajectories_3d_down[frame_idx]  # (N, 3)
            best_id, best_dist = -1, float('inf')
            for i in range(pts.shape[0]):
                if not vis[i] or deleted_trajectories[i]:
                    continue
                p = pts[i] - o
                t = np.dot(p, d)
                if t < 0:
                    continue
                closest = o + t * d
                dist = np.linalg.norm(pts[i] - closest)
                if dist < best_dist:
                    best_dist, best_id = dist, i
            return best_id, best_dist

        def _update_stats():
            n_sel = int(selected_trajectories.sum())
            n_del = int(deleted_trajectories.sum())
            gui_stats.content = f"**统计**：总计 {N_traj} | 已选 {n_sel} | 已删 {n_del} | 剩余 {N_traj - n_del}"

        @btn_rect_select.on_click
        def _(_):
            gui_pointer_status.content = "当前模式：**框选模式**（在场景中拖拽鼠标框选轨迹）"
            @server.scene.on_pointer_event("rect-select")
            def _(event):
                nonlocal selected_trajectories
                if len(event.screen_pos) < 2:
                    return
                (sx0, sy0), (sx1, sy1) = event.screen_pos[0], event.screen_pos[1]
                x0, x1 = min(sx0, sx1), max(sx0, sx1)
                y0, y1 = min(sy0, sy1), max(sy0, sy1)
                print(f"框选范围: x=[{x0:.3f},{x1:.3f}] y=[{y0:.3f},{y1:.3f}]")
                frame_idx = gui_timestep.value
                vis = visibility_mask_down[frame_idx]
                pts_world = 100 * trajectories_3d_down[frame_idx]
                client = event.client
                cam = client.camera
                cam_pos  = np.array(cam.position)
                cam_look = np.array(cam.look_at)
                cam_up   = np.array(cam.up_direction)
                fwd = cam_look - cam_pos
                fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
                right = np.cross(fwd, cam_up)
                right = right / (np.linalg.norm(right) + 1e-9)
                up = np.cross(right, fwd)
                fov = cam.fov  # radians
                aspect = cam.aspect
                selected_trajectories[:] = False
                hit = 0
                for i in range(pts_world.shape[0]):
                    if not vis[i] or deleted_trajectories[i]:
                        continue
                    p = pts_world[i] - cam_pos
                    z = np.dot(p, fwd)
                    if z <= 0:
                        continue
                    x_s = np.dot(p, right) / (z * np.tan(fov / 2) * aspect)
                    y_s = -np.dot(p, up) / (z * np.tan(fov / 2))
                    sx = (x_s + 1) / 2
                    sy = (y_s + 1) / 2
                    if x0 <= sx <= x1 and y0 <= sy <= y1:
                        selected_trajectories[i] = True
                        hit += 1
                print(f"投影检查: 共{np.sum(vis)}个可见点, 命中{hit}个")
                _update_stats()
                print(f"框选完成，选中 {int(selected_trajectories.sum())} 条轨迹")
            print("已进入框选模式，在场景中拖拽鼠标框选")

        @btn_click_select.on_click
        def _(_):
            picking_position[0] = False  # 确保不是选择修正位置模式
            gui_pointer_status.content = "当前模式：**选点模式**（在场景中单击轨迹点）"
            @server.scene.on_pointer_event("click")
            def _(event):
                if event.ray_origin is None:
                    return
                frame_idx = gui_timestep.value
                best_id, best_dist = _ray_to_nearest_traj(
                    event.ray_origin, event.ray_direction, frame_idx
                )
                if best_id < 0:
                    print("未找到附近的轨迹点")
                    return
                pinned_traj_id[0] = best_id
                selected_trajectories[:] = False
                selected_trajectories[best_id] = True
                pos = 100 * trajectories_3d_down[frame_idx, best_id]
                gui_pinned_id.content = f"选中轨迹 ID：**{best_id}**（射线距离 {best_dist:.2f}）"
                gui_fix_frame.value = frame_idx
                gui_fix_xyz.value = tuple(float(v) for v in pos)
                _update_stats()
                print(f"已选中轨迹 {best_id}，射线距离 {best_dist:.2f}")
            print("已进入选点模式，在场景中单击轨迹点")

        @btn_pick_position.on_click
        def _(_):
            if pinned_traj_id[0] < 0:
                print("请先选择一条轨迹")
                return
            picking_position[0] = True
            gui_pointer_status.content = "当前模式：**选择修正位置**（点击场景中的目标位置）"

            @server.scene.on_pointer_event("click")
            def _(event):
                if event.ray_origin is None or not picking_position[0]:
                    return

                # 直接使用射线与平面求交，获取点击的3D坐标
                frame_idx = gui_timestep.value
                o = np.array(event.ray_origin)
                d = np.array(event.ray_direction)
                d = d / (np.linalg.norm(d) + 1e-9)

                # 计算当前帧可见点的平均深度，作为交点平面
                vis = visibility_mask_down[frame_idx]
                pts_dyn = 100 * trajectories_3d_down[frame_idx]
                visible_pts = pts_dyn[vis]

                if len(visible_pts) == 0:
                    print("当前帧没有可见点，无法确定深度")
                    return

                # 使用可见点的平均位置作为平面中心
                plane_center = np.mean(visible_pts, axis=0)

                # 使用相机朝向作为平面法向量
                clients = server.get_clients()
                if clients:
                    client = list(clients.values())[0]
                    cam_pos = np.array(client.camera.position)
                    cam_look = np.array(client.camera.look_at)
                    plane_normal = cam_look - cam_pos
                    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-9)
                else:
                    # 默认使用Z轴作为法向量
                    plane_normal = np.array([0, 0, 1])

                # 计算射线与平面的交点
                # 平面方程: (P - plane_center) · plane_normal = 0
                # 射线方程: P = o + t * d
                # 求解: (o + t * d - plane_center) · plane_normal = 0
                denom = np.dot(d, plane_normal)
                if abs(denom) < 1e-6:
                    print("射线与平面平行，无法求交点")
                    return

                t = np.dot(plane_center - o, plane_normal) / denom
                if t < 0:
                    print("交点在射线反方向，无效")
                    return

                # 计算交点坐标
                intersection_pt = o + t * d

                gui_fix_xyz.value = tuple(float(v) for v in intersection_pt)
                target_dot.position = tuple(float(v) for v in intersection_pt)
                target_dot.visible = True
                picking_position[0] = False
                gui_pointer_status.content = "当前模式：正常旋转"
                server.scene.remove_pointer_callback()
                print(f"已选择修正位置: ({intersection_pt[0]:.2f}, {intersection_pt[1]:.2f}, {intersection_pt[2]:.2f})")

            print("已进入选择修正位置模式，点击场景中的目标位置")

        @btn_exit_pointer.on_click
        def _(_):
            server.scene.remove_pointer_callback()
            gui_pointer_status.content = "当前模式：正常旋转"
            print("已退出交互模式")

        # ---- 追踪模式按钮事件处理 ----
        @btn_mode_enter_click.on_click
        def _(_):
            gui_mode_status.content = "单点追踪：**选点模式**（在场景中单击轨迹点）"
            @server.scene.on_pointer_event("click")
            def _(event):
                if event.ray_origin is None:
                    return
                frame_idx = gui_timestep.value
                best_id, best_dist = _ray_to_nearest_traj(
                    event.ray_origin, event.ray_direction, frame_idx
                )
                if best_id < 0:
                    return
                display_mask[best_id] = True
                gui_mode_status.content = f"单点追踪：已选中 {int(display_mask.sum())} 条轨迹（点击继续添加）"
                print(f"追踪模式：已添加轨迹 {best_id}，共 {int(display_mask.sum())} 条")
            print("追踪模式：进入选点模式")

        @btn_mode_clear_click.on_click
        def _(_):
            display_mask[:] = False
            server.scene.remove_pointer_callback()
            gui_mode_status.content = "单点追踪：已清除所有选中点"
            print("追踪模式：已清除所有选中点")

        @btn_mode_exit_click.on_click
        def _(_):
            server.scene.remove_pointer_callback()
            gui_mode_status.content = "单点追踪：已退出选点模式（正常旋转）"
            print("追踪模式：已退出选点模式")

        @btn_mode_enter_rect.on_click
        def _(_):
            gui_mode_status.content = "区域框选：**框选模式**（在场景中拖拽鼠标框选区域）"
            @server.scene.on_pointer_event("rect-select")
            def _(event):
                if len(event.screen_pos) < 2:
                    return
                (sx0, sy0), (sx1, sy1) = event.screen_pos[0], event.screen_pos[1]
                x0, x1 = min(sx0, sx1), max(sx0, sx1)
                y0, y1 = min(sy0, sy1), max(sy0, sy1)
                frame_idx = gui_timestep.value
                vis = visibility_mask_down[frame_idx]
                pts_world = 100 * trajectories_3d_down[frame_idx]
                client = event.client
                cam = client.camera
                cam_pos  = np.array(cam.position)
                cam_look = np.array(cam.look_at)
                cam_up   = np.array(cam.up_direction)
                fwd = cam_look - cam_pos
                fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
                right = np.cross(fwd, cam_up)
                right = right / (np.linalg.norm(right) + 1e-9)
                up = np.cross(right, fwd)
                fov = cam.fov
                aspect = cam.aspect
                hit = 0
                for i in range(pts_world.shape[0]):
                    if not vis[i] or deleted_trajectories[i]:
                        continue
                    p = pts_world[i] - cam_pos
                    z = np.dot(p, fwd)
                    if z <= 0:
                        continue
                    x_s = np.dot(p, right) / (z * np.tan(fov / 2) * aspect)
                    y_s = -np.dot(p, up) / (z * np.tan(fov / 2))
                    sx = (x_s + 1) / 2
                    sy = (y_s + 1) / 2
                    if x0 <= sx <= x1 and y0 <= sy <= y1:
                        display_mask[i] = True
                        hit += 1
                gui_mode_status.content = f"区域框选：已选中 {int(display_mask.sum())} 条轨迹（可继续框选叠加）"
                print(f"追踪模式：框选完成，新增 {hit} 条，共 {int(display_mask.sum())} 条")
            print("追踪模式：进入区域框选模式")

        @btn_mode_clear_rect.on_click
        def _(_):
            display_mask[:] = False
            server.scene.remove_pointer_callback()
            gui_mode_status.content = "区域框选：已清除所有框选区域"
            print("追踪模式：已清除所有框选区域")

        @btn_mode_exit_rect.on_click
        def _(_):
            server.scene.remove_pointer_callback()
            gui_mode_status.content = "区域框选：已退出框选模式（正常旋转）"
            print("追踪模式：已退出区域框选模式")


        def _(_):
            if pinned_traj_id[0] < 0:
                print("请先在选点模式下点击一条轨迹")
                return
            frame = gui_fix_frame.value
            xyz   = tuple(float(v) for v in gui_fix_xyz.value)
            pin_keyframes.append((pinned_traj_id[0], frame, xyz))
            lines = [f"轨迹{tid} 帧{f}: ({x:.1f},{y:.1f},{z:.1f})"
                     for tid, f, (x, y, z) in pin_keyframes]
            gui_pins_info.content = "待应用的修正点：\n" + "\n".join(lines)
            # 添加修正点后清除高亮和红点
            selected_trajectories[:] = False
            target_dot.visible = False
            print(f"已添加修正点：轨迹 {pinned_traj_id[0]} 帧{frame} → {xyz}")

        @btn_apply_fix.on_click
        def _(_):
            from scipy.interpolate import CubicSpline
            from collections import defaultdict
            if not pin_keyframes:
                print("没有修正点可应用")
                return
            traj_pins = defaultdict(list)
            for tid, frame, xyz in pin_keyframes:
                traj_pins[tid].append((frame, xyz))
            for tid, pins in traj_pins.items():
                pins.sort(key=lambda x: x[0])
                if len(pins) < 2:
                    print(f"轨迹 {tid} 修正点不足2个，跳过")
                    continue
                frames = np.array([p[0] for p in pins])
                coords = np.array([p[1] for p in pins])
                f0, f1 = int(frames[0]), int(frames[-1])
                t_range = np.arange(f0, f1 + 1)
                for axis in range(3):
                    cs = CubicSpline(frames, coords[:, axis])
                    trajectories_3d_down[f0:f1+1, tid, axis] = cs(t_range) / 100.0

                # 强制更新修正帧范围内的点云颜色
                for f in range(f0, f1 + 1):
                    if f < len(dynamic_point_nodes):
                        vis = visibility_mask_down[f]
                        if vis[tid]:
                            pts = 100 * trajectories_3d_down[f]
                            dynamic_point_nodes[f].points = pts[vis]
                            dynamic_colors_original[f] = initial_colors[vis].copy()
                            dynamic_point_nodes[f].colors = initial_colors[vis]

                print(f"轨迹 {tid} 帧{f0}-{f1} 插值修正完成")
            pin_keyframes.clear()
            pinned_traj_id[0] = -1
            selected_trajectories[:] = False
            target_dot.visible = False
            clear_trail_flag[0] = True  # 通知主循环清空拖尾缓存
            gui_pinned_id.content = "选中轨迹：无"
            gui_pins_info.content = "待应用的修正点：无"
            _update_stats()

        @btn_clear_pins.on_click
        def _(_):
            pin_keyframes.clear()
            pinned_traj_id[0] = -1
            gui_pinned_id.content = "已标记轨迹：无"
            gui_pins_info.content = "待应用的修正点：无"
            print("已清除所有修正点")

        # 选择框场景节点
        box_node = server.scene.add_box(
            name="/selection_box",
            color=(255, 255, 0),
            dimensions=(1.0, 1.0, 1.0),
            wireframe=True,
            position=(0.0, 0.0, 0.0),
            visible=False
        )

        def update_selection_box():
            box_node.visible = False

        @btn_delete.on_click
        def _(_):
            nonlocal deleted_trajectories
            n_before = int(deleted_trajectories.sum())
            deleted_trajectories |= selected_trajectories
            selected_trajectories[:] = False
            n_after = int(deleted_trajectories.sum())
            gui_stats.content = f"**统计**：总计 {N_traj} | 已选 0 | 已删 {n_after} | 剩余 {N_traj - n_after}"
            print(f"已删除 {n_after - n_before} 条轨迹（累计删除 {n_after} 条）")

        @btn_keep_only.on_click
        def _(_):
            nonlocal deleted_trajectories
            # 把未选中的轨迹全部标记为删除
            deleted_trajectories |= ~selected_trajectories
            selected_trajectories[:] = False
            n_after = int(deleted_trajectories.sum())
            gui_stats.content = f"**统计**：总计 {N_traj} | 已选 0 | 已删 {n_after} | 剩余 {N_traj - n_after}"
            print(f"只保留选中轨迹，已隐藏 {n_after} 条")

        @btn_restore.on_click
        def _(_):
            nonlocal deleted_trajectories, selected_trajectories
            n = int(deleted_trajectories.sum())
            deleted_trajectories[:] = False
            selected_trajectories[:] = False
            gui_stats.content = f"**统计**：总计 {N_traj} | 已选 0 | 已删 0 | 剩余 {N_traj}"
            print(f"已恢复 {n} 条轨迹")

        @btn_export.on_click
        def _(_):
            if N_traj == 0:
                return
            remaining_mask = ~deleted_trajectories
            n_rem = int(remaining_mask.sum())
            if n_rem == 0:
                print("没有剩余轨迹可导出")
                return
            rem_idx = np.where(remaining_mask)[0] * DEFAULT_POINT_DOWNSAMPLE_RATE + 3

            # 保存轨迹和mask
            np.save(str(save_dir / "edited_trajectories.npy"), trajectories_3d[:, rem_idx])
            np.save(str(save_dir / "edited_visibility_mask.npy"), visibility_mask[:, rem_idx])

            # 保存场景信息（用于重新加载）
            c2w_path = ply_dir / "c2w.npy"
            if c2w_path.exists():
                c2w_data = np.load(str(c2w_path))
                np.save(str(save_dir / "c2w.npy"), c2w_data)

            # 读取第一帧 PLY 获取静态点云
            first_ply = sorted(ply_dir.glob("frame_*.ply"))[0]
            pcd = o3d.io.read_point_cloud(str(first_ply))
            pts = np.asarray(pcd.points)
            cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            np.savez_compressed(
                str(save_dir / "static_points.npz"),
                xyz=pts,
                rgb=cols
            )

            with open(save_dir / "edited_info.txt", 'w', encoding='utf-8') as f:
                f.write(f"原始轨迹数: {N_traj}\n已删除: {int(deleted_trajectories.sum())}\n剩余: {n_rem}\n")
                f.write(f"\n场景信息已保存，可用于重新加载：\n")
                f.write(f"- edited_trajectories.npy\n")
                f.write(f"- edited_visibility_mask.npy\n")
                f.write(f"- c2w.npy (相机参数)\n")
                f.write(f"- static_points.npz (静态点云)\n")

            print(f"已导出 {n_rem} 条轨迹到 {save_dir}/")
            print(f"场景信息已保存，可在 Gradio 展示模式中重新加载")

        @btn_save_vis.on_click
        def _(_):
            print("开始录制 .viser 文件...")
            was_playing = gui_playing.value
            gui_playing.value = False
            for node in dynamic_point_nodes:
                node.visible = False
            line_node.points = np.zeros((0, 2, 3))
            serializer = server.get_scene_serializer()
            rec_hp, rec_hc, rec_hi, rec_prev = [], [], [], -1
            try:
                for t in tqdm(range(num_frames), desc="录制中"):
                    rec_hp, rec_hc, rec_hi = update_scene_state(
                        t_curr=t, t_prev=rec_prev,
                        history_pos=rec_hp, history_col=rec_hc, history_ind=rec_hi
                    )
                    rec_prev = t
                    serializer.insert_sleep(1.0 / gui_framerate.value)
                data = serializer.serialize()
                if gui_show_dynamic.value and gui_show_traj.value:
                    fname = "pc_line.viser"
                elif gui_show_traj.value:
                    fname = "line.viser"
                elif gui_show_dynamic.value:
                    fname = "pc.viser"
                else:
                    fname = "output.viser"
                (save_dir / fname).write_bytes(data)
                print(f"已保存 {save_dir / fname}（{len(data)/1024/1024:.2f} MB）")
            except Exception as e:
                print(f"录制失败: {e}")
            finally:
                gui_playing.value = was_playing
                gui_timestep.value = 0


        update_selection_box()

    # 共享标志：通知主循环清空拖尾缓存
    clear_trail_flag = [False]

    # ==========================================================================
    # Main Loop
    # ==========================================================================
    prev_timestep = -1
    live_history_pos = []
    live_history_col = []
    live_history_ind = [] 
    print_counter = 0

    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        # 修正后清空拖尾缓存，避免旧坐标残留
        if clear_trail_flag[0]:
            live_history_pos.clear()
            live_history_col.clear()
            live_history_ind.clear()
            line_node.points = np.zeros((0, 2, 3))
            clear_trail_flag[0] = False

        t_curr = gui_timestep.value

        # <--- Update call parameters, receiving 3 return values
        live_history_pos, live_history_col, live_history_ind = update_scene_state(
            t_curr=t_curr,
            t_prev=prev_timestep,
            history_pos=live_history_pos,
            history_col=live_history_col,
            history_ind=live_history_ind 
        )
        
        # Camera state printing removed (was too noisy in terminal)

        prev_timestep = t_curr
        time.sleep(1.0 / gui_framerate.value)
 
if __name__ == "__main__":
    main()
