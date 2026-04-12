"""诊断 Viser mask 对齐问题"""
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from scipy.spatial import cKDTree

ply_dir = Path("E:/bishe2/Track4World/results/human_grasp_3d_efep_masked/3d_efep_output")
mask_dir = Path("E:/bishe2/Track4World/results/human_grasp_3d_efep_masked/mask")
video_path = Path("E:/bishe2/embodied_data/human_grasp/human_grasp_video0.mp4")

# 测试第 10 帧
frame_idx = 10

# 读取 frame PLY
frame_pcd = o3d.io.read_point_cloud(str(ply_dir / f"frame_{frame_idx:03d}.ply"))
frame_cols = np.asarray(frame_pcd.colors) * 255

# 读取视频帧
cap = cv2.VideoCapture(str(video_path))
for _ in range(frame_idx): cap.read()
ret, frame = cap.read()
cap.release()

frame_resized = cv2.resize(frame, (448, 320))
frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
frame_flat_rgb = frame_rgb.reshape(-1, 3).astype(np.float32)

# 读取 mask
mask_img = cv2.imread(str(mask_dir / f"mask_{frame_idx:04d}.png"), cv2.IMREAD_GRAYSCALE)
mask_resized = cv2.resize(mask_img, (448, 320), interpolation=cv2.INTER_NEAREST)
mask_flat = (mask_resized.flatten() > 127).astype(np.float32)

print(f"Frame {frame_idx}:")
print(f"  frame PLY points: {len(frame_cols)}")
print(f"  video pixels: {len(frame_flat_rgb)}")
print(f"  mask dynamic ratio: {mask_flat.mean():.4f}")

# 颜色匹配
tree = cKDTree(frame_flat_rgb)
distances, indices = tree.query(frame_cols, k=1)

matched = (distances < 5).sum()
print(f"  color match: {matched}/{len(frame_cols)} ({matched/len(frame_cols)*100:.1f}%)")

# 查询 mask
frame_dyn_mask = mask_flat[indices]
print(f"  PLY dynamic points: {frame_dyn_mask.sum():.0f}/{len(frame_dyn_mask)} ({frame_dyn_mask.mean()*100:.2f}%)")

# 检查动态点的空间分布
dyn_pts = np.asarray(frame_pcd.points)[frame_dyn_mask > 0.5]
if len(dyn_pts) > 0:
    print(f"  dynamic points bbox: X=[{dyn_pts[:,0].min():.3f}, {dyn_pts[:,0].max():.3f}], "
          f"Y=[{dyn_pts[:,1].min():.3f}, {dyn_pts[:,1].max():.3f}]")
else:
    print("  NO dynamic points found!")

# 检查 flow PLY
flow_pcd = o3d.io.read_point_cloud(str(ply_dir / f"flow_{frame_idx:03d}.ply"))
flow_cols = np.asarray(flow_pcd.colors) * 255
print(f"\n  flow PLY points: {len(flow_cols)}")

# flow 继承 frame 的 mask
frame_color_to_dyn = {}
for j, c in enumerate(frame_cols):
    key = tuple(np.round(c).astype(int))
    frame_color_to_dyn[key] = frame_dyn_mask[j]

flow_dyn_mask = np.array([frame_color_to_dyn.get(tuple(np.round(c).astype(int)), 0.0) for c in flow_cols])
print(f"  flow dynamic points: {flow_dyn_mask.sum():.0f}/{len(flow_dyn_mask)} ({flow_dyn_mask.mean()*100:.2f}%)")
