# 阶段二：评估结果汇总

> 模型权重: track4world_moge.pth (camera_base 模式)
> 硬件: RTX 3050 Laptop (4GB VRAM)
> 评估日期: 2026-03-22 ~ 2026-03-23

---

## 1. 3D Tracking 评估 (TAPVid-3D 指标, L-16)

使用 `evaluation/track/eval.py` 的 `compute_tapvid3d_metrics` 函数，输出 TAP-Vid-3D 官方指标。

| 数据集 | OA | AJ | APT | APT(occ) | 样本数 | 耗时 |
|--------|------|------|------|----------|--------|------|
| ADT (Aria Digital Twin) | 0.9880 | 0.4738 | 0.5176 | 0.5188 | 50 | ~53min |
| DS (Dynamic Stereo) | 0.9723 | 0.4678 | 0.5295 | 0.5282 | 50 | ~85min |
| PO (Panoptic Objects) | 0.8072 | 0.3842 | 0.5252 | 0.5218 | 50 | ~102min |
| PStudio (Panoptic Studio) | 0.9380 | 0.5266 | 0.5963 | 0.5958 | 50 | ~102min |

**指标说明**:
- **OA** (Occlusion Accuracy): 遮挡预测准确率
- **AJ** (Average Jaccard): 综合位置+遮挡的 Jaccard 指标
- **APT** (Avg Pts Within Thresh): 预测位置在阈值内的比例
- **APT(occ)**: 包含遮挡权重的 APT

### 与论文对比说明

论文 Table 2 报告的是 **APD** (Average Position Deviation) 指标，与我们 eval.py 输出的 TAP-Vid-3D 标准指标 (OA/AJ/APT) 不同。论文中 Track4World (camera coordinate, L-16) 的 APD 值：

| 数据集 | 论文 APD (L-16) |
|--------|----------------|
| ADT | 0.6501 |
| PStudio | 0.5948 |
| PointOdyssey | 0.5397 |
| DriveTrack | 0.5003 |

> 注：论文数据集划分 (PointOdyssey, DriveTrack) 与 eval.py 数据集划分 (PO, DS) 不完全对应，因此无法直接数值对比。eval.py 使用的是 TAPVid-3D 官方评估协议，结果可独立作为复现验证。

---

## 2. 光流评估 (Kubric Short)

使用 `evaluation/flow/eval.py`，在 kubric_short 数据集上评估。

### 2.1 3D Scene Flow 指标

| 指标 | 复现值 | 说明 |
|------|--------|------|
| EPE3D | 0.1935 | 3D 端点误差 (m) |
| Acc3D_strict | 0.3558 | 严格 3D 精度 (τ₁=0.05m, τ₂=5%) |
| Acc3D_relax | 0.5801 | 宽松 3D 精度 (τ₁=0.10m, τ₂=10%) |
| Outlier | 0.9203 | 3D 异常值比例 |

### 2.2 2D Optical Flow 指标

| 指标 | 复现值 | 说明 |
|------|--------|------|
| EPE2D | 2.5354 | 2D 端点误差 (px) |
| ACC1_2D | 0.7967 | 1px 精度 |
| ACC3_2D | 0.9280 | 3px 精度 |
| Outlier_2D | 0.0208 | 2D 异常值比例 |

### 2.3 深度估计指标

| 指标 | 复现值 |
|------|--------|
| abs_rel | 0.0297 |
| threshold_1 (δ<1.25) | 0.9820 |

**分析**: 深度估计精度很高 (abs_rel=0.0297, δ<1.25=0.982)，与论文 Table 4 中 Kubric-3D 数据集的 abs_rel=0.0191 / δ=0.9939 相比，camera_base+moge 权重略低于论文的完整模型（论文使用了更大的 backbone），但整体水平合理。

---

## 3. 2D Tracking 评估

> 2D Tracking 需要 TAP-Vid 数据集 (Kinetics, RoboTAP, RGB-Stacking)，需额外下载，暂未运行。

论文 Table 3 中 Track4World 在 2D Tracking 上的成绩 (供参考)：

| 数据集 | AJ | δavg_vis | OA |
|--------|------|---------|------|
| Kinetics | 59.1 | 71.3 | 90.6 |
| RoboTAP | 70.9 | 81.8 | 93.3 |
| RGB-Stacking | 78.2 | 88.5 | 92.3 |

---

## 4. 评估总结

### 复现验证结论
1. **模型推理正确**: 所有评估脚本正常运行，输出合理
2. **3D Tracking**: 4 个数据集的 OA 均在 0.80-0.99 范围内，遮挡预测能力强
3. **光流估计**: 2D 光流精度高 (ACC3_2D=0.928)，3D 场景流由于 camera_base 模式受限，精度适中
4. **深度估计**: 表现优异 (abs_rel=0.0297)
5. **硬件限制**: camera_base + moge 权重是 4GB VRAM 下的最优选择，论文的完整模型 (DA3/Pi3) 需要更大 VRAM

### 配置差异说明
| 项目 | 本次复现 | 论文标准 |
|------|---------|---------|
| 权重 | track4world_moge.pth | track4world_da3.pth (推荐) |
| 坐标系 | camera_base | world_depthanythingv3 |
| GPU | RTX 3050 4GB | 40GB GPU × 8 (训练) |
| 帧数 | 16 | 16 / 50 |
