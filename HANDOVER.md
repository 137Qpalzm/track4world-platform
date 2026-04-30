# HANDOVER.md — 毕设2 工作交接文档

> 最后更新: 2026-05-01
> 状态: **Viser 前端全部完成，进入论文撰写阶段**

---

## 当前工作重点（2026-05-01）

### 最新完成（2026-05-01）— Viser 前端最终优化

1. ✅ **两个Tab布局**
   - Tab1 - 📺 播放控制：追踪模式选择 + 播放/外观/轨迹/相机控制
   - Tab2 - ✏️ 编辑与导出：交互模式 + 轨迹修正 + 区域管理 + 导出

2. ✅ **点击选择修正位置**（射线与平面求交，直接获取3D坐标，不再手动输入XYZ）

3. ✅ **追踪模式**（Tab1内联动，无需跳Tab）
   - 前景追踪：默认显示所有动态轨迹
   - 单点追踪：切换后出现「进入选点模式」/「退出选点模式」/「清除选中点」
   - 区域框选：切换后出现「进入区域框选」/「退出区域框选」/「清除区域框选」
   - 独立 `display_mask` 数组控制显示，与编辑功能的 `selected_trajectories` 完全分离

4. ✅ **相机初始参数调整**（场景向左偏移，减少左侧空白）
   - 位置：`(1.2, 2.5, 0.0)`，观察点：`(-1.5, 0.0, 0.0)`，向上：`(-0.08, 0.0, 0.4)`

### 历史已完成（2026-04-25）
- Viser GUI 中文化、框选/选点/轨迹修正/ID定位/删除恢复/导出等核心功能

### 待完成
1. ⏳ **论文撰写** — 技术方案、实验结果、系统设计
2. ⏳ **展示视频录制** — 基于 100 帧数据录制演示
3. ⏳ **答辩准备** — PPT、演示视频、Q&A

---

## 核心文件

| 文件 | 说明 |
|------|------|
| `Track4World/visualization/vis_3d_efep_world.py` | Viser 主可视化脚本（含全部编辑功能） |
| `results/results/test_da3_100frames/3d_efep_output/` | 100 帧推荐展示数据 |

## 启动命令

```bash
cd e:/bishe2
E:/Conda/envs/track4world/python.exe Track4World/visualization/vis_3d_efep_world.py \
    --ply_dir results/results/test_da3_100frames/3d_efep_output \
    --save_dir recordings/test_edit
```

## 关键环境

```
项目路径:   E:/bishe2/
Conda 环境: track4world
Python:     E:/Conda/envs/track4world/python.exe
GPU:        RTX 3050 Laptop 4GB VRAM
Viser:      1.0.15
```

> 最后更新: 2026-04-24
> 项目: 面向具身数据采集场景的关键点追踪系统
> 状态: **Viser 前端编辑功能开发中，轨迹修正功能待实现**

---

## 当前工作重点（2026-04-24）

### 已完成
1. ✅ **Viser GUI 中文化** — 所有控件改为中文
2. ✅ **Tab 分页布局** — 右上角分为"播放控制"/"编辑与导出"两个标签页
3. ✅ **区域框选功能** — XYZ 滑块定义 3D 选择框，黄色线框可视化，框内轨迹高亮白色
4. ✅ **轨迹删除/恢复** — 删除选中轨迹、恢复全部、统计信息实时更新
5. ✅ **导出功能** — 导出剩余轨迹 .npy + 保存 .viser 文件，合并到同一 tab

### 待完成（优先级排序）
1. ⏳ **轨迹修正（钉钉子）** — 核心功能，下一步立即实现
2. ⏳ **展示视频录制** — 基于 100 帧数据
3. ⏳ **论文撰写**
4. ⏳ **答辩准备**

---

## 轨迹修正功能设计（下一步实现）

### 方案：钉钉子 + 2D 投影图交互

**核心问题**：Viser 不提供 3D 视口鼠标事件，用户无法直接点击轨迹。
**解决方案**：在"编辑与导出" tab 中用 `add_image` 显示当前帧 2D 投影图，用 XY 滑块指定位置，后端反投影找最近轨迹。

**用户流程**：
1. 播放暂停在发现错误的帧
2. 点击"生成投影图"按钮 → GUI 显示当前帧点云投影（带轨迹点）
3. 用 X/Y 滑块在图上指定错误轨迹的大致位置
4. 点击"匹配最近轨迹" → 后端反投影找最近轨迹 ID，高亮显示（变红色）
5. 用户确认后，指定"修正起点帧"和"修正终点帧"的目标坐标
6. 点击"应用修正" → 中间帧三次样条插值，实时刷新显示

**技术要点**：
- 投影图生成：用 `c2w[t]` 相机矩阵将 3D 轨迹点投影到 2D，PIL 绘图
- 反投影：像素坐标 + 深度 → 3D 坐标（用相机内参），最近邻匹配
- 插值：`scipy.interpolate.CubicSpline` 对选中轨迹的 XYZ 分别插值

---

## 核心文件

| 文件 | 说明 |
|------|------|
| `Track4World/visualization/vis_3d_efep_world.py` | Viser 主可视化脚本（含编辑功能） |
| `results/results/test_da3_100frames/3d_efep_output/` | 100 帧推荐展示数据 |
| `results/test_da3_250frames/3d_efep_output/` | 250 帧备用数据（21GB，本地无法完整加载） |
| `项目规划.md` | 总体规划 |

## 启动命令

```bash
cd e:/bishe2
E:/Conda/envs/track4world/python.exe Track4World/visualization/vis_3d_efep_world.py \
    --ply_dir results/results/test_da3_100frames/3d_efep_output \
    --save_dir recordings/test_edit
```

## 关键环境信息

```
项目路径:      E:/bishe2/
Conda 环境:    track4world
Python 路径:   E:/Conda/envs/track4world/python.exe
GPU:           RTX 3050 Laptop 4GB VRAM
Viser 版本:    1.0.15
```

> 最后更新: 2026-04-20 (✅ 250帧完整数据推理成功！)
> 项目: 面向具身数据采集场景的关键点追踪系统
> 状态: **AutoDL成功完成250帧推理，动态轨迹质量显著提升，准备进入前端开发阶段**

---

## 🎯 最新进展（2026-04-20）

### 成功完成250帧大规模推理
**关键成果**：
- ✅ **250帧完整推理**（DA3+world坐标系+DINO/SAM2 mask）
- ✅ **动态轨迹数量**：12,653条（100帧数据）→ 预计30,000+条（250帧）
- ✅ **长轨迹稳定性**：5,999条轨迹持续51-100帧（100帧数据）
- ✅ **动态点覆盖率**：8-9%（正常水平）
- ✅ **内存优化**：Viser可视化脚本添加gc垃圾回收，避免OOM

### 技术突破
1. **解决显存瓶颈**：发现Track4World在处理大帧数时存在显存累积问题
   - 100帧：正常运行
   - 200帧：150帧后开始卡顿
   - 300帧：228帧后指数级减速
   - **最终方案**：250帧为最佳平衡点（约8秒视频，足够展示）

2. **Viser内存优化**：
   - 添加 `import gc` 模块
   - 每10帧强制垃圾回收
   - 处理完大数组后立即 `del` 释放
   - 静态背景处理各阶段后执行 `gc.collect()`
   - **效果**：内存占用从98%峰值降至稳定水平

3. **数据压缩效率**：
   - 原始数据：~20GB（轨迹npy文件巨大）
   - 压缩后：1.2GB（tar.gz压缩率 ~17:1）
   - 大量NaN值和重复数据，压缩效果极佳

---

## 🎯 重大突破（2026-04-19 下午）

### 问题根源分析
**为什么之前无法达到官方展示效果？**

1. **坐标系错误**: 使用了`camera_base`（相机坐标系），导致背景随相机移动
2. **模型选择错误**: 使用MOGE而非DA3，深度估计质量较低
3. **分割方法错误**: 使用背景减除（3-4%覆盖率），而非DINO+SAM2
4. **可视化脚本错误**: 使用`vis_3d_efep.py`而非`vis_3d_efep_world.py`

### 官方推荐配置（README标准流程）

```bash
# 步骤1: DINO + SAM2 分割动态物体
python scripts/run_dino_sam2.py \
    --video-path demo_data/cat.mp4 \
    --sam2-checkpoint checkpoints/sam2.1_hiera_large.pt \
    --output-dir results/cat \
    --text-prompt "cat."

# 步骤2: Track4World 3d_efep 推理（世界坐标系）
python demo.py \
    --mp4_path demo_data/cat.mp4 \
    --coordinate world_depthanythingv3 \
    --mode 3d_efep \
    --Ts -1 \
    --ckpt_init checkpoints/track4world_da3.pth \
    --save_base_dir results/cat

# 步骤3: 世界坐标系可视化
python visualization/vis_3d_efep_world.py \
    --ply_dir results/cat/3d_efep_output \
    --save_dir recordings/cat
```

### 关键差异对比

| 特性 | 之前的方案 | 官方方案 | 效果差异 |
|------|-----------|---------|---------|
| 坐标系 | camera_base | world_depthanythingv3 | 背景从"移动"变"静止" |
| 深度模型 | MOGE | DA3 | 点云覆盖率从47%提升到60%+ |
| 前景分割 | 背景减除 | DINO+SAM2 | 覆盖率从3-4%提升到90%+ |
| 可视化 | vis_3d_efep.py | vis_3d_efep_world.py | 前景-背景清晰分离 |

---

## 📋 当前工作重点

### 已完成（2026-04-20）
1. ✅ **100帧测试推理** — 验证mask修复效果，动态点覆盖率8-9%
2. ✅ **250帧完整推理** — AutoDL成功完成，数据已下载到本地
3. ✅ **Viser内存优化** — 添加gc垃圾回收，避免本地可视化OOM
4. ✅ **显存瓶颈分析** — 确定250帧为AutoDL推理最佳平衡点
5. ✅ **数据压缩下载** — 20GB → 1.2GB，tar.gz高效压缩
6. ✅ **250帧数据解压** — 成功解压到本地（26GB）
7. ✅ **大数据可视化方案探索** — 测试多种降采样和内存映射方案
8. ✅ **确定展示方案** — 使用100帧数据作为最终展示（硬件限制）

### 技术难点与解决方案
**问题1：250帧轨迹数据过大（21.69GB）**
- 尝试方案：
  - ✅ 内存映射（mmap_mode='r'）
  - ✅ 分块处理避免OOM
  - ✅ 降采样轨迹（7023条 → 500条）
  - ✅ 轻量级点云可视化
  - ❌ 完整轨迹动画（浏览器渲染卡顿）
- 最终方案：**使用100帧数据展示**（内存可控，效果完整）

**问题2：Viser可视化内存占用高**
- 解决方案：
  - 添加gc垃圾回收（每10帧）
  - 大数组处理后立即del释放
  - 静态背景处理各阶段gc.collect()
  - 效果：内存占用从98%峰值降至稳定

### 当前数据状态
- **100帧数据**（推荐用于展示）：`results/results/test_da3_100frames/3d_efep_output/`
  - 动态轨迹：12,653条
  - 长轨迹（51-100帧）：5,999条
  - 动态点覆盖率：8-9%
  - 内存占用：~4GB（可完整加载）
  - ✅ 支持完整Viser动画播放
  
- **250帧数据**（备用）：`results/test_da3_250frames/3d_efep_output/`
  - 动态轨迹：~30,000条（估计）
  - 长轨迹（101-250帧）：~15,000条（估计）
  - 轨迹文件大小：21.69GB
  - ⚠️ 受限于本地4GB显存，无法完整可视化
  - 可用于静态截图或降采样展示

### 待完成（按优先级）
1. ⏳ **前端功能开发** — 框选区域、轨迹纠正等交互功能（使用20帧小数据测试）
2. ⏳ **录制展示视频** — 基于100帧数据录制Viser动画
3. ⏳ **前端集成100帧数据** — 将100帧数据集成到Gradio平台
4. ⏳ **论文撰写** — 技术方案、实验结果、系统设计
5. ⏳ **答辩准备** — PPT制作、演示视频、Q&A准备

---

## 总进度

| 阶段 | 状态 | 完成度 |
|------|------|--------|
| 一：环境搭建与模型复现 | **已完成** | 100% |
| 二：评估与分析 | **已完成** | 100% |
| 三：展示平台搭建 | **进行中** | ~75% |
| 四：论文撰写与答辩 | 未开始 | 0% |

### 阶段三进展详情
- ✅ 数据推理（250帧高质量数据）
- ✅ 可视化优化（Viser内存管理）
- ⏳ 前端交互功能（框选、纠正）
- ⏳ 展示视频录制

---

## 阶段一：环境搭建与模型复现 ✅

### 已完成
- [x] Conda 环境 `track4world` 搭建 (Python 3.11 + PyTorch 2.5.1+cu121)
- [x] Track4World 仓库克隆 + 全部依赖安装
- [x] 三个预训练权重下载 (da3/pi3/moge)
- [x] 三种模式 (2d/3d_ff/3d_efep) 推理成功验证
- [x] 代码结构梳理 → `docs/技术文档.md`
- [x] 项目规划文档 → `项目规划.md`
- [x] 一键展示脚本 → `stage1_showcase.py`
- [x] HTML 报告 → `stage1_results/report.html`

---

## 阶段二：评估与分析 ⏳

### 3D Tracking 评估 ✅ (TAPVid-3D, L-16, camera_base + moge)

| 数据集 | OA | AJ | APT | APT(occ) | 耗时 |
|--------|------|------|------|----------|------|
| ADT (Aria Digital Twin) | 0.988 | 0.474 | 0.518 | 0.519 | ~53min |
| DS (Dynamic Stereo) | 0.972 | 0.468 | 0.530 | 0.528 | ~85min |
| PO (Panoptic Objects) | 0.807 | 0.384 | 0.525 | 0.522 | ~102min |
| PStudio (Panoptic Studio) | 0.938 | 0.527 | 0.596 | 0.596 | ~102min |

### Flow 评估 ✅ (Kubric Short)

| 指标 | 值 | 指标 | 值 |
|------|------|------|------|
| EPE3D | 0.1935 | EPE2D | 2.5354 |
| Acc3D_strict | 0.3558 | ACC3_2D | 0.9280 |
| Acc3D_relax | 0.5801 | Outlier_2D | 0.0208 |
| abs_rel | 0.0297 | δ<1.25 | 0.9820 |

### 2D Tracking 评估 (部分)
- RGB-Stacking 数据集已下载到 `Track4World/evaluation/2d_track/tapvid_rgb_stacking/`
- RoboTAP 压缩包已下载到 `Track4World/evaluation/2d_track/robotap/`（未解压处理）
- 导师反馈：偏向工程实践，**不需要继续跑标准评估指标**，直接在具身场景下评估

---

## 阶段三：展示平台搭建 ⏳

### 导师指导方向
- 通过点追踪方式进行三维数据标注
- 两种场景：**真机数据**（机械臂+灵巧手/二指夹爪）、**人类数据**（人抓握物体）
- 导师已提供人类数据素材

### 已完成
- [x] Gradio 平台框架搭建（`platform/app.py`）
- [x] 推理后端封装（`platform/inference.py`）
- [x] 可视化工具（`platform/visualize.py`）
- [x] 人类抓握数据素材导入（`embodied_data/human_grasp/`）
- [x] GT 标注对比功能
- [x] 启动脚本（`启动平台.bat`）
- [x] Colab 推理环境配置（`colab_inference.ipynb`）
- [x] 平台"加载已有结果"功能（支持展示 Colab 结果）
- [x] Colab 成功跑通 2d 模式（50帧 21秒）

### 当前工作流程（已验证可行）

**Colab 推理**（3D 模式）：
1. Cell 1: 检查 GPU
2. Cell 2: 安装依赖 + 自动切换到 `b5bcd5d` 版本（包含内置 utils3d）
3. Cell 3: 下载权重
4. Cell 4: 上传视频
5. Cell 5: 运行 3d_ff 推理（MODE="3d_ff"）
6. Cell 6: 打包下载结果

**本机推理**（2D 模式）：
- 启动平台 → 上传视频 → 选择 2d 模式 → 推理
- 或命令行：`python demo.py --mp4_path 视频.mp4 --mode 2d --image_size 320 --max_frames 20`

**备用方案**（Colab 跑 2D）：
- Cell 2.5: 切换回新版本（master 分支）
- Cell 7: 运行 2d 推理（新版本不需要 utils3d.depth_to_points）

### 关键问题已解决
- ✅ utils3d.depth_to_points 缺失 → 使用 Track4World 内置 utils3d（commit b5bcd5d）
- ✅ Viser 3D 查看器无画面 → utils3d 修复后点云正常生成，Viser 可显示
- ✅ Colab numpy 版本冲突 → 强制重装 numpy 1.26.4
- ✅ 平台加载 Colab 结果 → "加载已有结果"功能已实现

### 已完成（本次会话 2026-04-18）

#### 问题诊断与解决
- ✅ **诊断Colab推理问题** — "像素点丢失"实际是坐标系问题
- ✅ **根因分析** — DA3模型使用world_depthanythingv3坐标系，导致Z值为负
- ✅ **创建修复脚本** — fix_pointcloud_coordinates.py（翻转Z轴）
- ✅ **修复点云数据** — 198个PLY文件已修复，Z值从负转正
- ✅ **修复Viser分辨率问题** — 448x320 → 448x336（正确匹配推理分辨率）
- ✅ **修复mask前6帧** — 用第6帧替换，覆盖率从66%降到10.45%

#### 深度问题分析
- ✅ **发现mask处理问题** — 颜色匹配成功率只有56.3%，导致动态点标记错误
- ✅ **发现轨迹追踪问题** — Viser使用颜色匹配追踪，失败率高导致轨迹显示为直线
- ✅ **发现根本原因** — Track4World在推理时就使用了mask，问题在推理阶段而非可视化阶段

#### 解决方案
- ✅ **更新Colab配置** — 默认使用MOGE+camera_base，避免坐标系问题
- ✅ **禁用mask推理** — 修改colab_inference.ipynb，不使用mask重新推理（测试20帧）
- ✅ **创建诊断工具** — test_mask_resolution.py, test_dynamic_mask.py

### 已完成（上次会话 2026-04-16）
- ✅ **分割模型调研完成** — 调研SAM 2、YOLOv8-seg、Mask2Former、SegFormer
- ✅ **安装ultralytics** — YOLOv8分割模型库
- ✅ **测试多个方案** — 混合v1、混合v2、纯YOLOv8
- ✅ **确定最优方案** — 组合方案（YOLOv8 + 背景减除）
- ✅ **创建改进脚本** — generate_mask_combined.py
- ✅ **效果验证** — 覆盖率从3-4%提升到9.6-17.7%（3-5x提升）

### 当前数据状态
- 旧mask：`E:/bishe2/Track4World/results/human_grasp_3d_efep_masked/mask/` (616帧，背景减除，3-4%覆盖率)
- 新mask测试：`E:/bishe2/Track4World/results/human_grasp_yolov8_only/mask/` (20帧，YOLOv8，9-10%覆盖率)
- **改进效果**：覆盖率提升2.5倍，完整标记人体区域，无背景噪声

### 分割模型方案对比

| 方案 | 覆盖率 | 稳定性 | 噪声 | 速度 | 推荐度 |
|------|--------|--------|------|------|--------|
| 旧方案（背景减除） | 3-4% | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| 混合v1 | 12-30% | ⭐ | ⭐ | ⭐⭐ | ⭐⭐ |
| 混合v2（改进） | 9-11% | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **纯YOLOv8（推荐）** | **9-10%** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** |

## 核心文件清单

## 核心文件清单

### 🆕 250帧完整数据（2026-04-20）

**AutoDL推理结果**:
- 压缩包: `test_da3_250frames_results.tar.gz` (1.2GB)
- 解压后路径: `results/test_da3_250frames/3d_efep_output/` (26GB)
- 包含内容:
  - 250个frame PLY文件（点云）
  - 250个flow PLY文件（光流）
  - 250个pc_dyn_mask NPY文件（动态mask）
  - trajectory_all_pointmap.npy（完整轨迹数据，21.69GB）
  - trajectory_all_pointmap_dyn_mask.npy（动态轨迹mask，7.23GB）
  - c2w.npy（相机位姿）

**数据质量指标**:
- 总轨迹点: 7,229,366个
- 动态轨迹数: 12,666条
- 超长轨迹（201-250帧）: 2,444条
- 长轨迹（101-200帧）: 1,331条
- 中等轨迹（51-100帧）: 2,212条
- 短轨迹（1-50帧）: 6,679条

**可视化限制**:
- ⚠️ 轨迹文件过大（21.69GB），本地4GB显存无法完整加载
- ✅ 可用于静态截图展示
- ✅ 可用于降采样可视化（500条轨迹）
- ❌ 无法实现完整动画播放

**降采样可视化脚本**:
- `vis_250frames_lite.py` — 轻量级点云查看器（无轨迹）
- `vis_250frames_trajectories.py` — 降采样轨迹线（静态，500条）

### 🆕 100帧数据（推荐用于展示）

**数据路径**: `results/results/test_da3_100frames/3d_efep_output/`

**数据质量**:
- 动态轨迹: 12,653条
- 长轨迹（51-100帧）: 5,999条
- 动态点覆盖率: 8-9%
- 轨迹文件大小: ~4GB（可完整加载）

**优势**:
- ✅ 内存占用可控，支持完整Viser动画
- ✅ 轨迹数量充足，展示效果好
- ✅ 可调整轨迹线粗细增强视觉效果
- ✅ 足够答辩和论文使用

**使用方法**:
```bash
cd e:/bishe2
E:/Conda/envs/track4world/python.exe Track4World/visualization/vis_3d_efep_world.py \
    --ply_dir results/results/test_da3_100frames/3d_efep_output \
    --save_dir recordings/test_100frames
```

### 🆕 Viser优化版本（2026-04-20）

**优化内容**:
- `Track4World/visualization/vis_3d_efep_world.py` — 添加内存管理
  - 导入gc模块
  - 每10帧强制垃圾回收
  - 大数组处理后立即del释放
  - 静态背景处理各阶段gc.collect()

**使用方法**:
```bash
cd e:/bishe2
E:/Conda/envs/track4world/python.exe Track4World/visualization/vis_3d_efep_world.py \
    --ply_dir results/test_da3_250frames/3d_efep_output \
    --save_dir recordings/test_250frames
```

### 🆕 AutoDL云服务器配置（2026-04-19-20）

**服务器信息**:
- GPU: RTX 4090 (32GB显存)
- Python: 3.11
- 环境: conda环境 `track4world`
- 数据盘: `/root/autodl-tmp/Track4World`（关机保存）
- 系统盘: `/root/miniconda3/`（关机保存）

**关键修复**:
- utils3d模块冲突：重命名numpy→numpy3d, torch→torch3d_real, io→io_backup
- 软链接：`/root/miniconda3/envs/track4world/lib/python3.11/site-packages/utils3d`

**推理经验**:
- 100帧：正常运行，约15分钟
- 150帧：开始出现显存累积，速度下降
- 200帧：150帧后显著减速
- 250帧：228帧后指数级减速，但能完成（约7分钟）
- 300帧+：不推荐，显存碎片化严重

**推理命令**（250帧最佳配置）:
```bash
conda activate track4world
cd /root/autodl-tmp/Track4World
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p results/test_da3_250frames/mask
cp results/input_copy/mask/*.png results/test_da3_250frames/mask/

python demo.py \
    --mp4_path input_copy.mp4 \
    --coordinate world_depthanythingv3 \
    --mode 3d_efep \
    --Ts 250 \
    --ckpt_init checkpoints/track4world_da3.pth \
    --image_size 448 \
    --save_base_dir results/test_da3_250frames 2>&1 | tee run_250frames.log

# 打包下载
tar -czf test_da3_250frames_results.tar.gz results/test_da3_250frames/3d_efep_output
```

**注意事项**:
- ⚠️ 必须先复制mask文件到结果目录
- ⚠️ 保存轨迹npy文件需要约20GB临时空间
- ⚠️ 推理前清理旧结果释放空间
- ✅ 后续前端开发不需要云服务器，本地20帧测试即可

### 🆕 前端更新（2026-04-19）

**新增功能**:
- `platform/app.py` 添加 `launch_open3d()` 函数
- Gradio UI 添加 Open3D 单帧查看按钮
- 支持选择帧号查看点云

**使用方法**:
```bash
cd E:/bishe2
E:/Conda/envs/track4world/python.exe platform/app.py
# 浏览器打开 http://localhost:7860
# Tab 5: Viser 3D 查看器 → Open3D 单帧查看
```

### 🆕 Kaggle推理文件（2026-04-19）

**Kaggle Notebook**:
- **kaggle_inference.ipynb** — Kaggle版推理配置（Python 3.10，解决Colab兼容性问题）⭐

**使用指南**:
- **docs/Kaggle使用指南.md** — 完整Kaggle使用教程（快速开始、参数配置、故障排除）⭐

**关键差异**:
| 特性 | Colab | Kaggle |
|------|-------|--------|
| Python版本 | 3.12 | 3.10 |
| numpy兼容性 | ❌ 与opencv冲突 | ✅ 完全兼容 |
| 路径前缀 | `/content/` | `/kaggle/working/` |
| 下载方式 | `files.download()` | Output面板 |
| GPU | T4 15GB | T4 x2 30GB |

### 🆕 官方方案文件（2026-04-19）

**Colab配置**:
- **colab_inference_official.ipynb** — 官方推荐配置（world坐标系+DA3+DINO/SAM2）⭐
- **cell2_完全修复版.txt** — Cell 2修复版代码（解决依赖冲突）⭐

**本地脚本**:
- **run_official_visualization.bat** — 一键启动官方可视化 ⭐

**核心文档**:
- **README_官方方案.md** — 资源导航和快速索引 ⭐
- **快速开始.md** — 5分钟快速上手指南 ⭐
- **最终总结.md** — 完整工作总结（含使用流程）⭐

**详细文档**:
- **docs/官方效果复现指南.md** — 完整操作指南（约200行）⭐
- **docs/新旧方案对比.md** — 新旧方案详细对比（约300行）⭐
- **docs/工作总结_官方方案.md** — 本次工作的完整总结 ⭐
- **docs/Colab故障排除指南.md** — Colab错误解决方案（新增）⭐⭐

### 旧方案文件（保留作为参考）

**Colab配置**:
- **colab_inference.ipynb** — 旧版配置（camera_base+MOGE）

**关键脚本**:
- **fix_pointcloud_coordinates.py** — 坐标系修复（Z轴翻转）
- **visualize_trajectories.py** — 3D轨迹可视化生成
- **vis_ultimate_fix.py** — 自定义可视化脚本

**结果数据**:
- **results/output_3d_efep/** — 旧版Colab推理结果（20帧）
  - 3d_efep_output/*.ply — 点云文件
  - 3d_efep_output/trajectory_all_pointmap.npy — 完整轨迹数据
- **trajectories_colab.html** — 3D轨迹可视化（885条轨迹）

**文档**:
- **HANDOVER.md** — 本文档
- **docs/最终诊断_轨迹可视化问题.md** — 问题分析
- **紧急解决方案.md** — 旧版可行方案

---

## 技术要点

### 1. Colab坐标系问题
- **问题**：旧版Track4World使用world坐标系，Z值为负
- **解决**：Cell 5.5自动检测并翻转Z轴
- **验证**：Z>0比例应为100%

### 2. 本地运行Track4World
- **关键参数**：`--coordinate camera_base`（避免下载DA3）
- **显存限制**：4GB只能处理320分辨率、5-10帧
- **推荐**：使用Colab推理完整数据

### 3. 轨迹数据提取
- **输入**：trajectory_all_pointmap.npy (T, N, 3)
- **处理**：过滤NaN，提取有效轨迹
- **输出**：Plotly HTML可视化

---

## 下一步行动

### 立即执行（前端功能开发）

**开发环境**：
- 使用20帧小数据测试功能（`--Ts 20`）
- 本地4GB显存足够开发和测试
- 功能验证通过后，集成100帧完整数据

**前端功能清单**：
1. **框选区域功能**
   - 在Viser/Gradio中实现3D区域框选
   - 筛选框选区域内的轨迹点
   - 导出选中区域的轨迹数据

2. **轨迹纠正功能**
   - 手动标记错误轨迹
   - 删除或修正异常轨迹点
   - 轨迹平滑处理

3. **动态点筛选功能**
   - 按轨迹长度筛选（显示N帧以上的轨迹）
   - 按运动幅度筛选（过滤静止点）
   - 按置信度筛选

4. **轨迹导出功能**
   - 导出为标准格式（JSON/CSV）
   - 支持选择性导出（框选区域/筛选后）
   - 生成可视化报告

**开发流程**：
```bash
# 1. 生成20帧测试数据（如果没有）
cd E:/bishe2/Track4World
python demo.py \
    --mp4_path ../embodied_data/human_grasp/human_grasp_video0.mp4 \
    --coordinate world_depthanythingv3 \
    --mode 3d_efep \
    --Ts 20 \
    --ckpt_init checkpoints/track4world_da3.pth \
    --image_size 448 \
    --save_base_dir ../results/test_20frames

# 2. 启动Gradio平台开发
cd E:/bishe2
E:/Conda/envs/track4world/python.exe platform/app.py

# 3. 功能验证通过后，替换为100帧数据
```

### 后续规划
- **展示视频录制**：基于100帧数据，录制Viser动画和前端交互演示
- **论文撰写**：技术方案、实验结果、系统架构、性能分析
- **答辩准备**：PPT制作、演示视频、Q&A准备
- **可选优化**：如需更多数据，可重新租用AutoDL（已有完整配置）

---

## 常见问题

### Q: 为什么本地无法运行？
A: 网络无法连接HuggingFace，且4GB显存不足

### Q: 如何避免下载DA3？
A: 使用`--coordinate camera_base`参数

### Q: 轨迹为什么是直线？
A: Viser/Plotly使用颜色匹配追踪，失败率高。需要使用Track4World原生的trajectory数据。

### Q: 如何获得正确的可视化？
A: 使用trajectory_all_pointmap.npy + visualize_trajectories.py生成

### 关键问题已解决：Colab推理坐标系问题

**问题描述**：Colab推理结果在可视化时出现"大面积像素点丢失"

**根本原因**：
- Colab使用DA3模型 + world_depthanythingv3坐标系
- 该坐标系导致所有点的Z值为负（-0.164到-0.023）
- 点云在相机后面，可视化时被错误渲染或剔除

**解决方案**：
1. **临时方案**：使用fix_pointcloud_coordinates.py翻转Z轴
   - 已修复：results_3d_efep_fixed/ (198个PLY文件)
   - Z值从负转正：[0.023, 0.164]
2. **长期方案**：修改Colab配置使用MOGE + camera_base
   - 已更新colab_inference.ipynb，默认USE_DA3=False
   - 与本地环境保持一致

### 待解决问题（优先级）
1. ⏳ **Colab重新推理** — 使用MOGE+camera_base重新推理（避免坐标系问题）
2. ⏳ **可视化验证** — 使用vis_ultimate_fix.py查看修复后的点云
3. ⏳ **评估改进效果** — 对比新旧结果的轨迹完整性和点云覆盖率
4. **Viser交互功能** — 手动区域选取、错误修正（备选）
5. **真机数据验证** — 机械臂抓取场景

### 下一步操作（推荐）

**立即执行（Colab测试）**:
```bash
# 1. 在Colab中运行
Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5（跳过Cell 4.5）→ Cell 6

# Cell 5配置：
# - IMAGE_SIZE = 448
# - MAX_FRAMES = 20（测试用）
# - USE_DA3 = False（使用MOGE）
# - 自动删除旧mask

# 2. 下载结果到本地
# 解压到：E:/bishe2/results_test_no_mask/

# 3. 验证结果
cd E:/bishe2
python vis_ultimate_fix.py --ply_dir results_test_no_mask/3d_efep_output

# 4. 检查点云Z值
python -c "
import open3d as o3d
import numpy as np
pcd = o3d.io.read_point_cloud('results_test_no_mask/3d_efep_output/frame_000.ply')
pts = np.asarray(pcd.points)
print(f'Z范围: [{pts[:, 2].min():.3f}, {pts[:, 2].max():.3f}]')
print(f'Z>0: {(pts[:, 2] > 0).sum()} / {len(pts)} ({(pts[:, 2] > 0).sum()/len(pts)*100:.1f}%)')
"
```

**如果测试成功**:
- 修改MAX_FRAMES = 500
- 重新推理完整数据
- 替换旧结果

**如果测试失败**:
- 检查Colab输出日志
- 确认是否使用了mask
- 确认坐标系参数

### 启动方式
双击 `启动平台.bat`，浏览器打开 http://localhost:7860

---

## 关键环境信息

```
项目路径:      E:/bishe2/
模型代码:      E:/bishe2/Track4World/
Conda 环境:    track4world
Python 路径:   E:/Conda/envs/track4world/python.exe
系统 Python:   C:/Python312/python.exe (3.12) ← 不要用这个跑模型!
GPU:           RTX 3050 Laptop 4GB VRAM
VRAM 限制:     只能用 camera_base + moge 权重 (DA3/Pi3 会 OOM)
HF 镜像:       HF_ENDPOINT=https://hf-mirror.com
```

## 关键命令

```bash
cd "E:/bishe2/Track4World"
T4W="E:/Conda/envs/track4world/python.exe"

# 2D 追踪推理
$T4W demo.py --mp4_path demo_data/cat.mp4 --mode 2d --image_size 320 --max_frames 20

# 3D 追踪推理 (必须用 camera_base + moge)
$T4W demo.py --mp4_path demo_data/cat.mp4 --mode 3d_ff \
  --coordinate camera_base --ckpt_init checkpoints/track4world_moge.pth \
  --image_size 448 --max_frames 10

# 启动 Gradio 平台
$T4W ../platform/app.py
```

---

## 用户偏好
- 沟通语言: 中文
- 重视成果可展示性
- 阶段三计划使用 Colab 免费 GPU 或租云服务器跑长帧高分辨率展示素材
- 偏好工程实践方向（导师反馈）

## 文件索引

| 文件 | 说明 |
|------|------|
| `项目规划.md` | 总体规划 (四个阶段, 目录结构) |
| `docs/技术文档.md` | 模型架构+代码结构+维度速查 |
| `platform/app.py` | Gradio 展示平台主入口 |
| `platform/inference.py` | 推理后端封装 |
| `platform/visualize.py` | 可视化工具 |
| `embodied_data/human_grasp/` | 人类抓握数据素材 |
| `stage1_results/report.html` | 阶段一 HTML 报告 |
| `stage2_results/eval_results.md` | 阶段二评估结果 |
| `Track4World/demo.py` | 模型推理入口 (已修改 float16) |
| `启动平台.bat` | 一键启动 Gradio 平台 |
| `colab_inference.ipynb` | Colab 推理 notebook（Cell 7 批量跑所有模式） |
| `HANDOVER.md` | 本文档 |
