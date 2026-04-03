# HANDOVER.md — 毕设2 工作交接文档

> 最后更新: 2026-03-28 (阶段三初期)
> 项目: 面向具身数据采集场景的关键点追踪系统的设计与实现
> 项目路径: E:/bishe2/ (从 d:/桌面/文件/毕设2/ 迁移，解决中文路径问题)
> 用途: 跨会话恢复工作进度

---

## 总进度

| 阶段 | 状态 | 完成度 |
|------|------|--------|
| 一：环境搭建与模型复现 | **已完成** | 100% |
| 二：评估与分析 | **基本完成** | ~85% |
| 三：展示平台搭建 | **进行中** | ~40% |
| 四：论文撰写与答辩 | 未开始 | 0% |

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

### 当前工作流程
1. **数据推理**：在 Colab 上运行 `colab_inference.ipynb`（Cell 7 可一次跑完所有模式）
2. **结果下载**：Cell 6 打包下载 zip，解压到 `Track4World/results/`
3. **平台展示**：启动平台，用"加载已有结果"输入目录路径查看

### 后续计划（优先级从高到低）
1. **2D 可视化优化** — 背景保持原始视频，仅将人物/物体轨迹渲染为彩色像素图
2. **4D 场景展示** — 时空轨迹的交互式可视化
3. **Viser 3D 点云查看器修复** — 解决无画面问题（需 gsplat 依赖）
4. **vis_3d_ff.py IndexError 修复** — PLY/NPY 数量不匹配时的数组越界

### 已知问题（低优先级）
- ffprobe 路径问题（Gradio 内部调用，不影响核心功能）
- 本机 4GB 显存限制（已通过 Colab 解决）

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
