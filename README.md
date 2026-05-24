# Video-sequence-based Human Motion Capture and Measurement

基于单目视频的人体运动捕捉与测量系统。输入任意视频，输出人体33个关节点的运动轨迹、关节角度/角速度/角加速度时序数据，支持手动打点追踪任意物体（如篮球），并支持纯黑背景的可视化回放与CSV轨迹导出。

## 项目目标

1. 从视频中逐帧检测人体33个关键关节点（MediaPipe Pose）
2. 跨帧跟踪关节点轨迹，对低置信度帧进行插值和平滑
3. 计算运动学指标：关节角度、角速度、角加速度、轨迹长度、运动范围(ROM)
4. 支持手动点击打点，使用 Lucas-Kanade 光流追踪任意物体（篮球等），计算瞬时速度、累计位移
5. 提供图形界面进行交互式录制、手动打点与纯黑背景回放
6. 导出所有追踪点（人体自动点 + 手动点）为CSV格式

## 技术路线

```
视频输入 → OpenCV读取帧 → MediaPipe Pose(33关键点) → 插值平滑
                               ↓
                       手动打点 → LK光流追踪 → 物体轨迹记录
                               ↓
                       运动学计算(角度/速度/加速度)
                               ↓
                       可视化(轨迹/曲线/热力图/动画)
                               ↓
                       数据导出(CSV/JSON/MAT)
```

- **姿态估计**：MediaPipe Pose Landmarker（0.10.x Tasks API），33关键点，CPU可运行
- **手动点追踪**：Lucas-Kanade 金字塔光流（`cv2.calcOpticalFlowPyrLK`），窗口大小41×41，4层金字塔，亚像素精度收敛。点击画面任意位置即可标记并开始追踪
- **运动学计算**：余弦定理（角度）、中心差分法（角速度/角加速度）、累计欧氏距离（轨迹长度）
- **GUI框架**：PyQt5 + 侧边栏，QThread后台处理防止界面卡死，鼠标事件过滤支持点击打点
- **可视化**：Matplotlib（静态图表）、OpenCV（实时绘制/动画渲染）

## 环境配置

```bash
git clone https://github.com/Soliluna26/Video-sequence-based-human-motion-capture-and-measurement.git
cd Video-sequence-based-human-motion-capture-and-measurement

pip install -r requirements.txt
```

依赖：`opencv-python` `opencv-contrib-python` `mediapipe` `PyQt5` `numpy` `scipy` `matplotlib` `Pillow` `PyYAML` `imageio`

首次运行时会自动下载 MediaPipe Pose Landmarker 模型文件（~10MB），保存在 `~/.mediapipe/models/`。

## 使用方式

### GUI 图形界面（推荐）

```bash
python main.py
```

操作流程：

1. 点击 **Open Video**，选择 mp4/avi/mov/gif 视频文件
2. （可选）点击 **Add Manual Point** 激活打点模式，在画面任意位置点击鼠标左键，添加手动追踪点；可重复添加多个点
3. 点击 **Start**，后台开始 MediaPipe 姿态估计 + LK光流追踪手动点，前台显示原视频
4. 视频结束或点击 **End**，进入回放就绪状态
5. 点击 **Replay**，纯黑背景回放：
   - 青色(Cyan)线条和点：人体骨架运动姿态
   - 蓝色(Blue)连续线条 + 采样点小圆：手动标记点的历史轨迹
   - 蓝色大圆点（实心 + 轮廓）：手动标记点当前位置
   - 左上角实时显示：`Velocity(mm/s)` `Distance(mm)` `Times(s)`
6. 点击 **Save Result**，将回放动画导出为 MP4 文件
7. 点击 **Export CSV**，导出所有追踪数据（人体关键点 + 手动点，含类型区分）

**手动点管理：**
- 录制过程中可随时点击 **Add Manual Point** 添加新追踪点（自动从当前帧开始光流跟踪）
- 右侧侧边栏显示所有手动点列表，每个点有红色删除按钮
- 光流跟丢后该点自动停用，旧轨迹保留在画面中；重新打点即可开始新轨迹
- 轨迹的每个采样点上画出蓝色小圆点，便于观察跟踪精度

### CLI 命令行模式（批量处理）

```bash
python main.py --input data/sample.mp4 --max_frames 200
python main.py --input data/sample.mp4 --output_dir results/ --export_format csv,json,mat
python main.py --input data/sample.gif --no_animation
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入视频路径（必填） | - |
| `--max_frames` | 最大处理帧数 | 全部 |
| `--output_dir` | 输出目录 | `output/` |
| `--export_format` | 导出格式(csv,json,mat) | `csv,json` |
| `--no_animation` | 跳过动画生成 | False |

CLI 模式的输出文件：

```
output/
├── trajectory_xy.png       # 关节空间轨迹
├── kinematics_curves.png   # 角度+角速度曲线
├── angle_heatmap.png       # 角度热力图
├── landmarks_3d.png        # 3D关键点散点
├── animation.mp4           # 轨迹叠加动画
├── kinematics.csv          # 时序数据表
├── metrics.json            # 汇总指标
└── kinematics.mat          # MATLAB格式(可选)
```

## 项目结构

```
├── main.py                  # 主入口（GUI + CLI 双模式）
├── requirements.txt
├── config/
│   └── landmarks.yaml       # 33关键点定义 + 10种关节角度配置 + 骨骼连接
├── src/
│   ├── gui/
│   │   ├── main_window.py   # PyQt5主窗口（侧边栏、鼠标打点、轨迹渲染、CSV导出）
│   │   └── video_worker.py  # QThread后台处理线程（MediaPipe + LK光流追踪）
│   ├── frame_loader.py      # 视频/GIF帧读取
│   ├── pose_estimator.py    # MediaPipe姿态估计封装
│   ├── point_manager.py     # 手动追踪点管理器（LK光流更新、历史记录）
│   ├── ball_tracker.py      # HSV篮球追踪（保留供CLI模式使用）
│   ├── tracker.py           # 关节点轨迹追踪与插值
│   ├── kinematics.py        # 运动学计算（角度/角速度/角加速度/轨迹长度/ROM）
│   ├── analyzer.py          # 动作分析（峰值检测/对称性/傅里叶分析）
│   ├── visualizer.py        # 可视化（5种图表 + 轨迹叠加动画）
│   └── exporter.py          # 数据导出（CSV/JSON/MAT）
└── tests/
    └── test_kinematics.py   # 运动学模块单元测试（18项）
```

## 运动学计算公式

### 关节角度（余弦定理）

对于三点 $P_1$（近端）、$P_2$（关节点/顶点）、$P_3$（远端），关节角度 $\theta$：

$$\theta = \arccos\left( \frac{\vec{v}_1 \cdot \vec{v}_2}{|\vec{v}_1| \cdot |\vec{v}_2|} \right) \times \frac{180}{\pi}$$

其中 $\vec{v}_1 = P_1 - P_2$，$\vec{v}_2 = P_3 - P_2$。输出范围 $[0°, 180°]$。

### 角速度（二阶中心差分）

$$\omega_i = \frac{\theta_{i+1} - \theta_{i-1}}{2 \Delta t} \quad (\text{deg/s})$$

首帧前向差分，末帧后向差分。

### 角加速度

对角速度序列再次应用中心差分：

$$\alpha_i = \frac{\omega_{i+1} - \omega_{i-1}}{2 \Delta t} \quad (\text{deg/s}^2)$$

### 轨迹总长度

$$L = \sum_{i=1}^{T-1} \sqrt{ (x_i - x_{i-1})^2 + (y_i - y_{i-1})^2 } \quad (\text{pixels})$$

### 关节运动范围 (ROM)

$$\text{ROM} = \theta_{\max} - \theta_{\min} \quad (\text{deg})$$

## 手动点追踪与速度计算

### Lucas-Kanade 金字塔光流

手动标记点使用 OpenCV 的 `cv2.calcOpticalFlowPyrLK` 进行帧间追踪：

- **搜索窗口**：41×41 像素，捕获物体周围足够纹理信息
- **金字塔层数**：4 层，应对快速移动造成的大位移
- **收敛精度**：epsilon = 0.01，最大迭代 30 次，亚像素精度收敛
- **特征阈值**：minEigThreshold = 0.001，过滤完全平坦的无纹理区域
- **丢跟处理**：光流状态返回 0 时自动停用该点，旧轨迹保留；用户重新打点即可恢复追踪

### 速度与位移

追踪点的瞬时速度通过对最近5帧的帧间位移做移动平均平滑后除以 $\Delta t$ 得到：

$$v_t = \frac{1}{k} \sum_{j=t-k+1}^{t} \frac{ \sqrt{ (x_j - x_{j-1})^2 + (y_j - y_{j-1})^2 } \cdot s }{ \Delta t } \quad (\text{mm/s})$$

其中 $k=5$（平滑窗口），$s$ 为像素到毫米的换算系数（默认 2.0 mm/px，可在 `main_window.py` 中的 `DEFAULT_MM_PER_PIXEL` 调整）。

累计位移为所有帧间步长的累加。

### CSV 导出格式

```csv
frame_idx, time_sec, point_type, point_id, name, x, y
0, 0.0000, human, 11, left_shoulder, 245.32, 180.67
0, 0.0000, human, 12, right_shoulder, 302.15, 178.94
...
0, 0.0000, manual, 0, manual_0, 410.50, 250.30
1, 0.0333, manual, 0, manual_0, 412.18, 248.75
```

`point_type` 为 `human`（MediaPipe自动检测）或 `manual`（用户手动标记），便于后续筛选分析。

## 关键点索引（MediaPipe Pose）

| 索引 | 名称 | 索引 | 名称 | 索引 | 名称 |
|------|------|------|------|------|------|
| 0 | nose | 11 | left_shoulder | 23 | left_hip |
| 1-10 | 面部特征点 | 12 | right_shoulder | 24 | right_hip |
|  |  | 13 | left_elbow | 25 | left_knee |
|  |  | 14 | right_elbow | 26 | right_knee |
|  |  | 15 | left_wrist | 27 | left_ankle |
|  |  | 16 | right_wrist | 28 | right_ankle |
|  |  | 17-22 | 手指关节 | 29-32 | 脚部关节 |

完整33点定义见 `config/landmarks.yaml`。

## 单元测试

```bash
python -m pytest tests/test_kinematics.py -v
```

覆盖6个测试类18项测试：角度计算（直角/平角/锐角/退化解）、角速度（常值/正弦解析对比）、角加速度（正弦二阶导对比）、轨迹长度（直线/对角线/NaN处理）、ROM、平滑。

## 实际测试数据

在篮球视频（548×520, 30fps, 557帧）上的测试结果：

- **姿态检测率**：98%（49/50帧通过，实测100帧中99帧检测成功）
- **手动点追踪**：LK光流逐帧跟踪，参数可调；用户可根据需要标记任意物体
- **处理速度**：约5-8 fps（CPU，取决于机器性能）

## 已知限制

1. **2D测量**：单目视频无深度信息，垂直于像平面的运动会造成角度偏差
2. **遮挡处理**：长时间遮挡（>30帧）下线性插值质量下降；手动点遮挡后光流自动停用
3. **像素尺度**：默认 mm/px 换算系数需要实际标定才能获得精确物理距离
4. **多人场景**：当前 MediaPipe 配置为单人次检测
5. **GIF精度**：GIF无标准FPS字段，默认按30fps处理
6. **速度计算**：移动平均平滑会引入约2-3帧的滞后
7. **光流漂移**：LK光流跟踪的是纹理特征而非物体质心，物体旋转或变形时可能逐渐漂移

## 参考

- MediaPipe Pose Landmarker: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- Winter, D.A. (2009). *Biomechanics and Motor Control of Human Movement* (4th ed.). Wiley.
- Sports2D: https://github.com/sportstech/Sports2D
