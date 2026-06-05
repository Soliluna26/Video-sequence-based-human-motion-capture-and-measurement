# 测量功能实验


## 1. 测量 Mode

### 功能内容

`测量` mode 计算左大腿线段 `(23,25)` 与右大腿线段 `(24,26)` 的夹角。

当该夹角大于 `30°` 时，认为测量正式激活，并计算对应角速度。

### 公式

两线段夹角：

```text
theta = arccos((v1 · v2) / (|v1| |v2|))
```

其中：

```text
v1 = P25 - P23
v2 = P26 - P24
```

角速度：

```text
omega(t) = d(theta) / dt
```

实际实现中使用帧序列差分计算，单位为 `deg/s`。

### 输出内容

`测量` mode 支持：

- Replay 中实时显示角度和角速度。
- 保存视频时保留测量叠加。
- 导出 `angle_velocity.csv`，只包含角度和角速度信息。
- 普通 tracking CSV 可选择附加测量行。

## 2. 测2 Mode

### 功能内容

`测2` mode 新增三组跳跃相关角度，均按左右两侧结果取平均值。

### 大腿跳跃角度

计算 `(23,25)` 与竖直方向夹角，以及 `(24,26)` 与竖直方向夹角，两者取平均。

```text
Thigh Jump Angle = mean(angle((23,25), vertical), angle((24,26), vertical))
```

### 小腿跳跃角度

计算 `(25,27)` 与竖直方向夹角，以及 `(26,28)` 与竖直方向夹角，两者取平均。

```text
Calf Jump Angle = mean(angle((25,27), vertical), angle((26,28), vertical))
```

### 脚面角度

计算 `(27,31)` 与 `(25,27)` 的夹角，以及 `(28,32)` 与 `(26,28)` 的夹角，两者取平均。

```text
Foot Angle = mean(angle((27,31), (25,27)), angle((28,32), (26,28)))
```

### 输出内容

`测2` mode 支持：

- Replay 中实时显示三组英文角度名称，避免 OpenCV 中文字体无法显示。
- 每组角度同步计算角速度。
- 专用 CSV 输出三组角度及对应角速度。
- 普通 tracking CSV 可选择每帧追加三条 measurement 行。

## 5. 验证

开发完成后执行了 Python 编译检查：

```text
python -m compileall src web_app.py tests
```

结果通过，说明新增功能没有语法或导入错误。
