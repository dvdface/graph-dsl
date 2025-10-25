# Graph-DSL: 基于图DSL的视频变化检测系统

一个功能强大的视频变化检测框架，提供流式API和智能状态管理，支持多种数据源和检测方法。

## 🎯 核心功能

- **视频变化检测** - 使用网格化方法检测视频帧之间的变化
- **多种检测方法** - 支持SSIM（结构相似性）和绝对差值两种比较方法
- **智能状态管理** - 支持"开始检测"、"停止检测"和"完整检测"三种模式
- **区域控制** - 支持包含/排除特定区域的变化检测
- **多数据源支持** - 支持视频文件、图像目录、摄像头三种输入源
- **并行处理** - 使用多线程并行计算网格变化，提高处理效率
- **调试可视化** - 提供调试视频输出和详细日志记录

## 🚀 快速开始

### 安装依赖

```bash
pip install opencv-python numpy tqdm ultralytics scikit-image
```

### 基本使用

```python
from graph.dsl import Data

# 基本变化检测
result = (
    Data("resource/01.mp4")
    .config(method="ssim", threshold=0.1, grid_size=(32, 16))
    .detect(type="both")
    .run()
)

print(result)
```

### 高级配置

```python
result = (
    Data("resource/01.mp4")
    .config(method="ssim", threshold=0.1, grid_size=(32, 16))
    .detect(type="both",
        start_exclude_area=[(0.0, 0.0, 1.0, 0.2)],  # 排除顶部20%区域
        stop_exclude_area=[(0.0, 0.0, 1.0, 0.2)]
    )
    .debug(output_dir="debug", filename="debug.mp4")
    .verbose(output_dir="logs", filename="verbose.txt")
    .run()
)
```

## 📊 主要组件

### 1. Data类 - 核心DSL接口
提供链式调用的DSL接口，支持：
- `.config()` - 配置检测参数
- `.detect()` - 设置检测区域和类型
- `.roi()` - 定义感兴趣区域
- `.filter()` - 应用图像过滤
- `.debug()` - 启用调试模式
- `.verbose()` - 启用详细日志

### 2. ProcessingEngine - 处理引擎
- 执行处理管道的核心引擎
- 实现状态机管理检测流程
- 处理并行网格计算和结果格式化

### 3. Source系统 - 数据源管理
支持多种输入源：
- **FileSource** - 视频文件输入
- **DirSource** - 图像目录输入
- **CameraSource** - 摄像头实时输入

### 4. GridWorker - 并行计算
- 并行网格计算工作器
- 实现SSIM和绝对差值两种检测算法
- 提供图像处理工具函数

## 🔧 API参考

### 配置方法

```python
.config(method="ssim", threshold=0.1, grid_size=(32, 16))
```

**参数：**
- `method`: 检测方法 ("ssim" 或 "abs")
- `threshold`: 变化阈值 (0-1)
- `grid_size`: 网格大小 (行, 列)

### 检测方法

```python
.detect(type="both", 
        include_area=[(0.0, 0.0, 1.0, 1.0)],
        exclude_area=[(0.0, 0.0, 1.0, 0.2)],
        start_include_area=[...],
        start_exclude_area=[...],
        stop_include_area=[...],
        stop_exclude_area=[...])
```

**参数：**
- `type`: 检测类型 ("start", "stop", "both")
- `include_area`: 通用包含区域
- `exclude_area`: 通用排除区域
- `start_*`: 开始检测专用区域
- `stop_*`: 停止检测专用区域

### 调试和日志

```python
.debug(output_dir="debug", filename="debug.mp4", delay_stop_frames=10)
.verbose(output_dir="logs", filename="verbose.txt")
```

## 📈 输出格式

系统输出结构化的JSON结果：

```json
{
  "params": {
    "change_method": "ssim",
    "change_threshold": 0.1,
    "grid_size": [32, 16],
    "detect_type": "both"
  },
  "result": {
    "123": {
      "offset": 1234.56,
      "ts": 1234567890.12,
      "change": [
        [0.125, 0.25, 0.25, 0.5, 0.85]
      ]
    }
  }
}
```

## 🎯 应用场景

- **视频监控** - 运动检测和异常监控
- **内容分析** - 视频内容变化分析
- **实时监控** - 摄像头变化监控
- **质量评估** - 视频质量比较和评估

## 🔍 检测方法

### SSIM (结构相似性)
- 基于结构相似性指数
- 对光照变化不敏感
- 适合检测结构变化

### 绝对差值
- 基于像素级绝对差值
- 对细微变化敏感
- 适合检测像素级变化

## 📝 示例项目

查看 `main.py` 文件了解完整的使用示例。

## 🛠️ 开发

### 项目结构
```
graph-dsl/
├── graph/
│   └── dsl/
│       ├── __init__.py      # 模块入口
│       ├── core.py          # 核心Data类和Node
│       ├── processing.py    # 处理引擎
│       ├── source.py        # 数据源管理
│       └── utils.py         # 工具函数
├── main.py                  # 示例程序
├── pyproject.toml           # 项目配置
└── README.md               # 项目说明
```

### 依赖要求
- Python >= 3.12.11
- OpenCV >= 4.5.0
- NumPy >= 1.20.0
- scikit-image >= 0.18.0

## 📄 许可证

本项目采用MIT许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！
