# 心理健康多模态分析系统

音视频多模态处理系统，支持视频人脸特征提取（CLIP/DINO）和音频心理健康分析（焦虑、抑郁、自杀倾向等）。

## 项目结构

```
mental_preprocess/
├── viapi.py              # 视频特征提取 API 服务 (端口 10031)
├── auapi.py              # 音频心理健康分析 API 服务 (端口 10030)
├── tsapi.py              # 视频时间序列分析 API 服务 (端口 10032)
├── combinedapi.py        # 总体分析服务 - 视频+音频 (端口 10033)
├── facedetector.py       # 人脸检测模块 (MediaPipe)
├── audiopreprocess.py     # 音频预处理工具
├── preprocess.py         # 命令行视频预处理工具
├── model/                # MattingNetwork 模型定义
│   └── model.py
├── dinov2/              # DINOv2 视觉骨干网络
├── audiotrain/          # 音频模型训练框架
├── DeepFilterNet3/      # 音频降噪模型
├── Time-Series-Library/ # 时间序列分类模型库
├── ckpts/               # 预训练模型权重
└── README.md
```

## 核心功能

### 1. 视频特征提取 (viapi.py)

基于 GPU 硬件加速的视频处理流水线，提取多层级视觉特征：

| 模块 | 技术方案 | 输出 |
|------|---------|------|
| 视频解码 | PyNvCodec (GPU) / OpenCV (CPU) | 原始帧 tensor |
| 人脸检测 | MediaPipe (RetinaFace) | BBOX, Landmark (478点), Blendshape (52个) |
| 图像分割 | RVM (Robust Video Matting) MobileNetV3 | 前景抠图 |
| 特征编码 | CLIP ViT-B/32 + DINOv2 ViT-B/14 | 6种特征向量 |

**输出特征类型：**
- `src_clip` / `src_dino`：原始帧特征
- `wobg_clip` / `wobg_dino`：背景去除后特征
- `crop_clip` / `crop_dino`：人脸裁剪区域特征

### 2. 音频心理健康分析 (auapi.py)

从音频中提取梅尔频谱图特征，结合 ResNet18 进行多任务预测：

| 模块 | 技术方案 | 参数 |
|------|---------|------|
| 音频提取 | FFmpeg | wav 16kHz |
| 音频降噪 | DeepFilterNet3 | - |
| 频谱生成 | TorchAudio MelSpectrogram | n_fft=400, hop=160, n_mels=80 |
| 分类模型 | ResNet18 | 4 个独立模型 |

**支持任务：**
- `anxious` - 焦虑检测
- `depression` - 抑郁检测
- `overall` - 整体心理健康评估
- `zisha` - 自杀风险评估

### 3. 视频时间序列分类 (combinedapi.py 内置)

基于 Time-Series-Library 的 4 个最优模型，启动时预加载，从视频 CLIP 特征直接推理：

| 任务 | 模型 | seq_len | enc_in |
|------|------|---------|--------|
| depression | Informer | 256 | 64 |
| anxious | Reformer | 256 | 64 |
| zisha | FiLM | 256 | 64 |
| overall | DLinear | 256 | 64 |

模型配置从 checkpoint 文件夹名自动解析（复用 `mentalts/infer.py`），无需额外 JSON 配置文件。

## API 接口

### 总体分析（combinedapi.py，端口 10033）

启动时自动加载：DeepFilterNet 降噪 → 4个音频 ResNet18 → 4个时间序列模型。

```bash
python combinedapi.py

# 全任务预测（音频 + 视频）
curl -X POST "http://127.0.0.1:10033/predict" \
     -F "file=@视频.mp4" -F "task=all"

# 单任务
curl -X POST "http://127.0.0.1:10033/predict" \
     -F "file=@视频.mp4" -F "task=depression"
```

**任务：** `anxious` `depression` `overall` `zisha` `all`

**返回：**
```json
{
  "status": "success",
  "audio": {
    "depression": {"predicted_class": 0, "confidence": 0.91, "all_class_probabilities": [0.91, 0.09]}
  },
  "video": {
    "depression": {"predicted_class": 0, "confidence": 0.99, "all_class_probabilities": [0.99, 0.01]}
  }
}
```

**中间文件：** 视频特征和音频文件均在系统临时目录，请求结束自动清理。

### 视频特征提取（viapi.py，端口 10031）

```bash
# 启动服务
python viapi.py

# 调用接口
curl -X POST "http://127.0.0.1:10031/process_video" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@视频文件.mp4" \
     -F "case_id=patient_001" \
     -F "target_frames=512" \
     -F "csv_out=true" \
     -F "out_list=src_clip,src_dino,wobg_clip,wobg_dino,crop_clip,crop_dino"
```

**参数说明：**
- `case_id`：业务标识符，用于生成输出文件名前缀
- `target_frames`：目标采样帧数（默认1024）
- `csv_out`：是否输出 CSV 包含人脸关键点和表情系数
- `out_list`：要提取的特征类型，用逗号分隔

**返回示例：**
```json
{
  "status": "success",
  "case_id": "patient_001",
  "processed_frames": 512,
  "time_cost_seconds": 12.35,
  "saved_files": [
    "output/src_clip/patient_001.npy",
    "output/src_dino/patient_001.npy",
    ...
  ],
  "csv_file": "output/csv/patient_001.csv"
}
```



### 音频分析（auapi.py，端口 10030）

```bash
# 启动服务
python auapi.py


# 单任务预测
curl -X POST "http://127.0.0.1:10030/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@视频文件.mp4" \
     -F "task=depression"
## task: [anxious, depression, overall, zisha]

# 全任务预测
curl -X POST "http://127.0.0.1:10030/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@视频文件.mp4" \
     -F "task=all"
```

**返回示例：**
```json
{
  "status": "success",
  "results": {
    "anxious": {"class": 0, "confidence": 0.85},
    "depression": {"class": 1, "confidence": 0.72},
    "overall": {"class": 0, "confidence": 0.91},
    "zisha": {"class": 0, "confidence": 0.95}
  }
}
```

### 视频心理健康分析服务

基于时间序列模型的时间序列分类服务，从 `configs/` 目录自动读取配置：

```bash
# 启动服务
python tsapi.py

# 查询可用任务
curl http://127.0.0.1:10032/tasks

# 单任务预测
curl -X POST "http://127.0.0.1:10032/predict/anxious" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@特征文件.npy"

# 全任务预测
curl -X POST "http://127.0.0.1:10032/predict/all" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@特征文件.npy"
```

**支持任务：** `anxious`, `depression`, `overall`, `zisha`, `all`

**返回示例：**
```json
{
  "filename": "特征文件.npy",
  "task": "all",
  "results": {
    "anxious": {
      "predicted_class": 0,
      "confidence": 0.95,
      "all_class_probabilities": [0.95, 0.05]
    },
    "depression": {
      "predicted_class": 1,
      "confidence": 0.87,
      "all_class_probabilities": [0.13, 0.87]
    },
    "overall": {...},
    "zisha": {...}
  }
}
```

**配置文件格式：** 存放在 `configs/` 目录，文件名为 `{task_name}.json`
```json
{
    "task_name": "classification",
    "model": "TimesNet",
    "target": "anxious",
    "seq_len": 256,
    "enc_in": 512,
    "checkpoint_path": "./checkpoints/classification_anxious_timesnet/checkpoint.pth"
}
```

## 命令行使用

### 视频预处理

```bash
CUDA_VISIBLE_DEVICES=0 python preprocess.py <gpu_id> <input_video> <output_video> <model_type>
```

示例：
```bash
CUDA_VISIBLE_DEVICES=0 python preprocess.py 0 'input.mp4' output.mp4 Video2_351
```
CUDA_VISIBLE_DEVICES=0 python preprocess.py 0 '艾顺娇_森林.mp4' output-pyenvcodec.mp4 Video2_351



### 音频批量推理

```bash
python -m audiotrain.infer \
    --config audiotrain/audio_resnet_depression.json \
    --image <mel_spectrogram_png_path>
```

## 处理流程图

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   原始视频文件                        │
                    └─────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
         ┌──────────────────┐                      ┌──────────────────┐
         │   视频处理分支     │                      │   音频处理分支     │
         │  (Video Pipeline) │                      │ (Audio Pipeline)  │
         └──────────────────┘                      └──────────────────┘
                    │                                           │
         ┌──────────┴──────────┐                   ┌─────────────┴─────────────┐
         ▼                     ▼                   ▼                           ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐                  ┌─────────┐
    │ PyNvCodec│         │ OpenCV  │         │ FFmpeg  │                  │DeepFilter│
    │(GPU解码) │         │(CPU解码) │         │音频提取  │                  │Net降噪   │
    └────┬────┘         └────┬────┘         └────┬────┘                  └────┬────┘
         └──────────┬──────────┘                 └─────────────┬────────────────┘
                    ▼                                           ▼
         ┌────────────────────────────────────────────────────────────────────┐
         │                        人脸检测 (MediaPipe)                          │
         │                   BBOX | Landmark (478点) | Blendshape (52个)       │
         └────────────────────────────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
         ┌──────────────────┐                      ┌─────────────────────────┐
         │ 图像分割/抠图      │                      │   梅尔频谱图生成          │
         │ (RVM MobileNetV3) │                      │   (TorchAudio)          │
         └────────┬─────────┘                      └────────────┬────────────┘
                  │                                             │
                  ▼                                             ▼
         ┌────────────────────────────────────────────────────────────────────┐
         │                      视觉特征提取 (CLIP + DINOv2)                   │
         │     src_clip/dino → wobg_clip/dino → crop_clip/dino               │
         └────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
         ┌────────────────────────────────────────────────────────────────────┐
         │                    时间序列分类 (Time-Series-Library)               │
         │     DLinear / Autoformer / Transformer / TimesNet                 │
         └────────────────────────────────────────────────────────────────────┘

         ┌────────────────────────────────────────────────────────────────────┐
         │                      音频建模 (ResNet18)                           │
         │           anxious | depression | overall | zisha                   │
         └────────────────────────────────────────────────────────────────────┘
```

## 环境依赖

```
torch >= 2.0
PyNvCodec / PytorchNvCodec
opencv-python
ffmpeg
torchaudio
clip (OpenAI)
dinov2 (Meta)
MediaPipe
DeepFilterNet3
fastapi
uvicorn
Time-Series-Library (时间序列分类模型库)
```

## 预训练模型

模型权重文件位于 `ckpts/` 目录：

| 文件 | 用途 |
|------|------|
| `ViT-B-32.pt` | CLIP ViT-B/32 视觉编码器 |
| `dinov2_vitb14_reg4_pretrain.pth` | DINOv2 ViT-B/14 |
| `rvm_mobilenetv3.pth` | RVM 抠图模型 |
| `mobilenet0.25_Final.pth` | RetinaFace 人脸检测 |
| `face_landmarks_detector_*.pth` | 人脸关键点检测 |
| `face_blendshapes.pth` | 表情系数预测 |
| `anx_model_best.pth` | 音频焦虑检测模型 |
| `dep_model_best.pth` | 音频抑郁检测模型 |
| `all_model_best.pth` | 音频整体评估模型 |
| `su_model_best.pth` | 音频自杀风险模型 |
| `video/zisha.pth` | 视频自杀风险模型 |
| `video/dep.pth` | 视频抑郁检测模型 |
| `video/anx.pth` | 视频焦虑检测模型 |
| `video/all.pth` | 视频整体评估模型 |


## 输出文件

```
output/
├── src_clip/          # 原始帧 CLIP 特征
├── src_dino/          # 原始帧 DINO 特征
├── wobg_clip/         # 背景去除后 CLIP 特征
├── wobg_dino/         # 背景去除后 DINO 特征
├── crop_clip/         # 人脸裁剪 CLIP 特征
├── crop_dino/         # 人脸裁剪 DINO 特征
└── csv/
    └── <case_id>.csv  # 人脸关键点和表情系数
```

特征文件格式为 `.npy`，shape 为 `(帧数, 特征维度)`。

## 技术细节

### GPU 内存优化

- 批量处理：视频帧按 batch_size=12 批量推理
- 动态抽帧：通过 `target_frames` 参数控制采样数量
- 模型预加载：服务启动时一次性加载所有模型

### 视频解码流程

```
NV12 → YUV420 → RGB → RGB_PLANAR → CUDA Tensor
```

### 人脸特征维度

- Landmark：478 点 × 3D = 1434 维
- Blendshape：52 维表情系数
- CLIP 特征：512 维
- DINO 特征：768 维

### 视频心理健康分析技术细节

- **模型输入**：时间序列特征 (seq_len, feature_dim) 或 (batch_size, seq_len, feature_dim)
- **支持模型**：DLinear, Autoformer, Transformer, TimesNet 等
- **配置文件**：JSON 格式，包含模型参数和权重路径
- **推理流程**：
  1. 加载配置文件和模型权重
  2. 处理输入视频或特征文件
  3. 模型推理获取分类结果
  4. 返回预测类别和置信度
