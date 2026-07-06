# 心理健康多模态分析

上传视频，返回音频 + 视频时间序列的分类结果。

## 启动

```bash
python combinedapi.py
```

服务启动时自动加载：DeepFilterNet 降噪 → 4 个音频 ResNet18 → 4 个时间序列模型。

## 调用

```bash
# 全任务（音频 + 视频）
curl -X POST http://127.0.0.1:10033/predict -F "file=@视频.mp4" -F "task=all"

# 单任务
curl -X POST http://127.0.0.1:10033/predict -F "file=@视频.mp4" -F "task=depression"
```

任务：`anxious` `depression` `overall` `zisha` `all`

## 返回

```json
{
  "status": "success",
  "audio": {
    "depression": {
      "predicted_class": 0,
      "confidence": 0.91,
      "all_class_probabilities": [0.91, 0.09]
    },
    "anxious": {
      "predicted_class": 0,
      "confidence": 0.85,
      "all_class_probabilities": [0.85, 0.15]
    }
  },
  "video": {
    "depression": {
      "predicted_class": 0,
      "confidence": 0.99,
      "all_class_probabilities": [0.99, 0.01]
    },
    "anxious": {
      "predicted_class": 0,
      "confidence": 1.0,
      "all_class_probabilities": [1.0, 0.0]
    }
  }
}
```

- `predicted_class` — 0 = No, 1 = Yes
- `all_class_probabilities` — `[P(No), P(Yes)]`
- 中间文件均在临时目录，请求结束自动清理
