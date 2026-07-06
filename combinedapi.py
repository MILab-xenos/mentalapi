import os
import sys
import json
import time
import tempfile
import subprocess
import importlib.util
import torch
import numpy as np
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn
from torchvision import transforms

ts_lib_path = os.path.join(os.path.dirname(__file__), 'mentalts')
sys.path.insert(0, ts_lib_path)

sys.path.insert(0, os.path.dirname(__file__))
# Ensure Time-Series-Library has highest priority for utils imports
sys.path.insert(0, ts_lib_path)
from df.enhance import enhance, init_df, load_audio

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

AUDIO_TASKS_CONFIG = {
    "anxious": {
        "config": "audiotrain/audio_resnet_anxious.json",
        "ckpt": "audiotrain/saved/models/Anxious_Cls/0203_081640/checkpoint-epoch10.pth"
    },
    "depression": {
        "config": "audiotrain/audio_resnet_depression.json",
        "ckpt": "audiotrain/saved/models/Depression_Cls/0203_080514/checkpoint-epoch10.pth"
    },
    "overall": {
        "config": "audiotrain/audio_resnet_overall.json",
        "ckpt": "audiotrain/saved/models/Overall_Cls/0203_075945/checkpoint-epoch10.pth"
    },
    "zisha": {
        "config": "audiotrain/audio_resnet_zisha.json",
        "ckpt": "audiotrain/saved/models/Zisha_Cls/0203_082701/checkpoint-epoch10.pth"
    }
}

GLOBAL_MODELS = {
    "df_model": None,
    "df_state": None,
    "audio_models": {},
    "ts_models": {},
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_audio_models():
    audio_path = os.path.join(os.path.dirname(__file__), 'audiotrain')
    sys.path.insert(0, audio_path)
    import model.model as mental_health_model
    for task_name, task_config in AUDIO_TASKS_CONFIG.items():
        with open(task_config["config"], 'r') as f:
            config = json.load(f)
        model = mental_health_model.MentalHealthResNet(
            num_classes=config['arch']['args']['num_classes']
        ).to(DEVICE)
        checkpoint = torch.load(task_config["ckpt"], map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        GLOBAL_MODELS["audio_models"][task_name] = model
    sys.path.pop(0)
    # Clear cached modules to avoid conflict with mentalts package
    for key in list(sys.modules.keys()):
        if key.startswith('model') or key.startswith('base') or key.startswith('parse_config') or key == 'utils' or key.startswith('utils.'):
            del sys.modules[key]

def load_ts_models():
    """Pre-load best-per-task TS classification models, selected by macro_f1 from CSV."""
    import pandas as pd
    import infer as ts_infer

    ts_ckpt_base = os.path.join(os.path.dirname(__file__), 'mentalts', 'checkpoints')
    csv_path = os.path.join(os.path.dirname(__file__), 'mentalts', 'classification_summary.csv')

    df = pd.read_csv(csv_path)
    # Filter to wobg_clip features (matching our video pipeline output)
    df = df[df['Setting'].str.contains('wobg_clip', na=False)]
    best_rows = df.loc[df.groupby('Target')['macro_f1'].idxmax()]

    TARGET_MAP = {'anxiety': 'anxious', 'depression': 'depression', 'suiside': 'zisha', 'overall': 'overall'}

    for _, row in best_rows.iterrows():
        task_name = TARGET_MAP.get(row['Target'], row['Target'])
        ckpt_dir = os.path.join(ts_ckpt_base, row['Setting'])
        print(f"-> 加载时间序列模型 [{task_name}] {row['Model']} (f1={row['macro_f1']:.4f})")
        model, args = ts_infer.load_model(ckpt_dir, DEVICE)
        GLOBAL_MODELS['ts_models'][task_name] = (model, args)
    print(f"-> 时间序列模型加载完成 ({len(GLOBAL_MODELS['ts_models'])} 个)")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("正在初始化服务...")
    print("-> 加载 DeepFilterNet 降噪模型")
    GLOBAL_MODELS["df_model"], GLOBAL_MODELS["df_state"], *_ = init_df("./DeepFilterNet3")
    print("-> 加载音频分析模型")
    load_audio_models()
    print("-> 加载时间序列分类模型")
    load_ts_models()
    print("-> 探测并选择最快的视频解码后端")
    _select_video_backend()
    print("服务准备就绪！")
    yield
    GLOBAL_MODELS.clear()

app = FastAPI(lifespan=lifespan, title="Combined Mental Health Analysis API")

def extract_audio_from_video(video_path, output_wav_path):
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        output_wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def denoise_audio(wav_path, output_png_path):
    waveform, sr = load_audio(wav_path, sr=16000)
    enhanced = enhance(GLOBAL_MODELS["df_model"], GLOBAL_MODELS["df_state"], waveform)
    enhanced = enhanced.cpu().numpy()

    import torchaudio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        n_fft=400, hop_length=160, n_mels=80
    )

    mel = mel_transform(torch.tensor(enhanced).unsqueeze(0))
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

    max_val = mel_db.max()
    if max_val > 0:
        mel_db = mel_db / max_val

    mel_np = mel_db.cpu().numpy()[0, 0]
    img = Image.fromarray((mel_np * 255).astype(np.uint8))
    img.save(output_png_path)

def predict_audio(task_name, mel_png_path):
    model = GLOBAL_MODELS["audio_models"].get(task_name)
    if model is None:
        raise HTTPException(status_code=400, detail=f"未知音频任务: {task_name}")

    img = Image.open(mel_png_path).convert('RGB')
    img_tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = torch.argmax(outputs, dim=1).item()
        confidence = probs[0, pred_class].item()
        all_probs = probs[0].cpu().numpy().tolist()

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "all_class_probabilities": [round(p, 4) for p in all_probs]
    }

# 视频后端候选：(名称, 模块文件, 说明)。viapi.py 用包名导入，其余按文件加载。
VIDEO_BACKENDS = [
    ("pyavgpu", "viapi-pyavgpu.py", "PyAV NVDEC GPU 硬解"),
    ("decord", "viapi-decord.py", "decord NVDEC GPU 硬解"),
    ("pyav", "viapi-pyav.py", "PyAV CPU 软解"),
]

# 选定的视频后端模块 (首次调用时探测 + 测速后缓存)
_SELECTED_VI = None


def _import_backend(name, filename):
    """按文件加载一个视频后端模块。viapi.py (PyNvCodec) 走包导入并先探测依赖。"""
    if name == "pynvcodec":
        import PyNvCodec  # noqa: F401  探测 NVDEC 硬解依赖是否可用
        import viapi as vi
        return vi
    path = os.path.join(os.path.dirname(__file__), filename)
    mod_name = "viapi_" + name
    spec = importlib.util.spec_from_file_location(mod_name, path)
    vi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vi)
    return vi


def _make_probe_clip(tmpdir):
    """生成一个短测试视频用于后端测速 (720p/60帧)。失败返回 None。"""
    clip_path = os.path.join(tmpdir, "backend_probe.mp4")
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc=duration=2:size=1280x720:rate=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", clip_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return clip_path if proc.returncode == 0 and os.path.exists(clip_path) else None


def _benchmark_backend(vi, clip_path):
    """对一个后端的解码器做一次解码计时。返回耗时(秒)，失败返回 None。"""
    try:
        gpu_id = getattr(vi, "GPU_ID", 0)
        dec = vi.CpuVideoDecoder(clip_path, gpu_id)
        t = time.time()
        n = 0
        for _ in dec.frames():
            n += 1
        if n == 0:
            return None
        return time.time() - t
    except Exception:
        return None


def _select_video_backend():
    """探测三种后端哪些能用，逐个测速，选最快的作为后端 (结果缓存)。"""
    global _SELECTED_VI
    if _SELECTED_VI is not None:
        return _SELECTED_VI

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_path = _make_probe_clip(tmpdir)

        results = []  # (name, desc, vi, elapsed)
        for name, filename, desc in VIDEO_BACKENDS:
            try:
                vi = _import_backend(name, filename)
            except Exception as e:
                print(f"-> 后端 [{name}] 不可用 (导入失败: {type(e).__name__}: {e})")
                continue

            elapsed = _benchmark_backend(vi, clip_path) if clip_path else None
            if elapsed is None:
                print(f"-> 后端 [{name}] 可导入但解码测试失败，跳过")
                continue
            print(f"-> 后端 [{name}] ({desc}) 解码测速: {elapsed:.3f}s")
            results.append((name, desc, vi, elapsed))

        if not results:
            raise RuntimeError("没有可用的视频解码后端 (pyavgpu/decord/pyav 均不可用)")

        # 选最快
        results.sort(key=lambda r: r[3])
        name, desc, vi, elapsed = results[0]
        print(f"=> 选定视频后端: [{name}] ({desc})，解码耗时 {elapsed:.3f}s 最快")
        _SELECTED_VI = vi
        return vi


def _load_viapi_module():
    """返回选定的视频处理后端 (首次运行时测速选最快，之后复用)。"""
    return _select_video_backend()

def get_video_features(video_path, case_id, tmpdir, target_frames=512):
    vi = _load_viapi_module()
    VI_GLOBAL_MODELS = vi.GLOBAL_MODELS
    CLIP_MEAN, CLIP_STD = vi.CLIP_MEAN, vi.CLIP_STD
    DINO_MEAN, DINO_STD = vi.DINO_MEAN, vi.DINO_STD
    process_video_pipeline = vi.process_video_pipeline
    device = vi.device
    GPU_ID = vi.GPU_ID

    torch.cuda.set_device(GPU_ID)

    if VI_GLOBAL_MODELS["detector"] is None:
        from facedetector import MPDetector
        VI_GLOBAL_MODELS["detector"] = MPDetector(device=device)

    if VI_GLOBAL_MODELS["matting"] is None:
        from model import MattingNetwork
        matting = MattingNetwork(variant='mobilenetv3').eval().to(device)
        matting.load_state_dict(torch.load('./ckpts/rvm_mobilenetv3.pth', map_location=device, weights_only=False))
        VI_GLOBAL_MODELS["matting"] = matting

    if VI_GLOBAL_MODELS["clip_model"] is None:
        import clip
        clip_model, _ = clip.load("ViT-B/32", download_root="./ckpts", device=device)
        clip_model.eval()
        VI_GLOBAL_MODELS["clip_model"] = clip_model

    if VI_GLOBAL_MODELS["dino_model"] is None:
        from dinov2.hub import dinov2_vitb14_reg
        dino_model = dinov2_vitb14_reg(pretrained=False)
        state_dict = torch.load("./ckpts/dinov2_vitb14_reg4_pretrain.pth", map_location=device, weights_only=False)
        dino_model.load_state_dict(state_dict)
        dino_model.to(device).eval()
        VI_GLOBAL_MODELS["dino_model"] = dino_model

    result = process_video_pipeline(
        encFilePath=video_path,
        out_list=["crop_clip", "wobg_clip"],
        csv_out=False,
        case_id=case_id,
        target_frames=target_frames,
        output_dir=tmpdir,
    )
    return result

def predict_video_ts(features_path, task_name):
    """Run time-series classification on extracted video features.
    Uses pre-loaded models with infer.py's infer_one logic."""
    import infer as ts_infer

    if task_name == "all":
        results = {}
        for task in GLOBAL_MODELS["ts_models"].keys():
            results[task] = _predict_single_ts(ts_infer, features_path, task)
        return results
    else:
        return {task_name: _predict_single_ts(ts_infer, features_path, task_name)}

def _predict_single_ts(ts_infer, features_path, task_name):
    """Run a single TS classification task using pre-loaded model + infer.py's infer_one."""
    entry = GLOBAL_MODELS["ts_models"].get(task_name)
    if not entry:
        raise HTTPException(status_code=400, detail=f"未知视频任务: {task_name}")

    model, args = entry
    arr = np.load(features_path)
    pred, probs = ts_infer.infer_one(model, args, arr, DEVICE)

    return {
        "predicted_class": pred,
        "confidence": round(float(probs[pred]), 4),
        "all_class_probabilities": [round(float(p), 4) for p in probs]
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="上传视频文件"),
    task: str = Form("all", description="预测任务: anxious, depression, overall, zisha, all")
):
    available_tasks = ["anxious", "depression", "overall", "zisha"]
    if task != "all" and task not in available_tasks:
        raise HTTPException(status_code=400, detail=f"不支持的任务，请选择: {available_tasks + ['all']}")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "input.mp4")
        case_id = "combined_analysis"

        with open(video_path, "wb") as f:
            f.write(await file.read())

        audio_wav_path = os.path.join(tmpdir, "audio.wav")
        audio_png_path = os.path.join(tmpdir, "mel.png")

        extract_audio_from_video(video_path, audio_wav_path)
        denoise_audio(audio_wav_path, audio_png_path)

        video_features_result = get_video_features(video_path, case_id, tmpdir, target_frames=512)
        video_features_path = video_features_result["saved_files"][0] if video_features_result["saved_files"] else None

        audio_results = {}
        tasks_to_run = available_tasks if task == "all" else [task]
        for task_name in tasks_to_run:
            audio_results[task_name] = predict_audio(task_name, audio_png_path)

        video_ts_results = {}
        if video_features_path:
            video_ts_results = predict_video_ts(video_features_path, task)

        return {
            "status": "success",
            "filename": file.filename,
            "audio": audio_results,
            "video": video_ts_results
        }

@app.get("/tasks")
def list_tasks():
    return {
        "available_tasks": ["anxious", "depression", "overall", "zisha", "all"]
    }

if __name__ == "__main__":
    uvicorn.run("combinedapi:app", host="0.0.0.0", port=10033, reload=False)