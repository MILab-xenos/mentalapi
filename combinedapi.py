import os
import sys
import json
import tempfile
import subprocess
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
    """Pre-load all 4 best-per-task TS classification models using infer.py patterns."""
    import infer as ts_infer

    ts_ckpt_base = os.path.join(os.path.dirname(__file__), 'mentalts', 'checkpoints')

    BEST_CKPTS = {
        'depression': 'classification_dep_cls_wobg_clip_seq256_enc64_lr0.0002_Informer_Mental_ftwobg_clip_sl256_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_seq256_wobg_clip_lr0.0002_0',
        'anxious':   'classification_anx_cls_wobg_clip_seq256_enc64_lr0.0002_Reformer_Mental_ftwobg_clip_sl256_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_seq256_wobg_clip_lr0.0002_0',
        'zisha':     'classification_sui_cls_wobg_clip_seq256_enc64_lr0.0002_FiLM_Mental_ftwobg_clip_sl256_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_seq256_wobg_clip_lr0.0002_0',
        'overall':   'classification_ovr_cls_wobg_clip_seq256_enc64_lr0.0002_DLinear_Mental_ftwobg_clip_sl256_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_seq256_wobg_clip_lr0.0002_0',
    }

    for task_name, ckpt_rel in BEST_CKPTS.items():
        ckpt_dir = os.path.join(ts_ckpt_base, ckpt_rel)
        print(f"-> 加载时间序列模型 [{task_name}]: {ckpt_rel[:50]}...")
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

def get_video_features(video_path, case_id, tmpdir, target_frames=512):
    from viapi import (
        GLOBAL_MODELS as VI_GLOBAL_MODELS,
        CLIP_MEAN, CLIP_STD, DINO_MEAN, DINO_STD,
        process_video_pipeline, device, GPU_ID
    )

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