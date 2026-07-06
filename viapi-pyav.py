import sys
import os
import csv
import time
import tempfile
from contextlib import asynccontextmanager
from types import SimpleNamespace

# --- CUDA featDLL 配置 (针对 Windows) ---
if os.name == "nt":
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file=sys.stderr)
        
    sys_path = os.environ.get("PATH", "")
    if sys_path:
        for path in sys_path.split(";"):
            if os.path.isdir(path):
                os.add_dll_directory(path)

import torch
import torch.nn.functional as F
import numpy as np
import av  # CPU 解码 (替代 PyNvCodec，Tesla M40 无 NVDEC 硬解单元)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn

# --- 引入自定义模型和外部模型 ---
import clip
from dinov2.hub import dinov2_vitb14_reg
from facedetector import MPDetector 
from model import MattingNetwork

# ================= 1. 全局配置与模型存储 =================

device = "cuda" if torch.cuda.is_available() else "cpu"
# 运行时自动获取当前 GPU ID
GPU_ID = torch.cuda.current_device() if torch.cuda.is_available() else 0

GLOBAL_MODELS = {
    "detector": None,
    "matting": None,
    "clip_model": None,
    "dino_model": None
}

# 预处理常量
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


# ================= 2. 服务启动预加载 =================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 正在预加载所有模型...")
    tp = time.time()
    
    # 1. Face Detector
    print("-> 加载 MPDetector")
    GLOBAL_MODELS["detector"] = MPDetector(device=device)
    
    # 2. Matting Network
    print("-> 加载 MattingNetwork")
    matting = MattingNetwork(variant='mobilenetv3').eval().to(device)
    matting.load_state_dict(torch.load('./ckpts/rvm_mobilenetv3.pth', map_location=device))
    GLOBAL_MODELS["matting"] = matting

    # 3. CLIP
    print("-> 加载 CLIP ViT-B/32")
    clip_model, _ = clip.load("ViT-B/32", download_root="./ckpts", device=device)
    clip_model.eval()
    GLOBAL_MODELS["clip_model"] = clip_model

    # 4. DINOv2
    print("-> 加载 DINOv2")
    dino_model = dinov2_vitb14_reg(pretrained=False)
    state_dict = torch.load("./ckpts/dinov2_vitb14_reg4_pretrain.pth", map_location=device)
    dino_model.load_state_dict(state_dict)
    dino_model.to(device).eval()
    GLOBAL_MODELS["dino_model"] = dino_model

    print(f"🎉 所有模型加载完毕，耗时 {time.time() - tp:.2f} 秒！")
    yield
    GLOBAL_MODELS.clear()


app = FastAPI(lifespan=lifespan, title="Video Feature Extraction API")

# ================= 3. 核心辅助函数 =================

class CpuVideoDecoder:
    """基于 PyAV 的 CPU 视频解码器，替代 PyNvCodec。

    对外接口与原 nvDec 对齐：Width()/Height()/Numframes() 以及逐帧迭代。
    每帧解码为 RGB 后转成 GPU 上的 (3, H, W)、[0,1] float 张量，
    与原 surface_to_tensor 的输出格式保持一致，下游逻辑无需改动。
    """
    def __init__(self, enc_file_path: str, gpu_id: int):
        self.path = enc_file_path
        self.gpu_id = gpu_id
        self.torch_device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        container = av.open(enc_file_path)
        stream = container.streams.video[0]
        self._w = stream.codec_context.width
        self._h = stream.codec_context.height
        # 优先用容器元数据估算总帧数，拿不到时回退为 0（由上层空转统计）
        self._num_frames = stream.frames or 0
        if self._num_frames <= 0 and stream.duration and stream.average_rate:
            self._num_frames = int(float(stream.duration * stream.time_base) * float(stream.average_rate))
        container.close()

    def Width(self) -> int:
        return self._w

    def Height(self) -> int:
        return self._h

    def Numframes(self) -> int:
        return self._num_frames

    def frames(self):
        """逐帧生成 GPU 上的 (3, H, W)、[0,1] float 张量。"""
        container = av.open(self.path)
        try:
            for frame in container.decode(video=0):
                rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
                tensor = torch.from_numpy(rgb).to(self.torch_device)
                tensor = tensor.permute(2, 0, 1).float().div_(255.0).clamp_(0.0, 1.0)
                yield tensor
        finally:
            container.close()

def fe_tensor(in_tensor, fe_model, fe_mean, fe_std, batch_size=512, mode="clip"):
    in_tensor = F.interpolate(in_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor(fe_mean).view(3, 1, 1).to(in_tensor.device)
    std = torch.tensor(fe_std).view(3, 1, 1).to(in_tensor.device)
    in_tensor = (in_tensor - mean) / std
    
    with torch.no_grad():
        all_features =[]
        for i in range(0, in_tensor.size(0), batch_size):
            batch = in_tensor[i:i + batch_size]
            if mode == "clip":
                image_features = fe_model.encode_image(batch)
            elif mode == "dino":
                image_features = fe_model(batch)
            all_features.append(image_features)
        all_features = torch.cat(all_features, dim=0)
    return all_features.cpu().numpy()

def convert_tensor_to_landmarks(landmarks_3d):
    return[SimpleNamespace(x=float(x.item())/1920, y=float(y.item())/1080) for x, y in landmarks_3d[:, :2]]

def draw_landmark_on_tensor(landmarks, src_tensor, color='red'):
    color_tensor = torch.tensor([0.0, 1.0, 0.0] if color == 'green' else[1.0, 0.0, 0.0], device=src_tensor.device).view(3, 1) 
    if landmarks:
        x_coords = (torch.tensor([lm.x for lm in landmarks], device=src_tensor.device) * src_tensor.shape[2]).long()
        y_coords = (torch.tensor([lm.y for lm in landmarks], device=src_tensor.device) * src_tensor.shape[1]).long()
        
        valid_mask = (x_coords >= 0) & (x_coords < src_tensor.shape[2]) & (y_coords >= 0) & (y_coords < src_tensor.shape[1])
        valid_x, valid_y = x_coords[valid_mask], y_coords[valid_mask]
        
        if len(valid_x) > 0:
            src_tensor[:, valid_y, valid_x] = color_tensor
            min_x, max_x = valid_x.min(), valid_x.max()
            min_y, max_y = valid_y.min(), valid_y.max()
            src_tensor[:, min_y, min_x:max_x+1] = color_tensor
            src_tensor[:, max_y, min_x:max_x+1] = color_tensor
            src_tensor[:, min_y:max_y+1, min_x] = color_tensor
            src_tensor[:, min_y:max_y+1, max_x] = color_tensor
    return src_tensor


# ================= 4. 处理流水线 =================

def process_video_pipeline(
    encFilePath: str, 
    out_list: list, csv_out: bool, 
    case_id: str, target_frames: int,
    output_dir: str = "output",
):
    tp = time.time()
    
    # 提取预加载模型
    detector12 = GLOBAL_MODELS["detector"]
    model = GLOBAL_MODELS["matting"]
    clip_model = GLOBAL_MODELS["clip_model"]
    dino_model = GLOBAL_MODELS["dino_model"]

    # 使用全局运行时 GPU_ID (CPU 解码 + GPU 推理)
    nvDec = CpuVideoDecoder(encFilePath, GPU_ID)

    csv_file, csv_writer = None, None
    if csv_out:
        os.makedirs(f"{output_dir}/csv", exist_ok=True)
        csv_filename = os.path.join(f'{output_dir}/csv', f"{case_id}.csv")
        csv_file = open(csv_filename, 'w', newline='')

    w, h = nvDec.Width(), nvDec.Height()
    total_frames = nvDec.Numframes()

    # 若无法直接读取帧数，空转获取总数
    if total_frames <= 0:
        true_frames = 0
        for _ in nvDec.frames():
            true_frames += 1
        total_frames = true_frames

    # 计算采样抽帧索引
    frame_indices = None
    if target_frames is not None and target_frames > 0:
        if total_frames > target_frames:
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        else:
            indices = np.arange(total_frames)
        frame_indices = set(indices.tolist())

    # CPU 解码已直接产出 RGB planar 张量，无需额外的色彩空间转换链
    bgr = torch.tensor([0, 0, 0]).view(3, 1, 1).to(device)
    rec = [None] * 4

    # 特征存储队列
    src_clip_q, src_dino_q = [],[]
    wobg_clip_q, wobg_dino_q = [],[]
    crop_clip_q, crop_dino_q = [],[]
    
    batch_size = 12
    frame_buffer =[]
    frame_idx = 0

    try:
        for src_tensor in nvDec.frames():
            # 筛选帧
            if frame_indices is not None and frame_idx not in frame_indices:
                frame_idx += 1
                continue

            frame_buffer.append(src_tensor)
            
            # 当缓存达到 batch_size 开始模型推理
            if len(frame_buffer) >= batch_size:
                stacked_tensor = torch.stack(frame_buffer, dim=0)
                
                # 原始帧特征提取 (Src Features)
                if 'src_clip' in out_list: src_clip_q.append(fe_tensor(stacked_tensor, clip_model, CLIP_MEAN, CLIP_STD, 1, "clip"))
                if 'src_dino' in out_list: src_dino_q.append(fe_tensor(stacked_tensor, dino_model, DINO_MEAN, DINO_STD, 1, "dino"))
                
                with torch.no_grad():
                    # 1. 人脸检测与关键点
                    boxes, landmarks_3d, blendshapes = detector12.detect(stacked_tensor*255, data_type="tensor", progress_bar=False)
                    
                    # 2. 抠取人像背景 (Matting)
                    fgr, pha, *rec = model(stacked_tensor.to(device), *rec, downsample_ratio=0.25)
                    dst_tensor = fgr * pha + bgr * (1 - pha)
                    
                    if 'wobg_clip' in out_list: wobg_clip_q.append(fe_tensor(dst_tensor, clip_model, CLIP_MEAN, CLIP_STD, 1, "clip"))
                    if 'wobg_dino' in out_list: wobg_dino_q.append(fe_tensor(dst_tensor, dino_model, DINO_MEAN, DINO_STD, 1, "dino"))

                    # 3. 裁剪并缩放面部区域 (Crop)
                    batch_cropped = torch.zeros_like(dst_tensor)
                    for i in range(dst_tensor.size(0)):
                        landmarks = convert_tensor_to_landmarks(landmarks_3d[i])
                        dst_tensor[i] = draw_landmark_on_tensor(landmarks, dst_tensor[i], color='red')
                        face_blendshapes = blendshapes[i]
                        
                        # 仅在第一次写入 CSV 表头
                        if csv_out and csv_writer is None:
                            landmark_headers = sum([[f'x{kk}', f'y{kk}'] for kk in range(len(landmarks))],[])
                            blendshape_headers =[f'index{kk}' for kk in range(face_blendshapes.shape[0])]
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow(landmark_headers + blendshape_headers)
                            
                        # 写入该帧数据
                        if csv_out:
                            lm_data = sum([[f"{lm.x:.4f}", f"{lm.y:.4f}"] for lm in landmarks], [])
                            bs_data =[f"{bs.item():.4f}" for bs in face_blendshapes]
                            csv_writer.writerow(lm_data + bs_data)
                            
                        # 处理人脸裁剪
                        if boxes[i].shape[0] > 0:
                            box = boxes[i]
                            x1, x2 = max(int(box[0].item()) - 6, 0), min(int(box[2].item()) + 6, dst_tensor.shape[3])
                            y1, y2 = max(int(box[1].item()) - 8, 0), min(int(box[3].item()) + 8, dst_tensor.shape[2])
                            
                            face = dst_tensor[i, :, y1:y2, x1:x2]
                            face_h, face_w = face.shape[1:]
                            target_h, target_w = dst_tensor.shape[2], dst_tensor.shape[3]
                            scale = min(target_w / face_w, target_h / face_h)
                            new_w, new_h = int(face_w * scale), int(face_h * scale)
                            
                            resized_face = F.interpolate(face.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
                            y_offset, x_offset = (target_h - new_h) // 2, (target_w - new_w) // 2
                            batch_cropped[i, :, y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_face
                            
                    if 'crop_clip' in out_list: crop_clip_q.append(fe_tensor(batch_cropped, clip_model, CLIP_MEAN, CLIP_STD, 1, "clip"))
                    if 'crop_dino' in out_list: crop_dino_q.append(fe_tensor(batch_cropped, dino_model, DINO_MEAN, DINO_STD, 1, "dino"))
                            
                frame_buffer =[]

            frame_idx += 1

    finally:
        if csv_file: csv_file.close()

    # ====== 拼接并保存所有的 NPY 特征 ======
    saved_paths =[]
    
    def save_npy(feature_list, folder):
        if feature_list:
            os.makedirs(f"{output_dir}/{folder}", exist_ok=True)
            path = f"{output_dir}/{folder}/{case_id}.npy"
            np.save(path, np.concatenate(feature_list, axis=0))
            saved_paths.append(path)

    save_npy(src_clip_q, "src_clip")
    save_npy(src_dino_q, "src_dino")
    save_npy(wobg_clip_q, "wobg_clip")
    save_npy(wobg_dino_q, "wobg_dino")
    save_npy(crop_clip_q, "crop_clip")
    save_npy(crop_dino_q, "crop_dino")

    return {
        "status": "success",
        "case_id": case_id,
        "processed_frames": len(frame_indices) if frame_indices else total_frames,
        "time_cost_seconds": round(time.time() - tp, 2),
        "saved_files": saved_paths,
        "csv_file": f"{output_dir}/csv/{case_id}.csv" if csv_out else None
    }


# ================= 5. API 路由 =================

@app.post("/process_video")
async def process_video_endpoint(
    file: UploadFile = File(...),
    case_id: str = Form(..., description="业务ID，用作生成文件的前缀"),
    target_frames: int = Form(1024, description="采样目标帧数"),
    csv_out: bool = Form(True, description="是否输出 CSV 包含 landmarks"),
    out_list: str = Form("src_clip,src_dino,wobg_clip,wobg_dino,crop_clip,crop_dino", description="提取的特征(用逗号分隔)")
):
    # 解析所需的特征数组
    feature_list =[x.strip() for x in out_list.split(",") if x.strip()]
    
    # 因为 PyNvCodec 必须读取物理文件，所以存入临时目录，请求结束后自动删除
    with tempfile.TemporaryDirectory() as temp_dir:
        input_mp4 = os.path.join(temp_dir, f"{case_id}_input.mp4")
        
        with open(input_mp4, "wb") as f:
            f.write(await file.read())
            
        try:
            result = process_video_pipeline(
                encFilePath=input_mp4, 
                out_list=feature_list, 
                csv_out=csv_out, 
                case_id=case_id, 
                target_frames=target_frames
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 使用 Uvicorn 运行服务
    uvicorn.run("viapi:app", host="0.0.0.0", port=10031, reload=False)