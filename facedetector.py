"""
This is an experimental module written primarily by @ljchang porting over
Tensorflow's MediaPipe face mesh model to PyTorch for better real-time
performance. It is not currently recommended for use. See this (closed) PR
for more discussion: https://github.com/cosanlab/py-feat/pull/228
"""

import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import Adam
from feat.data import Fex, ImageDataset, TensorDataset, VideoDataset
from skops.io import load, get_untrusted_types
from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
from feat.pretrained import AU_LANDMARK_MAP
from torch.utils.data import DataLoader
from PIL import Image
from feat.face_detectors.Retinaface.Retinaface_model import (
    RetinaFace,
    postprocess_retinaface,
)
from feat.au_detectors.MP_Blendshapes.MP_Blendshapes_test import (
    MediaPipeBlendshapesMLPMixer,
)
# from feat.identity_detectors.facenet.facenet_model import InceptionResnetV1
# from feat.emo_detectors.ResMaskNet.resmasknet_test import (
#     ResMasking,
# )
# from feat.emo_detectors.StatLearning.EmoSL_test import EmoSVMClassifier
from feat.utils import (
    set_torch_device,
    FEAT_EMOTION_COLUMNS,
    FEAT_FACEBOX_COLUMNS,
    FEAT_FACEPOSE_COLUMNS_6D,
    FEAT_IDENTITY_COLUMNS,
    MP_LANDMARK_COLUMNS,
    MP_BLENDSHAPE_NAMES,
    MP_BLENDSHAPE_MODEL_LANDMARKS_SUBSET,
)
from feat.utils.image_operations import (
    convert_image_to_tensor,
    convert_color_vector_to_tensor,
    extract_face_from_bbox_torch,
    inverse_transform_landmarks_torch,
    extract_hog_features,
    convert_bbox_output,
    compute_original_image_size,
)
from feat.utils.io import get_resource_path
from feat.utils.mp_plotting import FaceLandmarksConnections


def get_camera_intrinsics(batch_hw_tensor, focal_length=None):
    """
    Computes the camera intrinsic matrix for a batch of images.

    Args:
        batch_hw_tensor (torch.Tensor): A tensor of shape [B, 2] where B is the batch size, and each entry contains [H, W] for the height and width of the images.
        focal_length (torch.Tensor, optional): A tensor of shape [B] representing the focal length for each image in the batch. If None, the focal length will default to the image width for each image.

    Returns:
        K (torch.Tensor): A tensor of shape [B, 3, 3] containing the camera intrinsic matrices for each image in the batch.
    """
    # Extract the batch size
    batch_size = batch_hw_tensor.shape[0]

    # Extract heights and widths
    heights = batch_hw_tensor[:, 0]
    widths = batch_hw_tensor[:, 1]

    # If focal_length is not provided, default to image width for each image
    if focal_length is None:
        focal_length = widths  # [B]

    # Initialize the camera intrinsic matrices
    K = torch.zeros((batch_size, 3, 3), dtype=torch.float32)

    # Populate the intrinsic matrices
    K[:, 0, 0] = focal_length  # fx
    K[:, 1, 1] = focal_length  # fy
    K[:, 0, 2] = widths / 2  # cx
    K[:, 1, 2] = heights / 2  # cy
    K[:, 2, 2] = 1.0  # The homogeneous coordinate

    return K


def convert_landmarks_3d(fex):
    """
    Converts facial landmarks from a feature extraction object into a 3D tensor.

    Args:
        fex (Fex): Fex DataFrame containing 478 3D landmark coordinates

    Returns:
        landmarks (torch.Tensor): A tensor of shape [batch_size, 478, 3] containing the 3D coordinates (x, y, z) of 478 facial landmarks for each instance in the batch.
    """

    return torch.tensor(fex.landmarks.astype(float).values).reshape(fex.shape[0], 478, 3)


def estimate_gaze_direction(fex, gaze_angle="combined", metric="radians"):
    """
    Estimates the gaze direction based on the 3D facial landmarks of the eyes and irises.

    NOTES: This could eventually be added as Fex Method

    Args:
        fex (Fex): Fex DataFrame containing 478 3D landmark coordinates
        gaze_angle (str): Specifies which gaze angle to calculate (default='combined')
        metric (str): Specifies the unit for the resulting gaze angle (default='radians'):

    Returns:
        angle (torch.Tensor): A tensor of shape [batch_size] containing the estimated gaze angles for each
            instance in the batch, in the specified metric (radians or degrees).
    """

    # Landmark roi locations
    left_eye_roi = torch.tensor(
        [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173],
        dtype=int,
    )
    right_eye_roi = torch.tensor(
        [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
        dtype=int,
    )
    left_iris_roi = torch.tensor([468, 469, 470, 471, 472], dtype=int)
    right_iris_roi = torch.tensor([473, 474, 475, 476, 477], dtype=int)

    # Extract ROIs
    landmarks = convert_landmarks_3d(fex.landmarks)
    left_eye_landmarks = landmarks[:, left_eye_roi, :]
    right_eye_landmarks = landmarks[:, right_eye_roi, :]
    left_iris_landmarks = landmarks[:, left_iris_roi, :]
    right_iris_landmarks = landmarks[:, right_iris_roi, :]

    # Calculate the centers of the left and right eyes for the batch
    left_eye_center = torch.mean(left_eye_landmarks, dim=1)  # [batch_size, 3]
    right_eye_center = torch.mean(right_eye_landmarks, dim=1)  # [batch_size, 3]

    # Calculate the centers of the left and right irises for the batch
    left_iris_center = torch.mean(left_iris_landmarks, dim=1)  # [batch_size, 3]
    right_iris_center = torch.mean(right_iris_landmarks, dim=1)  # [batch_size, 3]

    # Calculate the gaze vectors for the left and right eyes
    left_gaze_vector = F.normalize(
        left_iris_center - left_eye_center, dim=1
    )  # [batch_size, 3]
    right_gaze_vector = F.normalize(
        right_iris_center - right_eye_center, dim=1
    )  # [batch_size, 3]

    if gaze_angle.lower() == "combined":
        combined_gaze_vector = F.normalize(
            (left_gaze_vector + right_gaze_vector) / 2, dim=1
        )  # [batch_size, 3]

        # Assuming the forward vector is along the camera's z-axis, repeated for the batch
        forward_vector = (
            torch.tensor([0, 0, 1], dtype=combined_gaze_vector.dtype)
            .unsqueeze(0)
            .repeat(combined_gaze_vector.size(0), 1)
        )  # [batch_size, 3]

        gaze_angles = torch.acos(
            torch.sum(combined_gaze_vector * forward_vector, dim=1)
            / (
                torch.norm(combined_gaze_vector, dim=1)
                * torch.norm(forward_vector, dim=1)
            )
        )
    elif gaze_angle.lower() == "left":
        # Assuming the forward vector is along the camera's z-axis, repeated for the batch
        forward_vector = (
            torch.tensor([0, 0, 1], dtype=left_gaze_vector.dtype)
            .unsqueeze(0)
            .repeat(left_gaze_vector.size(0), 1)
        )  # [batch_size, 3]

        gaze_angles = torch.acos(
            torch.sum(left_gaze_vector * forward_vector, dim=1)
            / (torch.norm(left_gaze_vector, dim=1) * torch.norm(forward_vector, dim=1))
        )
    elif gaze_angle.lower() == "right":
        # Assuming the forward vector is along the camera's z-axis, repeated for the batch
        forward_vector = (
            torch.tensor([0, 0, 1], dtype=right_gaze_vector.dtype)
            .unsqueeze(0)
            .repeat(right_gaze_vector.size(0), 1)
        )  # [batch_size, 3]

        gaze_angles = torch.acos(
            torch.sum(right_gaze_vector * forward_vector, dim=1)
            / (torch.norm(right_gaze_vector, dim=1) * torch.norm(forward_vector, dim=1))
        )
    else:
        raise NotImplementedError(
            "Only ['combined', 'left', 'right'] gaze_angle are currently implemented"
        )

    if metric.lower() == "radians":
        return gaze_angles
    elif metric.lower() == "degrees":
        return torch.rad2deg(gaze_angles)
    else:
        raise NotImplementedError("metric can only be ['radians', 'degrees']")


def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (pitch, roll, yaw).

    Parameters:
    -----------
    R : torch.Tensor
        A tensor of shape [batch_size, 3, 3] containing rotation matrices.

    Returns:
    --------
    euler_angles : torch.Tensor
        A tensor of shape [batch_size, 3] containing the Euler angles (pitch, roll, yaw) in radians.
    """
    sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)

    singular = sy < 1e-6

    pitch = torch.where(
        singular,
        torch.atan2(-R[:, 2, 1], R[:, 1, 1]),
        torch.atan2(R[:, 2, 1], R[:, 2, 2]),
    )
    roll = torch.atan2(-R[:, 2, 0], sy)
    yaw = torch.where(
        singular, torch.zeros_like(pitch), torch.atan2(R[:, 1, 0], R[:, 0, 0])
    )

    return torch.stack([pitch, roll, yaw], dim=1)


def estimate_face_pose(pts_3d, K, max_iter=100, lr=1e-3, return_euler_angles=True):
    """
    Estimate the face pose for a batch of 3D points using an iterative optimization approach.

    Args:
        pts_3d (torch.Tensor): A tensor of shape [batch_size, n_points, 3] representing the batch of 3D facial landmarks.
        K (torch.Tensor): A tensor of shape [batch_size, 3, 3] representing the camera intrinsic matrix for each image, or [3, 3] for a single shared intrinsic matrix.
        max_iter (int): The maximum number of iterations for the optimization loop. (default=100)
        lr (float): The learning rate for the Adam optimizer (default=1e-3)
        return_euler_angles (bool): If True, return 6 DOF (i.e., pitch, roll, and yaw angles) instead of the rotation matrix. (default=True)

    Returns:
        R_or_angles (torch.Tensor): If `return_euler_angles` is True, returns a tensor of shape [batch_size, 3] containing the Euler angles (pitch, roll, yaw). If `return_euler_angles` is False, returns a tensor of shape [batch_size, 3, 3] containing the rotation matrices.
        t (torch.Tensor): A tensor of shape [batch_size, 3] containing the estimated translation vectors.
    """

    # Ensure the dtype is consistent (e.g., float32)
    pts_3d = pts_3d.float()
    K = K.float()

    batch_size = pts_3d.size(0)

    # Check if K is a single matrix or a batch of matrices
    if K.dim() == 2:
        # If K is not batched, repeat it for each batch element
        K = K.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 3, 3]

    # Initial estimates for R and t (use identity and zeros for each batch)
    R = (
        torch.eye(3, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
        .requires_grad_(True)
    )  # [batch_size, 3, 3]
    t = torch.zeros(batch_size, 3, dtype=torch.float32).requires_grad_(
        True
    )  # [batch_size, 3]

    optimizer = Adam([R, t], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()

        # Rebuild the computation graph in every iteration
        pts_3d_proj = torch.bmm(pts_3d, R.transpose(1, 2)) + t.unsqueeze(
            1
        )  # [batch_size, n_points, 3]
        pts_2d_proj = torch.bmm(K, pts_3d_proj.transpose(1, 2)).transpose(
            1, 2
        )  # [batch_size, n_points, 3]

        # Normalize by the third coordinate
        pts_2d_proj = pts_2d_proj[:, :, :2] / pts_2d_proj[:, :, 2:].clamp(
            min=1e-7
        )  # [batch_size, n_points, 2]

        # Assuming directly facing camera means minimizing deviation from (x, y) plane
        loss = torch.mean(pts_3d_proj[:, :, 2] ** 2)  # Minimize z-coordinates to zero

        # Backpropagation
        loss.backward(retain_graph=True)
        optimizer.step()

        # Normalize R to keep it a valid rotation matrix (optional step)
        with torch.no_grad():  # Detach the graph here
            U, _, V = torch.svd(R)
            R.copy_(torch.bmm(U, V.transpose(1, 2)))  # Copy the values back to R in-place

    if return_euler_angles:
        # Convert rotation matrices to Euler angles (pitch, roll, yaw)
        euler_angles = rotation_matrix_to_euler_angles(R)
        return euler_angles, t
    else:
        return R, t


def plot_face_landmarks(
    fex,
    frame_idx,
    ax=None,
    oval_color="white",
    oval_linestyle="-",
    oval_linewidth=3,
    tesselation_color="gray",
    tesselation_linestyle="-",
    tesselation_linewidth=1,
    mouth_color="white",
    mouth_linestyle="-",
    mouth_linewidth=3,
    eye_color="navy",
    eye_linestyle="-",
    eye_linewidth=2,
    iris_color="skyblue",
    iris_linestyle="-",
    iris_linewidth=2,
):
    """Plots face landmarks on the given frame using specified styles for each part.

    Args:
        fex: DataFrame containing face landmarks (x, y coordinates).
        frame_idx: Index of the frame to plot.
        ax: Matplotlib axis to draw on. If None, a new axis is created.
        oval_color, tesselation_color, mouth_color, eye_color, iris_color: Colors for each face part.
        oval_linestyle, tesselation_linestyle, mouth_linestyle, eye_linestyle, iris_linestyle: Linestyle for each face part.
        oval_linewidth, tesselation_linewidth, mouth_linewidth, eye_linewidth, iris_linewidth: Linewidth for each face part.
        n_faces: Number of faces in the frame. If None, will be determined from fex.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Get frame data
    fex_frame = fex.query("frame == @frame_idx")
    n_faces_frame = fex_frame.shape[0]

    # Add the frame image
    ax.imshow(Image.open(fex_frame["input"].unique()[0]))

    # Helper function to draw lines for a set of connections
    def draw_connections(face_idx, connections, color, linestyle, linewidth):
        for connection in connections:
            start = connection.start
            end = connection.end
            line = plt.Line2D(
                [fex.loc[face_idx, f"x_{start}"], fex.loc[face_idx, f"x_{end}"]],
                [fex.loc[face_idx, f"y_{start}"], fex.loc[face_idx, f"y_{end}"]],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
            ax.add_line(line)

    # Face tessellation
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            tesselation_color,
            tesselation_linestyle,
            tesselation_linewidth,
        )

    # Mouth
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
            mouth_color,
            mouth_linestyle,
            mouth_linewidth,
        )

    # Left iris
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            iris_color,
            iris_linestyle,
            iris_linewidth,
        )

    # Left eye
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
            eye_color,
            eye_linestyle,
            eye_linewidth,
        )

    # Left eyebrow
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
            eye_color,
            eye_linestyle,
            eye_linewidth,
        )

    # Right iris
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            iris_color,
            iris_linestyle,
            iris_linewidth,
        )

    # Right eye
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
            eye_color,
            eye_linestyle,
            eye_linewidth,
        )

    # Right eyebrow
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
            eye_color,
            eye_linestyle,
            eye_linewidth,
        )

    # Face oval
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
            oval_color,
            oval_linestyle,
            oval_linewidth,
        )

    # Optionally turn off axis for a clean plot
    ax.axis("off")

    return ax


class MPDetector(nn.Module):
    def __init__(
        self,
        face_model="retinaface",
        landmark_model="mp_facemesh_v2",
        au_model="mp_blendshapes",
        device="cpu",
    ):
        super(MPDetector, self).__init__()

        self.info = dict(
            face_model=face_model,
            landmark_model=landmark_model,
            au_model=au_model,
        )
        self.device = set_torch_device(device)

        # Initialize Face Detector
        if face_model == "retinaface":


            
            face_config_file = "./ckpts/config.json"
            # face_config_file = hf_hub_download(
            #     repo_id="py-feat/retinaface",
            #     filename="config.json",
            #     cache_dir=get_resource_path(),
            # )
            with open(face_config_file, "r") as f:
                self.face_config = json.load(f)

            face_model_file = './ckpts/mobilenet0.25_Final.pth'
            # face_model_file = hf_hub_download(
            #     repo_id="py-feat/retinaface",
            #     filename="mobilenet0.25_Final.pth",
            #     cache_dir=get_resource_path(),
            # )
            face_checkpoint = torch.load(
                face_model_file, map_location=self.device, weights_only=False
            )

            self.face_detector = RetinaFace(cfg=self.face_config, phase="test")
            self.face_detector.load_state_dict(face_checkpoint)
            self.face_detector.eval()
            self.face_detector.to(self.device)
        else:
            raise ValueError(f"{face_model} is not currently supported.")

        # Initialize Landmark Detector
        if landmark_model == "mp_facemesh_v2":
            self.face_size = 256
            landmark_model_file = './ckpts/face_landmarks_detector_Nx3x256x256_onnx.pth'
            # landmark_model_file = hf_hub_download(
            #     repo_id="py-feat/mp_facemesh_v2",
            #     filename="face_landmarks_detector_Nx3x256x256_onnx.pth",
            #     cache_dir=get_resource_path(),
            # )
            self.landmark_detector = torch.load(
                landmark_model_file, map_location=self.device, weights_only=False
            )
            self.landmark_detector.eval()
            self.landmark_detector.to(self.device)
        else:
            raise ValueError(f"{landmark_model} is not currently supported.")

        # Initialize AU Detector
        if au_model == "mp_blendshapes":
            self.au_detector = MediaPipeBlendshapesMLPMixer()
            au_model_path ='ckpts/face_blendshapes.pth'
            # au_model_path = hf_hub_download(
            #     repo_id="py-feat/mp_blendshapes",
            #     filename="face_blendshapes.pth",
            #     cache_dir=get_resource_path(),
            # )
            au_checkpoint = torch.load(
                au_model_path, map_location=device, weights_only=True
            )
            self.au_detector.load_state_dict(au_checkpoint)
            self.au_detector.to(self.device)
        else:
            raise ValueError(f"{au_model} is not currently supported.")

    @torch.inference_mode()
    def detect_faces(self, images, face_size=256, face_detection_threshold=0.5):
        import time
        
        frames = convert_image_to_tensor(images, img_type="float32")
        batch_results = []
        

        for i in range(frames.size(0)):
            frame = frames[i, ...].unsqueeze(0)
            single_frame = torch.sub(
                frame, convert_color_vector_to_tensor(np.array([123, 117, 104])).to(frame.device)
            )
            
            predicted_locations, predicted_scores, predicted_landmarks = (
                self.face_detector.forward(single_frame.to(self.device))
            )

            face_output = postprocess_retinaface(
                predicted_locations,
                predicted_scores,
                predicted_landmarks,
                self.face_config,
                single_frame,
                device=self.device,
            )

            
            bbox = face_output["boxes"]
            facescores = face_output["scores"]

            if bbox.numel() != 0:
                extracted_faces, new_bbox = extract_face_from_bbox_torch(
                    frame / 255.0, bbox, face_size=face_size, expand_bbox=1.25
                )
            else:
                # extracted_faces = torch.zeros((1, 3, face_size, face_size))
                # bbox = torch.zeros((1, 4))
                # new_bbox = torch.zeros((1, 4))
                # facescores = torch.zeros((1))
                # Add device=frame.device to all torch.zeros calls
                extracted_faces = torch.zeros((1, 3, face_size, face_size), device=frame.device)
                bbox = torch.zeros((1, 4), device=frame.device)
                new_bbox = torch.zeros((1, 4), device=frame.device)
                facescores = torch.zeros((1), device=frame.device)

            frame_results = {
                "face_id": i,
                "faces": extracted_faces,
                "boxes": bbox,
                "new_boxes": new_bbox,
                "scores": facescores,
            }
            batch_results.append(frame_results)

        return batch_results

    @torch.inference_mode()
    def forward(self, faces_data):
        extracted_faces = torch.cat([face["faces"] for face in faces_data], dim=0)
        new_bboxes = torch.cat([face["new_boxes"] for face in faces_data], dim=0)
        n_faces = extracted_faces.shape[0]
        # Get landmarks
        landmarks = self.landmark_detector.forward(extracted_faces.to(self.device))[0]
        landmarks_3d = landmarks.reshape(n_faces, 478, 3)
        
        # Project landmarks back to original image
        img_size = (
            torch.tensor((1 / self.face_size, 1 / self.face_size))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        landmarks_2d = landmarks_3d[:, :, :2] * img_size
        rescaled_landmarks_2d = inverse_transform_landmarks_torch(
            landmarks_2d.reshape(n_faces, 478 * 2), new_bboxes.to(self.device)
        )
        new_landmarks = torch.cat(
            (
                rescaled_landmarks_2d.reshape(n_faces, 478, 2),
                landmarks_3d[:, :, 2].unsqueeze(2),
            ),
            dim=2,
        )

        # Get blendshapes
        blendshapes = self.au_detector(
            landmarks.reshape(n_faces, 478, 3)[
                :, MP_BLENDSHAPE_MODEL_LANDMARKS_SUBSET, :2
            ].to(self.device)
        ).squeeze(2).squeeze(2)

        # Get face boxes
        boxes = torch.cat(
            [face["new_boxes"] for face in faces_data],
            dim=0,
        )

        return boxes, new_landmarks, blendshapes

    def detect(
        self,
        inputs,
        data_type="image",
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        face_detection_threshold=0.5,
        progress_bar=True,
        **kwargs,
    ):
        import time
        data_loader = DataLoader(
            TensorDataset(inputs),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        all_boxes = []
        all_landmarks = []
        all_blendshapes = []

        # 初始化计时器
        # face_detection_total_time = 0
        # feature_extraction_total_time = 0

        # 处理每个批次
        for batch_data in data_loader:
            # 人脸检测计时
            # face_detection_start = time.time()
            faces_data = self.detect_faces(
                batch_data["Image"],
                face_size=self.face_size,
                face_detection_threshold=face_detection_threshold,
            )
            # face_detection_total_time += time.time() - face_detection_start
            
            # 特征提取计时
            # feature_extraction_start = time.time()
            boxes, landmarks, blendshapes = self.forward(faces_data)
            # feature_extraction_total_time += time.time() - feature_extraction_start
            
            all_boxes.append(boxes[0].unsqueeze(0))
            all_landmarks.append(landmarks[0].unsqueeze(0))
            all_blendshapes.append(blendshapes[0].unsqueeze(0))

        # # 打印总耗时
        # print(f"人脸检测总耗时: {face_detection_total_time:.4f}秒")
        # print(f"特征提取总耗时: {feature_extraction_total_time:.4f}秒")
        # print(f"总处理时间: {face_detection_total_time + feature_extraction_total_time:.4f}秒")
        # breakpoint()
        result = (
            torch.cat(all_boxes, dim=0),
            torch.cat(all_landmarks, dim=0),
            torch.cat(all_blendshapes, dim=0)
        )
        
        return result
