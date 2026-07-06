import argparse
import torch
import os
from pathlib import Path
from PIL import Image
import json
from torchvision import transforms
import model.model as mental_health_model


def infer_config_png(config_path, image_path, checkpoint_path=None, device='cuda'):
    with open(config_path, 'r') as f:
        config = json.load(f)

    arch_config = config['arch']
    model_class = getattr(mental_health_model, arch_config['type'])
    model = model_class(**arch_config.get('args', {}))
    model = model.to(device)

    if checkpoint_path is None:
        model_dir = Path('audiotrain/'+config['trainer']['save_dir']) / 'models' / config['name']
        for run_dir in sorted(model_dir.iterdir(), key=lambda x: x.name):
            ckpt = run_dir / 'checkpoint-epoch10.pth'
            if ckpt.exists():
                checkpoint_path = ckpt
                break
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, using randomly initialized weights")

    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        pred_class = torch.argmax(output, dim=1).item()
        pred_prob = probabilities[0, pred_class].item()
        all_probs = probabilities[0].cpu().numpy()

    num_classes = config['arch']['args'].get('num_classes', 2)
    class_names = [f"Class_{i}" for i in range(num_classes)]

    print(f"\nInference Results for: {image_path}")
    print(f"Predicted Class: {pred_class}")
    print(f"Confidence: {pred_prob:.4f}")
    print(f"All Class Probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, all_probs)):
        print(f"  {name}: {prob:.4f}")

    return pred_class, pred_prob, all_probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument('-c', '--config', required=True, type=str,
                        help='path to config json file')
    parser.add_argument('-i', '--image', required=True, type=str,
                        help='path to input image (png)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to checkpoint file (default: auto-detect)')
    parser.add_argument('-d', '--device', default='cuda', type=str,
                        help='device to use (default: cuda)')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    infer_config_png(args.config, args.image, args.resume, args.device)