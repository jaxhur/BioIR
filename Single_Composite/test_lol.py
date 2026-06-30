import argparse
import csv
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from tqdm import tqdm

from basicsr.models import create_model
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.options import parse


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Image folder does not exist: {folder}')
    return sorted([p for p in folder.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS])


def build_gt_index(gt_dir):
    gt_paths = list_images(gt_dir)
    by_relative = {p.relative_to(gt_dir).as_posix(): p for p in gt_paths}
    by_name = {}
    duplicated = set()
    for path in gt_paths:
        key = path.name
        if key in by_name:
            duplicated.add(key)
        by_name[key] = path
    return by_relative, by_name, duplicated


def make_pairs(lq_dir, gt_dir):
    lq_dir = Path(lq_dir)
    gt_dir = Path(gt_dir)
    lq_paths = list_images(lq_dir)
    gt_by_relative, gt_by_name, duplicated_names = build_gt_index(gt_dir)

    pairs = []
    missing = []
    for lq_path in lq_paths:
        relative_key = lq_path.relative_to(lq_dir).as_posix()
        gt_path = gt_by_relative.get(relative_key)
        if gt_path is None and lq_path.name not in duplicated_names:
            gt_path = gt_by_name.get(lq_path.name)
        if gt_path is None:
            missing.append(str(lq_path))
            continue
        pairs.append((lq_path, gt_path))

    if missing:
        preview = '\n'.join(missing[:10])
        raise FileNotFoundError(
            f'Could not find GT pairs for {len(missing)} low-light images. '
            f'First missing files:\n{preview}')
    if not pairs:
        raise RuntimeError(f'No image pairs found in {lq_dir} and {gt_dir}')
    return pairs


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def center_label(img, text, height=28):
    label = np.zeros((height, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(label, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([label, img], axis=0)


def save_comparison(path, low, restored, gt):
    if low.shape != gt.shape:
        gt = cv2.resize(gt, (low.shape[1], low.shape[0]), interpolation=cv2.INTER_AREA)
    if restored.shape != low.shape:
        restored = cv2.resize(restored, (low.shape[1], low.shape[0]), interpolation=cv2.INTER_AREA)
    canvas = np.concatenate([
        center_label(low, 'Low'),
        center_label(restored, 'BioIR'),
        center_label(gt, 'GT')
    ], axis=1)
    save_rgb(path, canvas)


def crop_border(img, border):
    if border == 0:
        return img
    return img[border:-border, border:-border, ...]


def to_metric_image(img, test_y_channel):
    if not test_y_channel:
        return img
    return rgb2ycbcr(img, y_only=True)


def calculate_psnr(restored, gt, border=0, test_y_channel=False):
    restored = crop_border(restored, border)
    gt = crop_border(gt, border)
    restored = to_metric_image(restored, test_y_channel).astype(np.float64)
    gt = to_metric_image(gt, test_y_channel).astype(np.float64)
    mse = np.mean((restored - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(restored, gt, border=0, test_y_channel=False):
    restored = crop_border(restored, border)
    gt = crop_border(gt, border)
    restored = to_metric_image(restored, test_y_channel)
    gt = to_metric_image(gt, test_y_channel)
    if restored.ndim == 2:
        return structural_similarity(restored, gt, data_range=255)
    try:
        return structural_similarity(restored, gt, channel_axis=2, data_range=255)
    except TypeError:
        return structural_similarity(restored, gt, multichannel=True, data_range=255)


def load_state_dict(weights_path):
    checkpoint = torch.load(weights_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ('params_ema', 'params', 'state_dict'):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith('module.') else key] = value
    return cleaned


def load_model(opt_path, weights_path, device):
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    opt['num_gpu'] = 0 if device.type == 'cpu' else max(1, opt.get('num_gpu', 1))

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    model = create_model(opt).net_g
    state_dict = strip_module_prefix(load_state_dict(weights_path))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, opt


def infer_one(model, img_rgb, device, factor):
    img = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, h, w = tensor.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    with torch.inference_mode():
        restored = model(tensor)
        if isinstance(restored, list):
            restored = restored[-1]
    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1).squeeze(0).permute(1, 2, 0)
    restored = restored.detach().cpu().numpy()
    return np.round(restored * 255.0).astype(np.uint8)


def write_metrics_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Test BioIR on LOL datasets.')
    parser.add_argument('--opt', required=True, help='Path to a LOL option YAML file.')
    parser.add_argument('--weights', required=True, help='Path to trained model weights.')
    parser.add_argument('--output_dir', default='./results_lol', help='Directory to save outputs.')
    parser.add_argument('--name', default=None, help='Result subfolder name. Defaults to option name.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--factor', type=int, default=32, help='Pad images to multiples of this value.')
    parser.add_argument('--crop_border', type=int, default=0)
    parser.add_argument('--test_y_channel', action='store_true')
    parser.add_argument('--save_comparison', action='store_true',
                        help='Also save low/restored/GT comparison images.')
    parser.add_argument('--no_comparison', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but torch.cuda.is_available() is False.')

    model, opt = load_model(args.opt, args.weights, device)
    dataset_name = args.name or opt['name']
    lq_dir = Path(opt['datasets']['val']['dataroot_lq'])
    gt_dir = Path(opt['datasets']['val']['dataroot_gt'])
    pairs = make_pairs(lq_dir, gt_dir)

    output_root = Path(args.output_dir) / dataset_name
    restored_dir = output_root / 'restored'
    comparison_dir = output_root / 'comparison'
    save_comparison_images = args.save_comparison and not args.no_comparison

    rows = []
    psnr_values = []
    ssim_values = []

    print(f'Testing {dataset_name} on {len(pairs)} image pairs with {device}.')
    for lq_path, gt_path in tqdm(pairs, unit='image'):
        low = load_rgb(lq_path)
        gt = load_rgb(gt_path)
        restored = infer_one(model, low, device, args.factor)

        relative = lq_path.relative_to(lq_dir)
        save_rgb(restored_dir / relative.with_suffix('.png'), restored)
        if save_comparison_images:
            save_comparison(comparison_dir / relative.with_suffix('.png'), low, restored, gt)

        if restored.shape != gt.shape:
            raise ValueError(
                f'Restored and GT shapes differ for {lq_path.name}: '
                f'{restored.shape} vs {gt.shape}')

        psnr = calculate_psnr(restored, gt, args.crop_border, args.test_y_channel)
        ssim = calculate_ssim(restored, gt, args.crop_border, args.test_y_channel)
        rows.append({'image': relative.as_posix(), 'psnr': f'{psnr:.6f}', 'ssim': f'{ssim:.6f}'})
        psnr_values.append(psnr)
        ssim_values.append(ssim)

    avg_psnr = float(np.mean(psnr_values))
    avg_ssim = float(np.mean(ssim_values))
    rows.append({'image': 'Average', 'psnr': f'{avg_psnr:.6f}', 'ssim': f'{avg_ssim:.6f}'})
    write_metrics_csv(output_root / 'metrics.csv', rows)

    print(f'Average PSNR: {avg_psnr:.6f} dB')
    print(f'Average SSIM: {avg_ssim:.6f}')
    print(f'Restored images: {restored_dir}')
    if save_comparison_images:
        print(f'Comparison images: {comparison_dir}')
    print(f'Metrics CSV: {output_root / "metrics.csv"}')


if __name__ == '__main__':
    main()
