import os
import torch
import numpy as np
import argparse
import glob
from monai.metrics import DiceHelper, compute_surface_dice
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete, LoadImage, ScaleIntensity

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 Video Predictor with Untrained U-Net Mask Selection")
    # parser.add_argument('--base_video_dir', type=str, help="Base directory containing all video directories", required=True)
    # parser.add_argument('--prefix', type=str, help="Prefix for video directories", default="BraTS20_Training_")
    parser.add_argument('--base_video_dir', type=str,  help="Base directory containing all video directories", default="/mnt/data/huangyifei/segment-anything-2/videos/Task09_Spleen_256")
    parser.add_argument('--prefix', type=str, help="Prefix for video directories", default="spleen_")
    return parser.parse_args()

# 初始化未训练的 U-Net 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
model.eval()  # 设置为推理模式

# 用 U-Net 模型预测 mask
def predict_mask(image):
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_mask = model(image)
    return AsDiscrete(threshold=0.5)(pred_mask).cpu().numpy().squeeze()

# 图像预处理函数
def load_and_preprocess_image(image_path):
    loader = LoadImage(image_only=True)
    image = loader(image_path)
    transform = ScaleIntensity()
    return transform(image)

# 主代码
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from sam2.build_sam import build_sam2_video_predictor

    # 初始化模型预测器
    sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    # Parse arguments
    args = parse_args()
    base_video_dir = args.base_video_dir
    prefix = args.prefix

    # Collect all video directories
    search_pattern = os.path.join(base_video_dir, f"{prefix}*")
    video_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]

    # Initialize metrics lists
    dice_scores = []
    nsd_scores = []

    # Loop over each video directory
    for video_dir in video_dirs:
        ann_obj_id = 1
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        print(f"Processing {video_dir}, number of frames: {len(frame_names)}")

        # Skip if no frames
        if len(frame_names) == 0:
            print(f"No frames found in {video_dir}, skipping...")
            continue

        frame_idx = len(frame_names) // 2  # 中间帧索引
        gt_mask = np.load(os.path.join(video_dir, frame_names[frame_idx].replace(".jpg", ".npy")))

        # Skip frames with insufficient foreground pixels
        if np.sum(gt_mask > 0) <= 256:
            print(f"Frame {frame_idx} skipped due to insufficient foreground points.")
            continue

        # Step 1: 使用 U-Net 预测中间帧的初始 mask
        image_path = os.path.join(video_dir, frame_names[frame_idx])
        input_image = load_and_preprocess_image(image_path)
        out_mask_logits = predict_mask(input_image)  # 使用原图生成预测的初始掩码

        # Step 2: 初始化 `inference_state` 并设置初始掩码
        inference_state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(inference_state)
        predictor.add_new_mask(inference_state=inference_state, frame_idx=frame_idx, obj_id=ann_obj_id, mask=out_mask_logits > 0.0)

        # Step 3: 进行正反方向的帧传播推理，存储分割结果
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=frame_idx-1, reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Step 4: 组装分割结果为 `video_seg_3d` 和 `gt_3d`
        video_seg_3d = np.stack([video_segments[k][1] for k in video_segments])
        gt_3d = np.stack([np.load(os.path.join(video_dir, frame_names[k].replace(".jpg", ".npy")))[None] for k in video_segments])
        print(video_seg_3d.shape, gt_3d.shape)

        # Step 5: 计算并评估 Dice 和 NSD 指标
        n_classes, batch_size = 1, 1
        spatial_shape = (video_seg_3d.shape[0], video_seg_3d.shape[2], video_seg_3d.shape[3])
        y_pred = torch.tensor(video_seg_3d).float().reshape(batch_size, n_classes, *spatial_shape).to(device)
        y = torch.tensor(gt_3d).float().reshape(batch_size, n_classes, *spatial_shape).to(device)

        score, not_nans = DiceHelper(include_background=False, sigmoid=True, softmax=True)(y_pred, y)
        dice = score.item()
        nsd = compute_surface_dice(y_pred, y, class_thresholds=[1]).item()
        dice_scores.append(dice)
        nsd_scores.append(nsd)
        print(f"Dice: {dice:.4f}, nsd: {nsd:.4f}")

        # 打印当前平均分数
        current_average_dice = np.mean(dice_scores)
        current_average_nsd = np.mean(nsd_scores)
        print(f"Current Average Dice: {current_average_dice:.4f}, Current Average nsd: {current_average_nsd:.4f}")

    # 最终平均分数
    final_average_dice = np.mean(dice_scores)
    final_average_nsd = np.mean(nsd_scores)
    print(f"Final Average Dice: {final_average_dice:.4f}, Final Average nsd: {final_average_nsd:.4f}")
