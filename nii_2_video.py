import os
from os.path import expanduser
import nibabel as nib
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt

# 将标签切片数据保存为 .npy 格式文件
def save_label_slice_as_npy(label_slice, output_file_path):
    np.save(output_file_path, label_slice)

# 将标签切片显示为灰度图像，并添加颜色条和标题，然后保存为图片文件
def save_label_slice(label_slice, output_file_path):
    plt.imshow(label_slice, cmap='gray')
    plt.colorbar()
    plt.title('Label Slice')
    plt.savefig(output_file_path)
    plt.close()

#  处理单个图像切片
# 将标签切片 label_slice 转换为二值化数组，非零值变为1。使用 save_label_slice_as_npy 保存二值标签切片。
# 将图像切片 img_slice 归一化到 0-255 范围，转换为 8 位无符号整数，使用 PIL.Image.fromarray 保存为图片文件。
def process_slice(img_slice, label_slice, subfolder_path, file_counter):
    label_slice_binary = np.where(label_slice != 0, 1, 0)
    label_output_file_path = os.path.join(subfolder_path, f"{file_counter:05d}.npy")
    save_label_slice_as_npy(label_slice_binary, label_output_file_path)
    
    img_slice_normalized = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
    img_slice_8bit = (img_slice_normalized * 255).astype(np.uint8)
    output_file_path = os.path.join(subfolder_path, f"{file_counter:05d}.jpg")
    Image.fromarray(img_slice_8bit).save(output_file_path)
    print(f"Saved {output_file_path}")

# 处理多个图像切片
# 遍历 image_folder_path 下的所有.nii.gz 文件，找到对应的 label 文件，并将切片保存到 output_folder_path 下的子文件夹中。
# 对于每个切片，使用 process_slice 函数处理。
# 对于每个切片，如果 threshold 为 0，则随机选择一个范围内的切片；否则，选择 foreground_counts 超过 threshold 的切片。
# 对于每个切片，使用 save_label_slice 函数保存二值标签切片，使用 PIL.Image.fromarray 保存图像切片。
def load_nii_gz_files(image_folder_path, label_folder_path, output_folder_path):
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.nii.gz') and not f.startswith('._')]
    os.makedirs(output_folder_path, exist_ok=True)
    
    for image_file in image_files:
        image_file_path = os.path.join(image_folder_path, image_file)
        label_file_path = os.path.join(label_folder_path, image_file)
        subfolder_name = os.path.splitext(os.path.splitext(image_file)[0])[0]
        subfolder_path = os.path.join(output_folder_path, subfolder_name)
        
        if not os.path.exists(label_file_path):
            print(f"Label file for {image_file} not found in {label_folder_path}")
            continue
        
        # 创建子文件夹
        os.makedirs(subfolder_path, exist_ok=True)
        img = nib.load(image_file_path)
        label = nib.load(label_file_path)
        
        img_data = img.get_fdata()
        label_data = label.get_fdata()
        
        threshold = args.threshold
        foreground_counts = np.sum(label_data > 0, axis=(0, 1))
        positive_indices = np.where(foreground_counts > threshold)[0]
        try:
            slices_with_positive_values = np.unique(positive_indices)
        except IndexError:
            continue
        if len(slices_with_positive_values) == 0:
            continue
        
        if threshold != 0:
            min_slice = max(0, np.min(slices_with_positive_values))
            max_slice = min(label_data.shape[2], np.max(slices_with_positive_values))
        else:
            min_slice = max(0, np.min(slices_with_positive_values) - np.random.randint(5, 10))
            max_slice = min(label_data.shape[2], np.max(slices_with_positive_values) + np.random.randint(5, 10))
        for i in range(min_slice, max_slice):
            process_slice(img_data[:, :, i], label_data[:, :, i], subfolder_path, i - min_slice)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process some paths.')
    # parser.add_argument('--image_folder', type=str, default='~/msd/Task07_Pancreas/imagesTr', 
    #                     help='Path to the image folder')
    # parser.add_argument('--label_folder', type=str, default='~/msd/Task07_Pancreas/labelsTr', 
    #                     help='Path to the label folder')
    # parser.add_argument('--output_folder', type=str, default='./videos/Task07_Pancreas',
    #                     help='Path to the output folder')
    # parser.add_argument('--threshold', type=int, default=256,
    #                     help='Threshold for the number of positive pixels in a slice')
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--image_folder', type=str, default='/mnt/data/huangyifei/segment-anything-2/new_msd/Task03_Liver/imagesTr', 
                        help='Path to the image folder')
    parser.add_argument('--label_folder', type=str, default='/mnt/data/huangyifei/segment-anything-2/new_msd/Task03_Liver/labelsTr', 
                        help='Path to the label folder')
    parser.add_argument('--output_folder', type=str, default='/mnt/data/huangyifei/segment-anything-2/videos/Task03_Liver',
                        help='Path to the output folder')
    parser.add_argument('--threshold', type=int, default=256,
                        help='Threshold for the number of positive pixels in a slice')

    args = parser.parse_args()
    load_nii_gz_files(expanduser(args.image_folder), expanduser(args.label_folder), expanduser(args.output_folder)+f"_{args.threshold}")
