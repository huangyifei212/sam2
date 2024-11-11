# python train_unet.py --data_dir /mnt/data/huangyifei/segment-anything-2/videos/Task09_Spleen_256 --num_epochs 50 --batch_size 4 --learning_rate 1e-4 --device_id 4
# python train_unet.py --data_dir /mnt/data/huangyifei/segment-anything-2/videos/Task03_Liver_256 --num_epochs 50 --batch_size 4 --learning_rate 1e-4 --device_id 4
# python train_unet.py --data_dir /mnt/data/huangyifei/segment-anything-2/videos/Task06_Lung_256 --num_epochs 50 --batch_size 4 --learning_rate 1e-4 --device_id 4
# python train_unet.py --data_dir /mnt/data/huangyifei/segment-anything-2/videos/Task07_Pancreas_256 --num_epochs 50 --batch_size 4 --learning_rate 1e-4 --device_id 4


# python test.py --data_dir /mnt/data/huangyifei/segment-anything-2/videos/Task09_Spleen_256 --num_epochs 50 --batch_size 4 --learning_rate 1e-4 --device_id 4
python test.py --data_dir /mnt/data/huangyifei/segment-anything-2/videos/Task03_Liver_256 --num_epochs 50 --batch_size 4 --learning_rate 1e-4 --device_id 4
python test.py --data_dir /mnt/data/huangyifei/segment-anything-2/videos/Task06_Lung_256 --num_epochs 50 --batch_size 4 --learning_rate 1e-4 --device_id 4
python test.py --data_dir /mnt/data/huangyifei/segment-anything-2/videos/Task07_Pancreas_256 --num_epochs 50 --batch_size 4 --learning_rate 1e-4 --device_id 4