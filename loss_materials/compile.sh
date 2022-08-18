# cd /home/fengwen/yolov5
# python3 train.py --data  /home/fengwen/yolov5/data/coco128.yaml --cfg /home/fengwen/yolov5/models/yolov5s.yaml   --weights yolov5s.pt --device 0

# python train.py  --batch 64 --data coco.yaml --weights yolov5s.pt --device 0

# watch -n 1 nvidia-smi

#  python3 train.py --data ./dataset/coco128.yaml --cfg ./models/yolov5s.yaml --weights '' --batch-size 32


# python3 train.py --data  /home/fengwen/yolov5/data/coco128.yaml --cfg /home/fengwen/yolov5/models/yolov5s.yaml   --weights yolov5s.pt --device 0 --batch-size 16


# tourch:  python3 train.py --data  /home/fengwen/yolov5/data/coco128.yaml --cfg /home/fengwen/yolov5/models/yolov5s.yaml   --weights '/home/fengwen/yolov5/yolov5s.pt' --device 0 --batch-size 16 --epochs 20 --device 0 

# oneflow: python3 train.py --data ./dataset/coco128.yaml --cfg ./models/yolov5s.yaml --weights '/home/fengwen/one-yolo/yolov5s' --batch-size 16  --epochs 2 

# nvidia-smi -q -d MEMORY|tee -a oneflowgpu.log

# watch -n  3 'nvidia-smi -q -d MEMORY|tee -a gpu.log'

# nvidia-smi -q -d MEMORY|tee -a touchgpu.log


# torch 版本 ：48a85314bc80d8023c99bfb114cea98d71dd0591


# #pytourch: 
# python3  -m  torch.distributed.run --nproc_per_node    8   /home/fengwen/torch-yolo5/train.py --data /home/fengwen/torch-yolo5/data/coco128.yaml  --cfg /home/fengwen/torch-yolo5/models/yolov5s.yaml --weights /home/fengwen/weights/yolov5s.pt  --batch-size 16 

# # oneflow:
# python3  -m  oneflow.distributed.launch --nproc_per_node 8 /home/fengwen/one-yolo/train.py  --data /home/fengwen/torch-yolo5/data/coco128.yaml   --cfg   /home/fengwen/torch-yolo5/models/yolov5s.yaml --weights /home/fengwen/weights/yolov5s   --batch-size 16 --epochs 50



# python3 /home/fengwen/one-yolo/train.py  --data /home/fengwen/coco_yaml/coco128.yaml    --weights /home/fengwen/weights/one-yolov5s   --batch-size 16 --epochs 100 --device 0 --cfg /home/fengwen/torch-yolo5/models/yolov5s.yaml

# wait

# python3  /home/fengwen/torch-yolo5/train.py --data /home/fengwen/coco_yaml/coco128.yaml --weights /home/fengwen/weights/yolov5s.pt   --batch-size 16  --epochs 20 --device 0  --cfg /home/fengwen/torch-yolo5/models/yolov5s.yaml



#  python3 /home/fengwen/one-yolo/train.py --data /home/fengwen/coco_yaml/coco.yaml  --batch-size 16 --cfg /home/fengwen/torch-yolo5/models/yolov5s.yaml --epochs 300 --project /dataset/fengwen_data/one-yolo --noplots




#  python3 -m pip install --pre oneflow -f  https://staging.oneflow.info/branch/dev_sync_batchnorm_merge_fix_batchnorm_cudnn_mode/cu112



#  python3  -m  oneflow.distributed.launch --nproc_per_node 8   /home/fengwen/one-yolo/train.py --data /home/fengwen/coco_yaml/coco.yaml  --batch-size 16 --cfg /home/fengwen/torch-yolo5/models/yolov5s.yaml --epochs 300 --project /dataset/fengwen_data/one-yolo --noplots




#  dev_sync_batchnorm_merge_fix_batchnorm_cudnn_mode 

#  python3 -m pip install --pre oneflow -f  https://staging.oneflow.info/branch/dev_sync_batchnorm_merge_fix_batchnorm_cudnn_mode/cu112

#  python3 -m pip install --pre oneflow -f  https://staging.oneflow.info/branch/dev_fix_batchnorm_cudnn_mode/cu112 


#  python3  -m  oneflow.distributed.launch --nproc_per_node 2 /home/fengwen/one-yolo/train.py --data /home/fengwen/coco_yaml/coco128.yaml  --batch-size 16 --cfg /home/fengwen/torch-yolo5/models/yolov5s.yaml --epochs 300 --project /dataset/fengwen_data/one-yolo --noplots











 python3  -m  oneflow.distributed.launch --nproc_per_node 8 /home/fengwen/one-yolo/train.py --data /home/fengwen/coco_yaml/coco.yaml --batch-size 16 --cfg /home/fengwen/torch-yolo5/models/yolov5s.yaml --epochs 300 --project /dataset/fengwen_data/one-yolo --noplots noval nosave --multi-scale