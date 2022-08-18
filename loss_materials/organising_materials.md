 ### 版本号
 ```
 >>> import oneflow
loaded library: /lib/libibverbs.so.1
>>> import  torch
>>> oneflow.__version__
'0.8.1.dev20220731+cu112'
>>> torch.__version__
'1.11.0+cu102'
>>> 

 ```
 ###  指令 
 ```
oneflow：   python3 train.py --data ./dataset/coco128.yaml --cfg ./models/yolov5s.yaml --weights './yolov5s' --batch-size 4  --epochs 50  --device 0 
pytorch：   python3 train.py --data  /home/fengwen/yolov5/data/coco128.yaml --cfg /home/fengwen/yolov5/models/yolov5s.yaml   --weights '/home/fengwen/yolov5/yolov5s.pt'  --batch-size 16 --epochs 50  --device  0

oneflow观察显存并写入log文件： watch -n  2 'nvidia-smi -q -d MEMORY|tee -a /home/fengwen/loss_materials/nvidia_smi_log/oneflowgpu.log'
pytorch观察显存并写入log文件： watch -n  2 'nvidia-smi -q -d MEMORY|tee -a /home/fengwen/loss_materials/nvidia_smi_log/pytorchgpu.log'

```

# 跑 batch-size 32
oneflow: python3 train.py --data  /home/fengwen/yolov5/data/coco128.yaml --cfg /home/fengwen/yolov5/models/yolov5s.yaml --weights './yolov5s' --batch-size 32  --epochs 4  --device 0 
oneflow:  watch -n  1 'nvidia-smi -q -d MEMORY|tee -a /home/fengwen/loss_materials/nvidia_smi_log/batch_size_32/oneflowgpu.log'

pytorch: python3 train.py --data  /home/fengwen/yolov5/data/coco128.yaml --cfg /home/fengwen/yolov5/models/yolov5s.yaml   --weights '/home/fengwen/yolov5/yolov5s.pt'  --batch-size 32 --epochs 3  --device  0
pytorch: watch -n  1 'nvidia-smi -q -d MEMORY|tee -a /home/fengwen/loss_materials/nvidia_smi_log/batch_size_32/pytorchgpu.log'


#  跑batch-size 16

oneflow: python3 train.py --data /home/fengwen/yolov5/data/coco128.yaml --cfg /home/fengwen/yolov5/models/yolov5s.yaml --weights './yolov5s' --batch-size 16  --epochs 4  --device 0 
oneflow:  watch -n  1 'nvidia-smi -q -d MEMORY|tee -a /home/fengwen/loss_materials/nvidia_smi_log/batch_size_8/oneflowgpu.log'

pytorch: python3 train.py --data  /home/fengwen/yolov5/data/coco128.yaml --cfg /home/fengwen/yolov5/models/yolov5s.yaml   --weights '/home/fengwen/yolov5/yolov5s.pt'  --batch-size 8 --epochs 4  --device  0
pytorch: watch -n  1 'nvidia-smi -q -d MEMORY|tee -a /home/fengwen/loss_materials/nvidia_smi_log/batch_size_8/pytorchgpu.log'







