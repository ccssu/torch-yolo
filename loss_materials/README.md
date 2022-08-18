# yolov5s 显存占用测试结果

- Tesla V100 SXM2 16GB

|bs|框架|显存占用|
|--|--|--|
|4|PyTorch|6956 MiB|
|4|OneFlow|3990 MiB|
|8|PyTorch|7096 MiB|
|8|OneFlow|5956 MiB|
|16|PyTorch|7122 MiB|
|16|OneFlow|10310 MiB|
|32|PyTorch|13874 MiB|
|32|OneFlow|15142 MiB|

# yolov5s Eager速度测试结果

|bs|框架|速度|
|--|--|--|
|4|PyTorch|2.7468314170837402s|
|4|OneFlow|1.6759819984436035s|
|8|PyTorch|1.2717576026916504s|
|8|OneFlow|1.9733967781066895s|
|16|PyTorch|1.6148931980133057s|
|16|OneFlow|1.0597877502441406s|
|32|PyTorch|1.2105162143707275s|
|32|OneFlow||


>>> oneflow.__version__
'0.8.1.dev20220731+cu112'
>>> torch.__version__
'1.11.0+cu102'

 8 卡 跑了2个epoch  平均 17min
 预计 oneflow 8卡跑完300个epoch = 85h
