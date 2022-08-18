# # ! /bin/bash

# # 修改 克隆链接为自己ssh链接
# git clone git@github.com:ultralytics/yolov5.git  # clone repo

# echo 'yes'

# curDir=$(cd $(dirname $0);pwd)

# # echo $curDir

# #  pip 软件 源切换至国内镜像
# cd ~/
# mkdir -p ".pip"
# if  [ -f './.pip/pip.conf' ];
# then
#         echo " the file: pip.conf exis"
# else
#         cd .pip
#         echo "[global]" > pip.conf
#         echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> pip.conf
#         echo "[install]" >> pip.conf
#         echo "trusted-host = https://pypi.tuna.tsinghua.edu.cn">> pip.conf
# fi


# cd  ${curDir}"/yolov5"

# pip install -r requirements.txt

# python3 train.py  # train a model

# python3 test.py --weights yolov5s.pt  # test a model for Precision, Recall and mAP

# python3 detect.py --weights yolov5s.pt --source path/to/images  # run inference on images and videos

# python3 -m pip install oneflow

