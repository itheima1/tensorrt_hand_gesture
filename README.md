## 进入docker环境
docker run --gpus all -it --rm -v /root/ws/:/root/ws/ nvcr.io/nvidia/pytorch:20.11-py

# 进到工程目录下：
cd ~/ws/tensorrtHack2_1

##模型导出
不同设备环境，需要重新导出engine文件
python3 resnet50_demo.py -s

## yolov5模型导出
./yolov5 -s yolovs-hand.wts yolov5s-hand.engine s

## yolov5-int8优化模型导出
./yolov5 -s yolov5s-hand-int8.wts yolov5s-hand-int8.engine s

# 测试结果
测试没有优化的模型
python yolov5_trt12.py
测试int8量化的模型
python yolov5_trt12_int8.py

# 模型文件地址
链接: https://pan.baidu.com/s/1QaKLq4q-FRbuCl0asEJ9hg 提取码: qrtc
