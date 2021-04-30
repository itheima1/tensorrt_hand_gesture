#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import argparse
import os
import struct
import sys
import  numpy as np


import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import torch.nn.functional as F



BATCH_SIZE = 1
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 42
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"
EPS = 1e-5

WEIGHT_PATH_21 = "./wts/resnet50_21.wts"
ENGINE_PATH_21 = "./engine/resnet50_21.engine"


WEIGHT_PATH_GUESTURE = "./wts/resnet50-gesture.wts"
ENGINE_PATH_GUESTURE = "./engine/resnet50-gesture.engine"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_weights(file):

    weight_map = {}
    with open(file,"r") as f:
        lines = f.readlines()
    count = int(lines[0])

    assert count == len(lines) - 1

    for i in range(1,count + 1):
        splits = lines[i].strip().split(" ")
        name = splits[0]
        cur_count = int(splits[1])

        assert cur_count + 2 == len(splits)-1
        values = []
        for j in range(3,len(splits)):
            # print(">>>>",splits[j],len(splits))
            values.append(struct.unpack(">f",bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values,dtype=np.float32)
        print(name)
    return weight_map


def addBatchNorm2d(network,weight_map,input,layer_name,eps):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var  = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + eps)

    scale = gamma / var
    shift = -mean / var * gamma + beta

    return network.add_scale(input=input,
                             mode = trt.ScaleMode.CHANNEL,
                             shift = shift,
                             scale = scale)


def bottleneck(network,weight_map,input,in_channels,out_channels,stride,layer_name):
    conv1 = network.add_convolution(input=input,
                                    num_output_maps = out_channels,
                                    kernel_shape = (1,1),
                                    kernel = weight_map[layer_name + "conv1.weight"],
                                    bias = trt.Weights())

    assert conv1

    bn1 = addBatchNorm2d(network,weight_map,conv1.get_output(0),layer_name + "bn1",EPS)
    assert bn1

    relu1 = network.add_activation(bn1.get_output(0),type=trt.ActivationType.RELU)
    assert relu1

    conv2 = network.add_convolution(input=relu1.get_output(0),
                                    num_output_maps=out_channels,
                                    kernel_shape=(3,3),
                                    kernel=weight_map[layer_name + "conv2.weight"],
                                    bias=trt.Weights())
    assert conv2
    conv2.stride = (stride,stride)
    conv2.padding = (1,1)
    bn2 = addBatchNorm2d(network,weight_map,conv2.get_output(0),layer_name + "bn2",EPS)
    assert bn2
    relu2 = network.add_activation(bn2.get_output(0),type=trt.ActivationType.RELU)
    assert relu2

    conv3 = network.add_convolution(input=relu2.get_output(0),
                                    num_output_maps=out_channels * 4,
                                    kernel_shape=(1,1),
                                    kernel=weight_map[layer_name + "conv3.weight"],
                                    bias=trt.Weights())
    assert conv3
    bn3 = addBatchNorm2d(network,weight_map,conv3.get_output(0),layer_name+"bn3",EPS)
    if stride != 1 or in_channels != 4*out_channels:
        conv4 = network.add_convolution(
            input=input,
            num_output_maps=out_channels*4,
            kernel_shape = (1,1),
            kernel=weight_map[layer_name + "downsample.0.weight"],
            bias=trt.Weights()
        )
        assert conv4

        conv4.stride = (stride,stride)
        bn4 = addBatchNorm2d(network,weight_map,conv4.get_output(0),layer_name + "downsample.1", EPS)
        assert bn4

        ew1 = network.add_elementwise(bn4.get_output(0),bn3.get_output(0),trt.ElementWiseOperation.SUM)
    else:
        ew1 = network.add_elementwise(input,bn3.get_output(0),trt.ElementWiseOperation.SUM)
    assert ew1

    relu3 = network.add_activation(ew1.get_output(0),type=trt.ActivationType.RELU)
    assert relu3

    return relu3

def createLenetEngine(maxBatchSize,weight_path,img_h,img_w,num_classes,builder,config,dt):
    weight_map = load_weights(weight_path)
    network = builder.create_network()

    data = network.add_input(INPUT_BLOB_NAME,dt,(3,img_h,img_w))
    assert data

    conv1 = network.add_convolution(input=data,
                                    num_output_maps=64,
                                    kernel_shape=(7,7),
                                    kernel=weight_map["conv1.weight"],
                                    bias=trt.Weights())
    assert conv1
    conv1.stride = (2,2)
    conv1.padding = (3,3)

    bn1 = addBatchNorm2d(network,weight_map,conv1.get_output(0),"bn1",EPS)
    assert bn1

    relu1 = network.add_activation(bn1.get_output(0),type=trt.ActivationType.RELU)
    assert relu1

    pool1 = network.add_pooling(input=relu1.get_output(0),window_size=trt.DimsHW(3,3),type=trt.PoolingType.MAX)
    assert pool1
    pool1.stride = (2,2)
    pool1.padding = (1,1)

    x = bottleneck(network,weight_map,pool1.get_output(0),64,64,1,"layer1.0.")
    x = bottleneck(network,weight_map,x.get_output(0),256,64,1,"layer1.1.")
    x = bottleneck(network,weight_map,x.get_output(0),256,64,1,"layer1.2.")

    x = bottleneck(network,weight_map,x.get_output(0),256,128,2,"layer2.0.")
    x = bottleneck(network,weight_map,x.get_output(0),512,128,1,"layer2.1.")
    x = bottleneck(network,weight_map,x.get_output(0),512,128,1,"layer2.2.")
    x = bottleneck(network,weight_map,x.get_output(0),512,128,1,"layer2.3.")

    x = bottleneck(network,weight_map,x.get_output(0),512,256,2,"layer3.0.")
    x = bottleneck(network,weight_map,x.get_output(0),1024,256,1,"layer3.1.")
    x = bottleneck(network,weight_map,x.get_output(0),1024,256,1,"layer3.2.")
    x = bottleneck(network,weight_map,x.get_output(0),1024,256,1,"layer3.3.")
    x = bottleneck(network,weight_map,x.get_output(0),1024,256,1,"layer3.4.")
    x = bottleneck(network,weight_map,x.get_output(0),1024,256,1,"layer3.5.")

    x = bottleneck(network,weight_map,x.get_output(0),1024,512,2,"layer4.0.")
    x = bottleneck(network,weight_map,x.get_output(0),2048,512,1,"layer4.1.")
    x = bottleneck(network,weight_map,x.get_output(0),2048,512,1,"layer4.2.")

    pool2 = network.add_pooling(x.get_output(0),window_size=trt.DimsHW(7,7),type=trt.PoolingType.AVERAGE)

    assert pool2
    pool2.stride = (1,1)

    print("fc.weight:",weight_map["fc.weight"].shape,pool2.get_output(0).shape)

    fc1 = network.add_fully_connected(input=pool2.get_output(0),num_outputs=num_classes,kernel=weight_map["fc.weight"],bias=weight_map["fc.bias"])

    assert fc1
    fc1.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(fc1.get_output(0))

    # 构建engine
    builder.max_batch_size = maxBatchSize
    builder.max_workspace_size = 1 << 20
    engine = builder.build_engine(network,config)

    del network
    del weight_map

    return engine

def APIToModel_Hand(maxBatchSize):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    engine = createLenetEngine(maxBatchSize,WEIGHT_PATH_21,INPUT_H,INPUT_W,OUTPUT_SIZE,builder,config,trt.float32)
    assert engine

    with open(ENGINE_PATH_21,"wb") as f:
        f.write(engine.serialize())

    del engine
    del builder

def APIToModel_Gesture(maxBatchSize):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    engine = createLenetEngine(maxBatchSize,WEIGHT_PATH_GUESTURE,224,224,14,builder,config,trt.float32)
    assert engine

    with open(ENGINE_PATH_GUESTURE,"wb") as f:
        f.write(engine.serialize())

    del engine
    del builder

def doInference(context,host_in,host_out,batchSize):
    engine = context.engine
    assert engine.num_bindings == 2

    devide_in = cuda.mem_alloc(host_in.nbytes)
    devide_out = cuda.mem_alloc(host_out.nbytes)
    bindings = [int(devide_in),int(devide_out)]
    stream = cuda.Stream()

    cuda.memcpy_htod_async(devide_in,host_in,stream)
    context.execute_async(bindings=bindings,stream_handle = stream.handle)
    cuda.memcpy_dtoh_async(host_out,devide_out,stream)
    stream.synchronize()


import cv2 as cv
import torch
labels = ['one', 'five', 'first', 'ok', 'heart single', 'yearh', 'three', 'four', 'six', 'i love you', 'gun',
                  'thumb up', 'nine', 'pink']


def draw_bd_handpose(img_,hand_,x,y):
    thick = 2
    colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55)]
    #
    cv.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['1']['x']+x), int(hand_['1']['y']+y)), colors[0], thick)
    cv.line(img_, (int(hand_['1']['x']+x), int(hand_['1']['y']+y)),(int(hand_['2']['x']+x), int(hand_['2']['y']+y)), colors[0], thick)
    cv.line(img_, (int(hand_['2']['x']+x), int(hand_['2']['y']+y)),(int(hand_['3']['x']+x), int(hand_['3']['y']+y)), colors[0], thick)
    cv.line(img_, (int(hand_['3']['x']+x), int(hand_['3']['y']+y)),(int(hand_['4']['x']+x), int(hand_['4']['y']+y)), colors[0], thick)
    cv.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['5']['x']+x), int(hand_['5']['y']+y)), colors[1], thick)
    cv.line(img_, (int(hand_['5']['x']+x), int(hand_['5']['y']+y)),(int(hand_['6']['x']+x), int(hand_['6']['y']+y)), colors[1], thick)
    cv.line(img_, (int(hand_['6']['x']+x), int(hand_['6']['y']+y)),(int(hand_['7']['x']+x), int(hand_['7']['y']+y)), colors[1], thick)
    cv.line(img_, (int(hand_['7']['x']+x), int(hand_['7']['y']+y)),(int(hand_['8']['x']+x), int(hand_['8']['y']+y)), colors[1], thick)
    cv.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['9']['x']+x), int(hand_['9']['y']+y)), colors[2], thick)
    cv.line(img_, (int(hand_['9']['x']+x), int(hand_['9']['y']+y)),(int(hand_['10']['x']+x), int(hand_['10']['y']+y)), colors[2], thick)
    cv.line(img_, (int(hand_['10']['x']+x), int(hand_['10']['y']+y)),(int(hand_['11']['x']+x), int(hand_['11']['y']+y)), colors[2], thick)
    cv.line(img_, (int(hand_['11']['x']+x), int(hand_['11']['y']+y)),(int(hand_['12']['x']+x), int(hand_['12']['y']+y)), colors[2], thick)

    cv.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['13']['x']+x), int(hand_['13']['y']+y)), colors[3], thick)
    cv.line(img_, (int(hand_['13']['x']+x), int(hand_['13']['y']+y)),(int(hand_['14']['x']+x), int(hand_['14']['y']+y)), colors[3], thick)
    cv.line(img_, (int(hand_['14']['x']+x), int(hand_['14']['y']+y)),(int(hand_['15']['x']+x), int(hand_['15']['y']+y)), colors[3], thick)
    cv.line(img_, (int(hand_['15']['x']+x), int(hand_['15']['y']+y)),(int(hand_['16']['x']+x), int(hand_['16']['y']+y)), colors[3], thick)
    cv.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['17']['x']+x), int(hand_['17']['y']+y)), colors[4], thick)
    cv.line(img_, (int(hand_['17']['x']+x), int(hand_['17']['y']+y)),(int(hand_['18']['x']+x), int(hand_['18']['y']+y)), colors[4], thick)
    cv.line(img_, (int(hand_['18']['x']+x), int(hand_['18']['y']+y)),(int(hand_['19']['x']+x), int(hand_['19']['y']+y)), colors[4], thick)
    cv.line(img_, (int(hand_['19']['x']+x), int(hand_['19']['y']+y)),(int(hand_['20']['x']+x), int(hand_['20']['y']+y)), colors[4], thick)

def drawhand(img,outputs):

    print(outputs)
    pts_hand = {}
    for i in range(int(outputs.shape[0] / 2)):
        x = (outputs[i * 2 + 0] * float(img_width))
        y = (outputs[i * 2 + 1] * float(img_height))

        pts_hand[str(i)] = {}
        pts_hand[str(i)] = {
            "x": x,
            "y": y,
        }
    draw_bd_handpose(img, pts_hand, 0, 0)  # 绘制关键点连线

    # ------------- 绘制关键点
    for i in range(int(outputs.shape[0] / 2)):
        x = (outputs[i * 2 + 0] * float(img_width))
        y = (outputs[i * 2 + 1] * float(img_height))

        cv.circle(img, (int(x), int(y)), 3, (255, 50, 60), -1)
        cv.circle(img, (int(x), int(y)), 1, (255, 150, 180), -1)


def drawrect(img,x,y,label):
    cv.putText(img,label,(x,y),cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s",action="store_true")
    parser.add_argument("-d",action="store_true")
    args = parser.parse_args()

    if args.s:
        APIToModel_Gesture(BATCH_SIZE)
        APIToModel_Hand(BATCH_SIZE)
    else:

        img = cv.imread("hand_1.jpg")
        img_width = img.shape[1]
        img_height = img.shape[0]
        # 输入图片预处理
        img_ = cv.resize(img, (INPUT_H, INPUT_W), interpolation=cv.INTER_CUBIC)
        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.

        img_ = img_.transpose(2, 0, 1)
        img_ = torch.from_numpy(img_)
        img_ = img_.unsqueeze_(0)

        # 识别人手的21个关键点
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime

        with open(ENGINE_PATH_21,"rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        context = engine.create_execution_context()
        assert context

        data = np.ones(BATCH_SIZE * 3 * INPUT_H * INPUT_W,dtype=np.float32)
        host_in = cuda.pagelocked_empty(BATCH_SIZE * 3 * INPUT_H * INPUT_W,dtype = np.float32)

        # np.copyto(host_in,data.ravel())
        np.copyto(host_in, img_.ravel())

        host_out = cuda.pagelocked_empty(OUTPUT_SIZE,dtype = np.float32)
        print(host_in.shape,host_out.shape)
        doInference(context,host_in,host_out,BATCH_SIZE)

        print(host_out)
        # 绘制人手
        drawhand(img,host_out)


        # 识别人的手势
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime

        with open(ENGINE_PATH_GUESTURE, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        context = engine.create_execution_context()
        assert context


        host_in = cuda.pagelocked_empty(BATCH_SIZE * 3 * INPUT_H * INPUT_W, dtype=np.float32)

        # np.copyto(host_in,data.ravel())
        np.copyto(host_in, img_.ravel())

        host_out = cuda.pagelocked_empty(14, dtype=np.float32)

        doInference(context, host_in, host_out, BATCH_SIZE)

        print(len(host_out),host_out)
        outputs = torch.from_numpy(host_out)


        outputs = F.softmax(outputs,dim=0)
        print(outputs)
        outputs = outputs.cpu().detach().numpy()
        index = np.argmax(outputs)
        label = labels[index]
        drawrect(img,5,25,label)

        cv.imshow('image', img)

        if cv.waitKey() == 27:
            pass

