import tensorrt as trt
import os
import common
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
model_path = "./resnet50.onnx"
engine_path = "./resnet_hand.engine"

def create_engine():
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_path, 'rb') as model:
            flag = parser.parse(model.read())
            if not flag:
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        # 构建engine对象
        network.get_input(0).shape = [1,3,256,256]
        # builder.int8_mode = True



        engine = builder.build_cuda_engine(network)
        # 序列化engine
        print("构建engine成功")
        with open(engine_path,"wb") as f:
            f.write(engine.serialize())
        return engine

def load_engine(engine_path):
    if os.path.exists(engine_path):
        # 读取engine
        with open(engine_path,"rb") as f,trt.Runtime(TRT_LOGGER) as runtime:
            print("load engine success")
            return runtime.deserialize_cuda_engine(f.read())
    print("load engine error!!!")
    return None


def get_engine():
    engine = None
    # create_engine()
    if os.path.exists(engine_path):
        engine = load_engine(engine_path)
    else:
        engine = create_engine()
    return engine


import cv2 as cv
import torch
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


if __name__ == '__main__':
    img = cv.imread("1.jpg")
    img_width = img.shape[1]
    img_height = img.shape[0]
    # 输入图片预处理
    img_ = cv.resize(img, (256,256), interpolation=cv.INTER_CUBIC)
    img_ = img_.astype(np.float32)
    img_ = (img_ - 128.) / 256.

    img_ = img_.transpose(2, 0, 1)
    img_ = torch.from_numpy(img_)
    img_ = img_.unsqueeze_(0)

    print(img_.shape)

    # 使用engine进行推理
    with get_engine() as engine,engine.create_execution_context() as context:
        print(context.get_binding_shape(0))
        print(context.get_binding_shape(1))
        inputs = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        outputs = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)

        print(inputs,type(inputs),inputs.shape,img_.shape)

        np.copyto(inputs,img_.ravel())

        d_input = cuda.mem_alloc(inputs.nbytes)
        d_output = cuda.mem_alloc(outputs.nbytes)

        stream = cuda.Stream()

        cuda.memcpy_htod_async(d_input,inputs,stream)

        # 推理
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

        cuda.memcpy_dtoh_async(outputs, d_output, stream)

        stream.synchronize()

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



        cv.imshow('image', img)

        if cv.waitKey() == 27:
            pass













