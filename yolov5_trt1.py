"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import random
import sys
import threading
import time

import cv2 as cv
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision

INPUT_W = 256
INPUT_H = 256
CONF_THRESH = 0.1
IOU_THRESHOLD = 0.4
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv.rectangle(img, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv.rectangle(img, c1, c2, color, -1, cv.LINE_AA)  # filled
        cv.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv.LINE_AA,
        )


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

        # # 加载21个关键点模型
        self.context21 = self.get_contextFromEngine("./resnet50.engine")
        # # 加载手势识别模型
        # self.contenxt_guesture = self.get_contextFromEngine("./resnet-gesture.engine")

    def infer(self, input_image_path):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        # self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(input_image_path)
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()

        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        result_boxes, result_scores, result_classid = self.post_process(
            output, origin_h, origin_w
        )
        # Draw rectangles and labels on the original image
        for i in range(len(result_boxes)):
            box = result_boxes[i]
            plot_one_box(
                box,
                image_raw,
                label="{}:{:.2f}".format(
                    categories[int(result_classid[i])], result_scores[i]
                ),
            )
            print(box)

            hand_img = image_raw[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
            pre_hand_img = process_img(hand_img)


            # 使用resnet 推理每只手的21个关键点
            host_in = cuda.pagelocked_empty(1 * 3 * INPUT_H * INPUT_W, dtype=np.float32)
            np.copyto(host_in, pre_hand_img.ravel())
            host_out = cuda.pagelocked_empty(42, dtype=np.float32)

            self.doInference(self.context21,host_in,host_out,1)
            print(">>>>>",host_out)
            # 使用resnet 推理每只手的手势

            # host_in = cuda.pagelocked_empty(1 * 3 * INPUT_H * INPUT_W, dtype=np.float32)
            # np.copyto(host_in, pre_hand_img.ravel())
            # host_out = cuda.pagelocked_empty(14, dtype=np.float32)
            #
            # doInference(context, host_in, host_out, 1)
            #
            # print(len(host_out), host_out)
            # outputs = torch.from_numpy(host_out)

        # Remove any context from the top of the context stack, deactivating it.
        # self.cfx.pop()
        parent, filename = os.path.split(input_image_path)
        save_name = os.path.join(parent, "output_" + filename)
        # 　Save image
        cv.imwrite(save_name, image_raw)


    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def preprocess_image(self, input_image_path):
        """
        description: Read an image from image path, convert it to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        print(input_image_path)
        image_raw = cv.imread(input_image_path)
        h, w, c = image_raw.shape
        image = cv.cvtColor(image_raw, cv.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # to a torch Tensor
        pred = torch.Tensor(pred).cuda()
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # Do nms
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        return result_boxes, result_scores, result_classid

    def get_contextFromEngine(self,engine_path):
        # 识别人手的21个关键点
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        context = engine.create_execution_context()
        assert context
        return context

    def doInference(self,context, host_in, host_out, batchSize):
        engine = context.engine
        assert engine.num_bindings == 2

        devide_in = cuda.mem_alloc(host_in.nbytes)
        devide_out = cuda.mem_alloc(host_out.nbytes)
        bindings = [int(devide_in), int(devide_out)]
        stream = cuda.Stream()

        cuda.memcpy_htod_async(devide_in, host_in, stream)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_out, devide_out, stream)
        stream.synchronize()

class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)





def process_img(img):
    img_width = img.shape[1]
    img_height = img.shape[0]
    # 输入图片预处理
    img_ = cv.resize(img, (INPUT_H, INPUT_W), interpolation=cv.INTER_CUBIC)
    img_ = img_.astype(np.float32)
    img_ = (img_ - 128.) / 256.

    img_ = img_.transpose(2, 0, 1)
    img_ = torch.from_numpy(img_)
    img_ = img_.unsqueeze_(0)

    return img_


if __name__ == "__main__":
    # load custom plugins
    PLUGIN_LIBRARY = "./libmyplugins.so"
    ctypes.CDLL(PLUGIN_LIBRARY)
    # 识别出人手的位置
    engine_file_path = "./yolov5s-hand.engine"

    # load coco labels

    categories = ["hand"]

    # a  YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path)



    # from https://github.com/ultralytics/yolov5/tree/master/inference/images
    input_image_paths = ["1.jpg"]

    yolov5_wrapper.infer("1.jpg")

    # for input_image_path in input_image_paths:
    #     # create a new thread to do inference
    #     thread1 = myThread(yolov5_wrapper.infer, ["./"+input_image_path])
    #     thread1.start()
    #     thread1.join()
    # cv.waitKey()
    # destroy the instance
    yolov5_wrapper.destroy()
