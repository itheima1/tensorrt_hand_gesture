"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import random
import sys
import threading
import time

import cv2
import numpy as np

import tensorrt as trt
import torch
import torchvision

from trt_lite2 import TrtLite

INPUT_W = 256
INPUT_H = 256

INPUT_SHAPE = (3, INPUT_H, INPUT_W)
CONF_THRESH = 0.1
IOU_THRESHOLD = 0.4
labels = ['one', 'five', 'first', 'ok', 'heart single', 'yearh', 'three', 'four', 'six', 'i love you', 'gun',
          'thumb up', 'nine', 'pink']

BATCH_SIZE = 2

ENGINE_PATH_21 = "./engine/resnet50_21.engine"
ENGINE_PATH_GESTURE = "./engine/resnet50-gesture.engine"


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


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
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def draw_bd_handpose(img_, hand_, x, y):
    thick = 2
    colors = [(0, 215, 255), (255, 115, 55), (5, 255, 55), (25, 15, 255), (225, 15, 55)]
    #
    cv2.line(img_, (int(hand_['0']['x'] + x), int(hand_['0']['y'] + y)),
             (int(hand_['1']['x'] + x), int(hand_['1']['y'] + y)), colors[0], thick)
    cv2.line(img_, (int(hand_['1']['x'] + x), int(hand_['1']['y'] + y)),
             (int(hand_['2']['x'] + x), int(hand_['2']['y'] + y)), colors[0], thick)
    cv2.line(img_, (int(hand_['2']['x'] + x), int(hand_['2']['y'] + y)),
             (int(hand_['3']['x'] + x), int(hand_['3']['y'] + y)), colors[0], thick)
    cv2.line(img_, (int(hand_['3']['x'] + x), int(hand_['3']['y'] + y)),
             (int(hand_['4']['x'] + x), int(hand_['4']['y'] + y)), colors[0], thick)
    cv2.line(img_, (int(hand_['0']['x'] + x), int(hand_['0']['y'] + y)),
             (int(hand_['5']['x'] + x), int(hand_['5']['y'] + y)), colors[1], thick)
    cv2.line(img_, (int(hand_['5']['x'] + x), int(hand_['5']['y'] + y)),
             (int(hand_['6']['x'] + x), int(hand_['6']['y'] + y)), colors[1], thick)
    cv2.line(img_, (int(hand_['6']['x'] + x), int(hand_['6']['y'] + y)),
             (int(hand_['7']['x'] + x), int(hand_['7']['y'] + y)), colors[1], thick)
    cv2.line(img_, (int(hand_['7']['x'] + x), int(hand_['7']['y'] + y)),
             (int(hand_['8']['x'] + x), int(hand_['8']['y'] + y)), colors[1], thick)
    cv2.line(img_, (int(hand_['0']['x'] + x), int(hand_['0']['y'] + y)),
             (int(hand_['9']['x'] + x), int(hand_['9']['y'] + y)), colors[2], thick)
    cv2.line(img_, (int(hand_['9']['x'] + x), int(hand_['9']['y'] + y)),
             (int(hand_['10']['x'] + x), int(hand_['10']['y'] + y)), colors[2], thick)
    cv2.line(img_, (int(hand_['10']['x'] + x), int(hand_['10']['y'] + y)),
             (int(hand_['11']['x'] + x), int(hand_['11']['y'] + y)), colors[2], thick)
    cv2.line(img_, (int(hand_['11']['x'] + x), int(hand_['11']['y'] + y)),
             (int(hand_['12']['x'] + x), int(hand_['12']['y'] + y)), colors[2], thick)

    cv2.line(img_, (int(hand_['0']['x'] + x), int(hand_['0']['y'] + y)),
             (int(hand_['13']['x'] + x), int(hand_['13']['y'] + y)), colors[3], thick)
    cv2.line(img_, (int(hand_['13']['x'] + x), int(hand_['13']['y'] + y)),
             (int(hand_['14']['x'] + x), int(hand_['14']['y'] + y)), colors[3], thick)
    cv2.line(img_, (int(hand_['14']['x'] + x), int(hand_['14']['y'] + y)),
             (int(hand_['15']['x'] + x), int(hand_['15']['y'] + y)), colors[3], thick)
    cv2.line(img_, (int(hand_['15']['x'] + x), int(hand_['15']['y'] + y)),
             (int(hand_['16']['x'] + x), int(hand_['16']['y'] + y)), colors[3], thick)
    cv2.line(img_, (int(hand_['0']['x'] + x), int(hand_['0']['y'] + y)),
             (int(hand_['17']['x'] + x), int(hand_['17']['y'] + y)), colors[4], thick)
    cv2.line(img_, (int(hand_['17']['x'] + x), int(hand_['17']['y'] + y)),
             (int(hand_['18']['x'] + x), int(hand_['18']['y'] + y)), colors[4], thick)
    cv2.line(img_, (int(hand_['18']['x'] + x), int(hand_['18']['y'] + y)),
             (int(hand_['19']['x'] + x), int(hand_['19']['y'] + y)), colors[4], thick)
    cv2.line(img_, (int(hand_['19']['x'] + x), int(hand_['19']['y'] + y)),
             (int(hand_['20']['x'] + x), int(hand_['20']['y'] + y)), colors[4], thick)


def drawhand(img, outputs, img_width, img_height):
    # print(outputs)
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

        cv2.circle(img, (int(x), int(y)), 3, (255, 50, 60), -1)
        cv2.circle(img, (int(x), int(y)), 1, (255, 150, 180), -1)


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path, batch_size=1):

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        self.batch_size = batch_size
        input_volume = trt.volume(INPUT_SHAPE)
        self.numpy_array = np.zeros((self.batch_size, input_volume))

        trt_yolo = TrtLite(engine_file_path=engine_file_path)
        trt_yolo.print_info()

        # Execution context is needed for inference
        self.context = trt_yolo.engine.create_execution_context()

        self.buffers = trt_yolo.allocate_io_buffers(self.batch_size, True)
        self.trt_yolo = trt_yolo

        # 识别人手的21个关键点
        self.trt_lite21 = TrtLite(engine_file_path=ENGINE_PATH_21)
        self.trt_lite21.print_info()

        # 识别手势
        self.trt_lite_gesture = TrtLite(engine_file_path=ENGINE_PATH_GESTURE)
        self.trt_lite_gesture.print_info()

    def _load_imgs(self, image_paths):
        for idx, image_path in enumerate(image_paths):
            img_np, image_raw, h, w = self.preprocess_image(image_path)
            # print("---------------------------------",img_np.shape, image_raw.shape)
            self.numpy_array[idx] = img_np.ravel()
        return self.numpy_array

    def infer_batch(self, image_paths):

        # Load all images to CPU...
        imgs = self._load_imgs(image_paths)

        # -------------------- hand detect start
        t1 = time_synchronized()

        # ...copy them into appropriate place into memory...
        # (self.inputs was returned earlier by allocate_buffers())
        self.buffers[0] = torch.from_numpy(imgs.ravel()).cuda()

        bindings = [t.data_ptr() for t in self.buffers]

        self.trt_yolo.execute(bindings, BATCH_SIZE)

        host_outputs = self.buffers[1].clone().cpu().detach().numpy()

        torch.cuda.synchronize()

        t2 = time_synchronized()
        latency_in_sec = (t2 - t1)

        print("Latency - {:.2f} ms, handle_size: {}".format(latency_in_sec * 1000, len(image_paths)))

        # -------------------- hand detect end

        return latency_in_sec

    def doInference(self, index, image_path):
        # threading.Thread.__init__(self)

        # Do image preprocess
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_path)

        # -------------------- hand detect start
        t1 = time_synchronized()
        self.buffers[0] = torch.from_numpy(input_image.ravel()).cuda()

        bindings = [t.data_ptr() for t in self.buffers]

        self.trt_yolo.execute(bindings, BATCH_SIZE)

        host_outputs = self.buffers[1].clone().cpu().detach().numpy()

        torch.cuda.synchronize()

        t2 = time_synchronized()
        latency_in_sec = (t2 - t1)
        # -------------------- hand detect end

        # print(host_outputs.shape)
        output = host_outputs.ravel()
        # Do postprocess
        result_boxes, result_scores, result_classid = self.post_process(
            output, origin_h, origin_w
        )

        # print(output.shape,len(result_boxes))
        # Draw rectangles and labels on the original image
        for i in range(len(result_boxes)):
            box = result_boxes[i]
            # print("box>>>",box)

            # 截出手的部位
            image_hand = image_raw[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            # 推理手的21个特征点
            hand_data = self.preprocess_hand(image_hand)

            if hand_data is None:
                continue

            output21 = self.doInference_resnet(self.trt_lite21, hand_data.ravel())

            # 推理手势
            output_gesture = self.doInference_resnet(self.trt_lite_gesture, hand_data.ravel())
            # print("gesture:",output_gesture)
            index = np.argmax(output_gesture)
            label = labels[index]

            hand_width = int(box[2]) - int(box[0])
            hand_height = int(box[3]) - int(box[1])
            drawhand(image_hand, output21, hand_width, hand_height)

            # print("w,h:",hand_width,hand_height)
            # cv2.imwrite("hand_11.jpg", image_hand)

            plot_one_box(
                box,
                image_raw,
                label="{}:{:.2f}".format(
                    label, result_scores[i]
                ),
            )
        parent, filename = os.path.split(image_path)

        out_dir = "out_dir"
        exists = os.path.exists(out_dir)

        if not exists:
            os.mkdir(out_dir)

        save_name = os.path.join(out_dir, "output_" + filename)
        # Save image
        # print("save to :", save_name)
        cv2.imwrite(save_name, image_raw)

        print("Latency - {:.2f} ms, Saved: {}".format(latency_in_sec * 1000, save_name))

        # print("save img success")

        return latency_in_sec

    def doInference_resnet(self, trt_engine, data):
        i2shape = 1
        io_info = trt_engine.get_io_info(i2shape)
        # print(io_info)
        d_buffers = trt_engine.allocate_io_buffers(i2shape, True)
        # print(io_info[1][2])

        d_buffers[0] = data.cuda()

        bindings = [t.data_ptr() for t in d_buffers]

        # 进行推理
        trt_engine.execute(bindings, i2shape)

        output_data_trt = d_buffers[1].clone().cpu().detach().numpy()

        torch.cuda.synchronize()

        host_out = output_data_trt.ravel()

        return host_out

    def preprocess_hand(self, img):
        img_width = img.shape[1]
        img_height = img.shape[0]
        # print(img.shape)
        if img_width < 1 or img_height < 1:
            return None
        # 输入图片预处理
        img_ = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.

        img_ = img_.transpose(2, 0, 1)
        img_ = torch.from_numpy(img_)
        img_ = img_.unsqueeze_(0)
        return img_

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
        image_raw = cv2.imread(input_image_path)
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
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
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
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


class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


# class TRTInference(object):
#
#     def __init__(self, batch_size):
#         self.batch_size = batch_size
#         input_volume = trt.volume(INPUT_SHAPE)
#         self.numpy_array = np.zeros((self.batch_size, input_volume))
#
#         # This allocates memory for network inputs/outputs on both CPU and GPU
#         self.inputs, self.outputs, self.bindings, self.stream = \
#             self.allocate_buffers(yolov5_wrapper.trt_yolo.engine)


import sys


def run_infer(image_names):
    total_latency = 0
    total_count = 0
    for input_image_name in image_names:
        # create a new thread to do inference
        # thread1 = myThread(yolov5_wrapper.doInference, [dest_dir + input_image_name])
        # thread1.start()
        # thread1.join()
        image_path_str = os.path.join(dest_dir, input_image_name)
        latency_in_sec = yolov5_wrapper.doInference(total_count, image_path_str)
        total_latency += latency_in_sec
        total_count += 1
    latency_average = total_latency / total_count
    print("Average latency: {:.2f} ms with count: {}, batch_size={}".format(latency_average * 1000, total_count,
                                                                            BATCH_SIZE))


def run_infer_batch(image_names):
    total_imgs = len(image_names)

    total_latency = 0
    total_count = 0

    for idx in range(0, len(image_names), BATCH_SIZE):
        imgs = image_names[idx:idx + BATCH_SIZE]

        print("Infering image {}/{}".format(idx + 1, total_imgs))

        image_paths = [os.path.join(dest_dir, img) for img in imgs]
        latency_in_sec = yolov5_wrapper.infer_batch(image_paths)
        total_latency += latency_in_sec
        total_count += 1

    latency_average = total_latency / total_count
    print("Average latency: {:.2f} ms with count: {}, batch_size={}".format(latency_average * 1000, total_count,
                                                                            BATCH_SIZE))


if __name__ == "__main__":
    # load custom plugins
    PLUGIN_LIBRARY = "./libmyplugins.so"
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = "./yolov5s-hand-docker-comm.engine"

    # global BATCH_SIZE

    if len(sys.argv) > 1:
        BATCH_SIZE = int(sys.argv[1])

    # load coco labels

    categories = ["hand"]

    # a  YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path, BATCH_SIZE)

    print("-------trt_yolo -> max_batch_size:", yolov5_wrapper.trt_yolo.engine.max_batch_size)
    print("-------trt_lite21 -> max_batch_size:", yolov5_wrapper.trt_lite21.engine.max_batch_size)
    print("-------trt_lite_gesture -> max_batch_size:", yolov5_wrapper.trt_lite_gesture.engine.max_batch_size)

    # input_image_paths = ["images/2.jpg"]

    # dest_dir = "/home/ty/Workspace/Hackathon/datasets/hand_images"
    dest_dir = "/root/ws/hand_images"
    if len(sys.argv) > 2:
        dest_dir = sys.argv[2]

    image_names = os.listdir(dest_dir)
    print("image_count:", len(image_names))

    is_save_img = False
    if len(sys.argv) > 3 and sys.argv[3] == "1":
        is_save_img = True

    if is_save_img:
        run_infer(image_names)
    else:
        run_infer_batch(image_names)

    # destroy the instance
    # yolov5_wrapper.destroy()
