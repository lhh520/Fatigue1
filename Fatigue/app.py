#!/usr/bin/env python
# -*- coding:utf-8 -*-
from flask import Flask, render_template, Response
import cv2
import numpy as np
from pulse import Pulse
import time
from threading import Lock, Thread
from plot_cont import DynamicPlot
from capture_frames import CaptureFrames
from process_mask import ProcessMasks

from util import *
import multiprocessing as mp
import sys
from optparse import OptionParser
# 导入 camera
from camera import Camera

import cv2
import numpy as np
import torch
from torch import nn
from models import LinkNet34
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageFilter
import time
import sys
import numpy as np
from imutils import face_utils #imutils是在OpenCV基础之上的一个封装
import imutils
import dlib
import cv2
from retinaface import Retinaface


app = Flask(__name__)

global fraAMEE


#定义眼睛阈值和嘴阈值算式函数
def eye_detect_1(eye):
    #np.linalg.norm计算时间要比dist.euclidean要快  因此使用Numpy的算法
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) /(2.0 * C)
    return ear
#常数：眼睛长宽比/闪烁阈值

def mouth_detect_1(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[8])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar


class CaptureFrames():

    def __init__(self, bs, source, show_mask=False):
        self.frame_counter = 0
        self.batch_size = bs
        self.stop = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LinkNet34()
        self.model.load_state_dict(torch.load('linknet.pth', map_location='cpu'))
        self.model.eval()
        self.model.to(self.device)
        self.show_mask = show_mask

    def __call__(self, pipe, source):
        self.pipe = pipe
        self.capture_frames(source)

    def capture_frames(self, source):
        global fraAMEE
        retinaface = Retinaface()
        eye_Aspect_rato = 0.2
        eye_Flicker_threshold = 3
        # 常数：嘴长宽比/闪烁阈值
        mouth_Aspect_rato = 0.5
        mouth_Flicker_threshold = 3
        # 计数常数
        eCOUNTER = 0
        eTOTAL = 0
        mCOUNTER = 0
        mTOTAL = 0
        Roll = 0  # 整个循环内的帧计数
        Rolleye = 0  # 循环内闭眼帧数
        Rollmouth = 0  # 循环内打哈欠数
        hr = 0
        # 脸部位置检测器
        # 返回值就是一个矩形
        detect = dlib.get_frontal_face_detector()

        # 脸部特征位置检测器
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        # 获得左右眼和嘴的标志索引
        (lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        (rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        camera = cv2.VideoCapture(source)
        time.sleep(1)
        self.model.eval()
        (grabbed, frame) = camera.read()

        time_1 = time.time()
        self.frames_count = 0
        while grabbed:
            (grabbed, orig) = camera.read()
            if not grabbed:
                continue

            frame1 = imutils.resize(orig)
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            # 返回值为脸部区域的矩形框，数组形式
            rects = detect(gray, 0)
            # rects= retinaface.detect_image1(frame1)
            # print(type(rects))
            for rect in rects:
                # 检测器检测特征点
                shape = predictor(gray, rect)
                # 将特征点转化为数组组，返回值是68个特征点坐标
                shape = face_utils.shape_to_np(shape)
                # print(shape)
                # 获取左右眼睛和嘴的坐标
                lefteye = shape[lstart:lend]
                righteye = shape[rstart:rend]
                mouth = shape[mstart:mend]
                # 计算左右眼的ear值,嘴的mar值。
                leftear = eye_detect_1(lefteye)
                rightear = eye_detect_1(righteye)
                ear = (leftear + rightear) / 2.0
                mar = mouth_detect_1(mouth)
                # cv2.convexHull为获取图像凸包位置的函数
                lefteyehull = cv2.convexHull(lefteye)
                righteyehull = cv2.convexHull(righteye)
                mouthhull = cv2.convexHull(mouth)
                # print(lefteyehull)
                # cv2.drawContours是轮廓绘制函数
                # 第一个参数是指明在哪幅图像上绘制轮廓；image为三通道才能显示轮廓
                # 第二个参数是轮廓本身，在Python中是一个list;
                # 第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。
                cv2.drawContours(frame1, [lefteyehull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame1, [righteyehull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame1, [mouthhull], -1, (0, 255, 0), 1)
                # 进行画图操作，用矩形框标注人脸
                # left = rect.left()
                # top = rect.top()
                # right = rect.right()
                # bottom = rect.bottom()
                # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                # 循环，满足条件的眨眼次数+1
                if ear < eye_Aspect_rato:  # 阈值0.2
                    eCOUNTER += 1
                    Rolleye += 1
                # 连续3帧都满足条件的计为一次闭眼
                else:
                    if eCOUNTER >= eye_Flicker_threshold:  # 阈值3
                        eTOTAL += 1
                    eCOUNTER = 0
                cv2.putText(frame1, 'Blinks:{}'.format(eTOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame1, 'eCounter:{}'.format(eCOUNTER), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                cv2.putText(frame1, 'Ear:{:.2f}'.format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if mar > mouth_Aspect_rato:  # 阈值0.5
                    mCOUNTER += 1
                    Rollmouth += 1
                else:
                    if mCOUNTER >= mouth_Flicker_threshold:  # 阈值3
                        mTOTAL += 1
                    mCOUNTER = 0
                cv2.putText(frame1, 'Yawning:{}'.format(mTOTAL), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                cv2.putText(frame1, 'mCounter:{}'.format(mCOUNTER), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                # cv2.putText(frame1, 'Mar:{:.2f}'.format(mar), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame1, 'HR:{}'.format(hr), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if eTOTAL >= 12 or mTOTAL >= 6:
                cv2.putText(frame1, "SLEEP!SLEEP!!", (360, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 255, 200), 2)
            if hr > 120:
                cv2.putText(frame1, "DANGER!DANGER!", (360, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 255, 200), 2)
            #cv2.imshow('pilao', frame1)
            fraAMEE=frame1
            Roll += 1

            try:

                file = open('/Fatigue\\mq.txt', encoding="utf-8")
                stt = file.read()
                hr = int(stt)
                file.close()
            except:
                print("wait")

            if Roll == 150:
                # 计算Perclos模型得分
                perclos = (Rolleye / Roll) + (Rollmouth / Roll) * 0.2
                # 在前端UI输出perclos值
                # Ui_MainWindow.printf(window, "过去150帧中，Perclos得分为" + str(round(perclos, 3)))
                print("过去150帧中，Perclos得分为" + str(round(perclos, 3)))
                # 当过去的150帧中，Perclos模型得分超过0.38时，判断为疲劳状态
                if perclos > 0.38:
                    print("当前处于疲劳状态")
                else:
                    print("当前处于清醒状态")

                # 归零
                # 将三个计数器归零
                # 重新开始新一轮的检测
                Roll = 0
                Rolleye = 0
                Rollmouth = 0

            shape = orig.shape[0:2]

            frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256), cv2.INTER_LINEAR)

            k = cv2.waitKey(1)
            if k != -1:
                self.terminate(camera)
                break

            a = img_transform(Image.fromarray(frame))
            a = a.unsqueeze(0)
            imgs = Variable(a.to(dtype=torch.float, device=self.device))
            pred = self.model(imgs)

            pred = torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])
            mask = pred.data.cpu().numpy()
            mask = mask.squeeze()

            # im = Image.fromarray(mask)
            # im2 = im.filter(ImageFilter.MinFilter(3))
            # im3 = im2.filter(ImageFilter.MaxFilter(5))
            # mask = np.array(im3)

            mask = mask > 0.8
            orig[mask == 0] = 0
            self.pipe.send([orig])

            # if self.show_mask:
            #     cv2.imshow('mask', orig)

            if self.frames_count % 30 == 29:
                time_2 = time.time()
                sys.stdout.write(f'\rFPS: {30 / (time_2 - time_1)}')
                sys.stdout.flush()
                time_1 = time.time()

            self.frames_count += 1

        self.terminate(camera)

    def terminate(self, camera):
        self.pipe.send(None)
        cv2.destroyAllWindows()
        camera.release()


class RunPOS():
    def __init__(self, sz=270, fs=28, bs=30, plot=False):
        self.batch_size = bs
        self.frame_rate = fs
        self.signal_size = sz
        self.plot = plot

    def __call__(self, source):
        time1 = time.time()

        mask_process_pipe, chil_process_pipe = mp.Pipe()

        self.plot_pipe = None
        if self.plot:
            self.plot_pipe, plotter_pipe = mp.Pipe()
            self.plotter = DynamicPlot(self.signal_size, self.batch_size)
            self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
            self.plot_process.start()
        process_mask = ProcessMasks(self.signal_size, self.frame_rate, self.batch_size)
        mask_processer = mp.Process(target=process_mask, args=(chil_process_pipe, self.plot_pipe, source,), daemon=True)
        mask_processer.start()

        capture = CaptureFrames(self.batch_size, source, show_mask=True)
        capture(mask_process_pipe, source)
        mask_processer.join()
        if self.plot:
            self.plot_process.join()
        time2 = time.time()
        print(f'time {time2 - time1}')



@app.route('/')
def index():
    """视频流主页"""
    return render_template('index.html')


def gen():
    global fraAMEE
    """视频流生成函数"""
    source = 0
    frame = 2
    batchsize = 5
    # source = args.source
    batch_size = bs
    frame_rate = fs
    signal_size = sz
    plot = plot


    time1 = time.time()

    mask_process_pipe, chil_process_pipe = mp.Pipe()

    plot_pipe = None
    if True:
        plot_pipe, plotter_pipe = mp.Pipe()
        plotter = DynamicPlot(self.signal_size, self.batch_size)
        self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()
    process_mask = ProcessMasks(self.signal_size, self.frame_rate, self.batch_size)
    mask_processer = mp.Process(target=process_mask, args=(chil_process_pipe, self.plot_pipe, source,), daemon=True)
    mask_processer.start()

    capture = CaptureFrames(self.batch_size, source, show_mask=True)
    capture(mask_process_pipe, source)
    mask_processer.join()
    if self.plot:
        self.plot_process.join()
    time2 = time.time()
    print(f'time {time2 - time1}')




    runPOS = RunPOS(270, frame, batchsize, True)
    runPOS(source)

    ret1, buffer = cv2.imencode('.jpg', fraAMEE)
    # 将缓存里的流数据转成字节流
    frame = buffer.tobytes()


    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/video_feed')
def video_feed():
    """视频流路由(route).放到 img 标签的 src 属性."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
