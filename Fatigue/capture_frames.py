import math

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
import subprocess as sp
#rtsp_url = 'rtsp://127.0.0.1:8554/video'
localhost='192.168.1.102'
rtsp_url='rtsp://'+localhost+':8554/video'
#rtsp_url = 'rtsp://192.168.1.102:8554/video'
import datetime
from statistics import mode
from public_mysql import connect
from torch1 import FaceCNN

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

def preprocess_input(images):
    """ preprocess input by substracting the train mean
    # Arguments: images or image of any shape
    # Returns: images or image with substracted train mean (129)
    """
    images = images/255.0
    return images
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)
def implement1(eye,mou,rate,pico,reason,path):
    '''执行SQL语句'''
    db = connect()
    cursor = db.cursor()
    for i in range(1):
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        try:
            cursor.execute("insert into user(eye,mou,rate,pico,reason,update_time,path) \
                                          values('%d','%d','%d','%s','%s','%s','%s')" % \
                           (eye, mou, rate, pico,reason, dt,path))
            result = cursor.fetchone()
            db.commit()

        except Exception:
            db.rollback()


    cursor.close()
    db.close()


# 获取最大的人脸
def _largest_face(dets):
    if len(dets) == 1:
        return 0

    face_areas = [(det.right() - det.left()) * (det.bottom() - det.top()) for det in dets]

    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(dets)):
        if face_areas[index] > largest_area:
            largest_index = index
            largest_area = face_areas[index]

    print("largest_face index is {} in {} faces".format(largest_index, len(dets)))

    return largest_index


# 从dlib的检测结果抽取姿态估计需要的点坐标
def get_image_points_from_landmark_shape(landmark_shape):
    POINTS_NUM_LANDMARK = 68
    if landmark_shape.num_parts != POINTS_NUM_LANDMARK:
        print("ERROR:landmark_shape.num_parts-{}".format(landmark_shape.num_parts))
        return -1, None

    # 2D image points. If you change the image, you need to change vector
    image_points = np.array([
        (landmark_shape.part(30).x, landmark_shape.part(30).y),  # Nose tip
        (landmark_shape.part(8).x, landmark_shape.part(8).y),  # Chin
        (landmark_shape.part(36).x, landmark_shape.part(36).y),  # Left eye left corner
        (landmark_shape.part(45).x, landmark_shape.part(45).y),  # Right eye right corne
        (landmark_shape.part(48).x, landmark_shape.part(48).y),  # Left Mouth corner
        (landmark_shape.part(54).x, landmark_shape.part(54).y)  # Right mouth corner
    ], dtype="double")

    return 0, image_points


# 用dlib检测关键点，返回姿态估计需要的几个点坐标
def get_image_points(img,detector,predictor):
    # gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )  # 图片调整为灰色
    dets = detector(img, 0)

    if 0 == len(dets):
        print("ERROR: found no face")
        return -1, None
    largest_index = _largest_face(dets)
    face_rectangle = dets[largest_index]

    landmark_shape = predictor(img, face_rectangle)

    return get_image_points_from_landmark_shape(landmark_shape)


# 获取旋转向量和平移向量
def get_pose_estimation(img_size, image_points):
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])

    # Camera internals

    focal_length = img_size[1]
    center = (img_size[1] / 2, img_size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # print("Camera Matrix :{}".format(camera_matrix))

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # print("Rotation Vector:\n {}".format(rotation_vector))
    # print("Translation Vector:\n {}".format(translation_vector))
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs


# 从旋转向量转换为欧拉角
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)

    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0][0] / theta
    y = math.sin(theta / 2) * rotation_vector[1][0] / theta
    z = math.sin(theta / 2) * rotation_vector[2][0] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    #print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    #print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    # 单位转换：将弧度转换为度
    Y = int((pitch / math.pi) * 180)
    X = int((yaw / math.pi) * 180)
    Z = int((roll / math.pi) * 180)

    return 0, Y, X, Z


def get_pose_estimation_in_euler_angle(landmark_shape, im_szie):
    try:
        ret, image_points = get_image_points_from_landmark_shape(landmark_shape)
        if ret != 0:
            print('get_image_points failed')
            return -1, None, None, None

        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(im_szie,
                                                                                                   image_points)
        if ret != True:
            print('get_pose_estimation failed')
            return -1, None, None, None

        ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
        if ret != 0:
            print('get_euler_angle failed')
            return -1, None, None, None

        euler_angle_str = 'Y:{}, X:{}, Z:{}'.format(pitch, yaw, roll)
        print(euler_angle_str)
        return 0, pitch, yaw, roll

    except Exception as e:
        print('get_pose_estimation_in_euler_angle exception:{}'.format(e))
        return -1, None, None, None




class FaceCNN(nn.Module):
    # 初始化网络结构
    def __init__(self):
        super(FaceCNN, self).__init__()

        # 第一次卷积、池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.BatchNorm2d(num_features=64),  # 归一化
            nn.RReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大值池化
        )

        # 第二次卷积、池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第三次卷积、池化
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 参数初始化
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),
        )

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 数据扁平化
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y





class CaptureFrames():
    def __init__(self, bs, source, show_mask=False):
        self.frame_counter = 0
        self.batch_size = bs
        self.stop = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LinkNet34()
        self.model.load_state_dict(torch.load('linknet.pth',map_location='cpu'))
        self.model.eval()
        self.model.to(self.device)
        self.show_mask = show_mask

    def __call__(self, pipe, source):
        self.pipe = pipe
        self.capture_frames(source)

    def capture_frames(self, source):
        retinaface = Retinaface()
        perclos=0
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
        hr=0
        #分心次数
        fenxi=0
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
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # source=0
        source="疲劳驾驶.mp4"
        camera = cv2.VideoCapture(source)
        #fps = int(camera.get(cv2.CAP_PROP_FPS))
        fps = 1
        #time.sleep(1)
        self.model.eval()
        (grabbed, frame) = camera.read()

        time_1 = time.time()
        self.frames_count = 0
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # width = 480
        # height = 640
        command = ['ffmpeg',
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', "{}x{}".format(width, height),
                   '-r', str(fps),
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'ultrafast',
                   '-f', 'rtsp',
                   rtsp_url]
        p = sp.Popen(command, stdin=sp.PIPE)


        #图像的获取

        detection_model_path = 'haarcascade_frontalface_default.xml'
        classification_model_path = 'E:\\Fatigue\\Fatigue\\model_net.pkl'
        frame_window = 10

        emotion_labels = {0: 'angry', 1: 'disgust', 2: 'sad', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

        # 加载人脸检测模型
        face_detection = cv2.CascadeClassifier(detection_model_path)

        # 加载表情识别
        emotion_classifier = torch.load(classification_model_path)
        emotion_window = []
        while grabbed:
            (grabbed, orig) = camera.read()
            if not grabbed:
                continue
            #p.stdin.write(orig.tostring())




            frame1 = imutils.resize(orig)
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            # 返回值为脸部区域的矩形框，数组形式
            rects = detect(gray, 0)
            #rects= retinaface.detect_image1(frame1)
            #print(type(rects))



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
                cv2.putText(frame1, 'Yawning:{}'.format(mTOTAL), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame1, 'mCounter:{}'.format(mCOUNTER), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #cv2.putText(frame1, 'Mar:{:.2f}'.format(mar), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame1, 'HR:{}'.format(hr), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if eTOTAL >= 12 or mTOTAL >= 1:
                cv2.putText(frame1, "SLEEP!SLEEP!!", (360, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 255, 200), 2)
                #implement1(eye, mou, rate, pico, reason):
                dt = datetime.datetime.now().strftime("%H%M%S")
                #D:\face\web\img
                path='D:\\face\\web\\img\\' + dt + '.png'
                #path2='http://localhost:8000/img/' + dt + '.png'
                path2 = 'http://'+localhost+':8000/img/' + dt + '.png'
                cv2.imwrite(path, frame1)
                implement1(eTOTAL,mTOTAL,hr,str(round(perclos, 3)),"闭眼或者哈欠次数过多",path2)
                eTOTAL=0
                mTOTAL=0
            if hr>120:
                cv2.putText(frame1, "DANGER!DANGER!", (360, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 255, 200), 2)
                dt = datetime.datetime.now().strftime("%H%M%S")
                #path = 'E:\\images\\' + dt + '.jpg'
                path = 'D:\\face\\web\\img\\' + dt + '.png'
                cv2.imwrite(path, frame1)
                #path2 = 'http://localhost:8000/img/' + dt + '.png'
                path2 = 'http://' + localhost + ':8000/img/' + dt + '.png'
                implement1(eTOTAL, mTOTAL, hr, str(round(perclos, 3)), "心跳过快",path2)

            gray11 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

            faces = face_detection.detectMultiScale(gray11, 1.3, 5)

            # 对于所有发现的人脸
            for (x, y, w, h) in faces:
                # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)

                # 获取人脸图像
                face = gray[y:y + h, x:x + w]

                try:
                    # shape变为(48,48)
                    face = cv2.resize(face, (48, 48))
                except:
                    continue

                # 扩充维度，shape变为(1,48,48,1)
                # 将（1，48，48，1）转换成为(1,1,48,48)
                face = np.expand_dims(face, 0)
                face = np.expand_dims(face, 0)
                # 人脸数据归一化，将像素值从0-255映射到0-1之间
                face = preprocess_input(face)
                new_face = torch.from_numpy(face)
                new_new_face = new_face.float().requires_grad_(False)

                # 调用我们训练好的表情识别模型，预测分类
                emotion_arg = np.argmax(emotion_classifier.forward(new_new_face).detach().numpy())
                emotion = emotion_labels[emotion_arg]

                emotion_window.append(emotion)

                if len(emotion_window) >= frame_window:
                    emotion_window.pop(0)

                try:
                    # 获得出现次数最多的分类
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                # 在矩形框上部，输出分类文字
                cv2.putText(frame1, 'emotion:'+emotion_mode, (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #cv2.putText(frame1, emotion_mode, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow('pilao', frame1)

            #进行分心的检测
            im=orig
            size111 = im.shape
            if size111[0] > 700:
                h = size111[0] / 3
                w = size111[1] / 3
                im = cv2.resize(im, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
                size111 = im.shape
            ret222, image_points11 = get_image_points(im,detect,predictor)
            try:
                ret222, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(size111,image_points11)
                if ret222 != True:
                    print('get_pose_estimation failed')
                    continue
                ret222, pitch1, yaw1, roll1 = get_euler_angle(rotation_vector)
                if yaw1<-10and yaw1>10:
                    fenxi=fenxi+1
                print(type(yaw1))
                euler_angle_str = 'Y:{}, X:{}, Z:{}'.format(pitch1, yaw1, roll1)
                print(euler_angle_str)
            except:
                print('无法准确检测人脸')




            #分心检测结束

            p.stdin.write(frame1.tostring())
            #print(frame1.shape)
            Roll += 1

            try:
                file = open('mq.txt', encoding="utf-8")
                stt = file.read()
                hr=int(stt)
                file.close()
            except:
                print("wait")
            if Roll == 150:
                # 计算Perclos模型得分
                perclos = (Rolleye / Roll) + (Rollmouth / Roll) * 0.2
                #分心计算公式：
                feixinlos=1.0*fenxi/150
                # 在前端UI输出perclos值
                # Ui_MainWindow.printf(window, "过去150帧中，Perclos得分为" + str(round(perclos, 3)))
                print("过去150帧中，Perclos得分为" + str(round(perclos, 3)))
                # 当过去的150帧中，Perclos模型得分超过0.38时，判断为疲劳状态
                if perclos > 0.38:
                    cv2.putText(frame1, "perclos>0.38|DANGER!DANGER!", (360, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 255, 200), 2)
                    dt = datetime.datetime.now().strftime("%H%M%S")
                    path = 'D:\\face\\web\\img\\' + dt + '.png'
                    cv2.imwrite(path, frame1)
                    #path2 = 'http://localhost:8000/img/' + dt + '.png'
                    path2 = 'http://' + localhost + ':8000/img/' + dt + '.png'
                    #path = 'E:\\images\\' + dt + '.jpg'
                    implement1(eTOTAL, mTOTAL, hr, str(round(perclos, 3)), "可能处于疲劳状态",path2)
                    print("当前处于疲劳状态")
                else:
                    print("当前处于清醒状态")
                if feixinlos>0.35:
                    cv2.putText(frame1, "heartclos>0.35|DANGER!DANGER!", (360, 360), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (84, 255, 200), 2)
                    dt = datetime.datetime.now().strftime("%H%M%S")
                    path = 'D:\\face\\web\\img\\' + dt + '.png'
                    cv2.imwrite(path, frame1)
                    # path2 = 'http://localhost:8000/img/' + dt + '.png'
                    path2 = 'http://' + localhost + ':8000/img/' + dt + '.png'
                    # path = 'E:\\images\\' + dt + '.jpg'
                    implement1(eTOTAL, mTOTAL, hr, str(round(perclos, 3)), "可能处于疲劳状态", path2)
                    print("当前处于疲劳状态")
                else:
                    print("当前处于清醒状态")



                # 归零
                # 将三个计数器归零
                # 重新开始新一轮的检测
                Roll = 0
                Rolleye = 0
                Rollmouth = 0
                fenxi=0


            shape = orig.shape[0:2]

            frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(256,256), cv2.INTER_LINEAR )
            
            
            k = cv2.waitKey(1)
            if k != -1:
                self.terminate(camera)
                break

            a = img_transform(Image.fromarray(frame))
            a = a.unsqueeze(0)
            imgs = Variable(a.to(dtype=torch.float, device=self.device))
            pred = self.model(imgs)
            
            pred= torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])
            mask = pred.data.cpu().numpy()
            mask = mask.squeeze()
            
            # im = Image.fromarray(mask)
            # im2 = im.filter(ImageFilter.MinFilter(3))
            # im3 = im2.filter(ImageFilter.MaxFilter(5))
            # mask = np.array(im3)
            
            mask = mask > 0.8
            orig[mask==0]=0
            self.pipe.send([orig])

            # if self.show_mask:
            #     cv2.imshow('mask', orig)
            
            if self.frames_count % 30 == 29:
                time_2 = time.time()
                sys.stdout.write(f'\rFPS: {30/(time_2-time_1)}')
                sys.stdout.flush()
                time_1 = time.time()


            self.frames_count+=1

        self.terminate(camera)

    
    def terminate(self, camera):
        self.pipe.send(None)
        cv2.destroyAllWindows()
        camera.release()
        



