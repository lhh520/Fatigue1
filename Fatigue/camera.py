#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
    
class Camera(object):
    """ 通过opencv读取摄像头"""

    def __init__(self):
        u=0
        #self.cap = cv2.VideoCapture(0)
        #self.frames = get_frames2()

    def __del__(self):
        u=1
        #self.cap.close()
        
    def get_frame(flag1,frame):
        # eTOTAL=20
        # flag, frame = self.cap.read()
        # cv2.putText(frame, 'Blinks:{}'.format(eTOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        assert flag1
        flag, jpg = cv2.imencode('.jpg', frame)
        assert flag1
        return np.array(jpg).tostring()
