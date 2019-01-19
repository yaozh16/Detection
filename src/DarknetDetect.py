#coding=utf-8

import darknet as dn
import cv2

class darknetDetector:
    def __init__(self):

        dn.set_gpu(0)
        self._net = dn.load_net("cfg/yolov3.cfg", "weights/yolov3.weights", 0)
        #self._net = dn.load_net("cfg/yolov3.cfg", "weights/yolov3.weights", 0)
        self._meta = dn.load_meta("cfg/coco.data")
        self.__tmpFile="tmp/tmp.jpg"

    def __array_to_image(self,arr):
        arr = arr.transpose(2,0,1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = (arr/255.0).flatten()
        data = dn.c_array(dn.c_float, arr)
        im = dn.IMAGE(w,h,c,data)
        return im
    def detectImage(self,image):
        cv2.imwrite(self.__tmpFile,image)
        return dn.detect(self._net,self._meta,self.__tmpFile,0.8)

