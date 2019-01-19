#coding=utf-8
import cv2
import time
from PreProcessor import *
from DarknetDetect import *
class videoPlayerDetector:
    def __init__(self):
        self.__preProcessor=preProcessor()
        self.__darknetDetect=darknetDetector()
    def startDetect(self,videoName,frameOperateFunc):
        self.__videoName=videoName
        self.__frameOperateFunc=frameOperateFunc
        T0=0
        T1=0
        T2=0
        T3=0
        Fcount=0
        vidcap = cv2.VideoCapture(self.__videoName)
        success,image = vidcap.read()

        StopFlag=False
        while success:
            Fcount+=1
            #preprocess
            T0=time.time()
            processed=self.__preProcessor.process(image)
            T1 +=time.time()-T0

            #playerDetect
            T0=time.time()
            boxes=self.__darknetDetect.detectImage(processed)

            T2 +=time.time()-T0


            #display
            T0=time.time()
            if(self.__frameOperateFunc!= None):
                StopFlag=self.__frameOperateFunc(image,boxes)
            T3 +=time.time()-T0
            print T1*1.0/Fcount,T2*1.0/Fcount,T3*1.0/Fcount

            success,image = vidcap.read()
            if(StopFlag==True):
                success=False

        vidcap.release()
        cv2.destroyAllWindows()



if __name__== "__main__":
    V=videoPlayerDetector()
    V.test()
