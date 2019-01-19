#coding=utf-8

import cv2
from VideoPlayerDetector import *


if __name__ == "__main__":
    def onFrame(displayImage,boxes):
        for each in boxes:
            rg=each[2]
            cv2.rectangle(displayImage,(int(rg[0]-rg[2]/2),int(rg[1]-rg[3]/2)),(int(rg[0]+rg[2]/2),int(rg[1]+rg[3]/2)),(0,255,0),1)
        cv2.imshow('Match Detection', displayImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False

    VPD=videoPlayerDetector()
    VPD.startDetect("../cutvideo.mp4",onFrame)




