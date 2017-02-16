#!/usr/bin/env python
from __future__ import print_function

import cv2
import sys
from service import Tick,getCameraVideoSize,CameraThread 

if __name__ == "__main__":
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("can't open camera")
        exit()
        
    while True:
        try:
            _,im = cap.read()
            cv2.imshow("preview",im)
            img = cv2.cvtColor(im,cv2.cv.CV_BGRA2GRAY)
            v = cv2.mean(img)
            print('\r','%.2f'%sum(v),end='')
            sys.stdout.flush()
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break
                
        except KeyboardInterrupt:
            break
        
    
    cv2.destroyAllWindows()
