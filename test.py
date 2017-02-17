#!/usr/bin/env python
from __future__ import print_function

import cv2
import sys


video_devID = 1

from service import Tick,getCameraVideoSize,CameraThread 

def _FindandSetId(control,ID,value,id_name,preset = None):
    if preset is None:
        preset = control.get_controls()
    matches = [x for x in preset if x['id'] == ID]
    if len(matches) == 0:
        print("No support %s" % id_name)
        return False
    else:
        control.set_control_value(ID,value)
        return True
    
    
def initialCamera(cap,devID = 0):
    import pyv4l2
    # I don't know why first time import will fail 
    # and second import can success    #  
    try:
        import pyv4l2.control 
    except ImportError:
        import pyv4l2.control
    
    from pyv4l2.control import Control
    control = Control('/dev/video%d'%devID)
    preset = control.get_controls()
    _FindandSetId(control,9963788,1,'Auto White Balance Control',preset)
    _FindandSetId(control, 10094849, 3, 'Exposure, Auto',preset)
    _FindandSetId(control,10094851,1,'Exposure, Auto Priority',preset)
    tick = Tick()
    while tick.msec() < 1500:
        cap.read()
    
    preset = control.get_controls()
    _FindandSetId(control,9963788,0,'Auto White Balance Control',preset)
    _FindandSetId(control, 10094849, 1, 'Exposure, Auto',preset)
    _FindandSetId(control,10094851,0,'Exposure, Auto Priority',preset)     
    print('initial camera finish')
    

if __name__ == "__main__":
        
    cap = cv2.VideoCapture(video_devID)
    if not cap.isOpened():
        print("can't open camera")
        exit()
    cap.set(3,320)
    cap.set(4,240)
    initialCamera(cap,video_devID)    
    while True:
        try:
            _,im = cap.read()
            cv2.imshow("preview",im)
            img = cv2.cvtColor(im,cv2.cv.CV_BGRA2GRAY)
            v = cv2.mean(img)
            #print('\r','%.2f'%sum(v),end='')
            sys.stdout.flush()
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break
                
        except KeyboardInterrupt:
            break
        
    
    cv2.destroyAllWindows()
