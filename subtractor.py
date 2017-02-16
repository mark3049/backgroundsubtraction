#!/usr/bin/env python
'''
Created on 2017/2/6

@author: mark
'''
from __future__ import print_function
import cv2
import argparse
#import numpy as np
import sys
from service import Tick



g_config_history = 2*60*4
PreviewWinName = None



def ReInitSubtractor(videocapture,subtractor,timeout=5):
    global PreviewWinName
    tick = Tick();
    print('re-initial video',end='')
    outformat = 'wait a minute (%d)'
    
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fsize, baseline = cv2.getTextSize(outformat % timeout,fontFace,1,4)
    baseline += 4
    
    while tick.msec()<timeout*1000:
        _,im = videocapture.read()
        if subtractor.reinit(im) is False:
            tick.reset()                
        if PreviewWinName is not None:
            width = im.shape[1]
            height = im.shape[0]
            
            text = outformat % int(timeout-tick.msec()/1000+1)
            textOrg = (width/2-fsize[0]/2,height/2-(fsize[1])/2)

            lefttop = (textOrg[0]-5,textOrg[1]-baseline-15)
            textEnd = (lefttop[0]+fsize[0]+10,lefttop[1]+fsize[1]+20)
            cv2.rectangle(im,lefttop,textEnd,(255,255,255),thickness=cv2.cv.CV_FILLED)
            cv2.putText(im,text,textOrg,fontFace,1,(0,0,0),1)
            cv2.imshow(PreviewWinName,im)
        if cv2.waitKey(1) & 0xff == 27:
            break;
    print('')
    _,im = videocapture.read()
    subtractor.apply(im)
    return

class Subtractor:
    def __init__(self,VideoSize,history = 2*60*4,scaleDown = 2):
        self.scale_size = (VideoSize[0]/scaleDown,VideoSize[1]/scaleDown)
        self.orgSize = VideoSize
        #self.mask = np.zeros((scale_size[0]/2,scale_size[1]/2,1),dtype = 'uint8')
        self.bgs = cv2.BackgroundSubtractorMOG2(history=history,varThreshold=9) #  60 sec * 4fps
        self.fgmask = None
        self.learning = True
        self.kernel3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.stoplearning_timeout = Tick();
        self.isDrawMask = False        

    def _applyBG(self,image,learning = -1):
        im = cv2.resize(image,self.scale_size)
        gb = cv2.GaussianBlur(im,(3,3),0)
        mask = self.bgs.apply(gb,learningRate = learning)
        _,mask = cv2.threshold(mask,128.0,255.0,cv2.THRESH_BINARY)
        #self.fgmask = cv2.erode(self.fgmask,self.kernel3x3,iterations=2)
        #self.fgmask = cv2.dilate(self.fgmask,self.kernel3x3,iterations=2)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,self.kernel3x3,iterations=2)
        
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(mask,contours,-1,(255,255,255),cv2.cv.CV_FILLED)        
        return mask;
    
    def _NonZeroRate(self,mask):
        count = float(cv2.countNonZero(mask));
        size = mask.shape[0]*mask.shape[1]
        count/= size
        count *= 100;
        return count;
         
    def reinit(self,image):
        mask = self._applyBG(image)
        count = self._NonZeroRate(mask)
        if count < 0.3: 
            return True
        else:
            return False
                
            
    def apply(self,image):
        if self.learning:
            mask = self._applyBG(image)
        else:
            mask = self._applyBG(image, 0)

        count = self._NonZeroRate(mask)
                
        if count > 40:
            return False
        elif count > 0.3:
            self.learning = False
            self.stoplearning_timeout.reset()
        else:
            if self.stoplearning_timeout.msec()>1000:
                self.learning = True

        self.fgmask = cv2.resize(mask,self.orgSize)

        if self.isDrawMask:
            self._drawMask(count,mask);
        return True
    
    def width(self):
        return self.scale_size[0]
    
    def height(self):
        return self.scale_size[1]
    
    def _drawMask(self,count,mask):
        img = mask;
        font = cv2.FONT_HERSHEY_PLAIN 
        if self.learning:
            buf = 'On %.2f' % count
        else:
            buf = 'Off %.2f' % count
            
        cv2.putText(img,buf,(10,40), font, 1,(127,127,127),4)
        cv2.putText(img,buf,(10,40), font, 1,(255,255,255),2)
        cv2.imshow("fgmask",img)


def ParserArgs():
    p = argparse.ArgumentParser(description='background subtractor')
    p.add_argument('-d','--dev' ,type = int,nargs='?',metavar='devid',dest='devid' , default=0)
    p.add_argument('-i','--ipcam',nargs='?',metavar='ip address',dest='ipcam_url')
    p.add_argument('-u','--user',nargs='?',metavar='ipcam username',dest='ipcam_name',default='admin')
    p.add_argument('-p','--passwd',nargs='?',metavar='ipcam passwd',dest='ipcam_pass',default='123456')
    p.add_argument('-r','--review',action="store_true",default = False)
    p.add_argument('-m','--mask',action="store_true",default = False)
    p.add_argument('-o','--output',action='store_false',default=True)
    p.add_argument('--history',type=int,nargs='?',default='480')
    p.add_argument('-s','--scaleDown',type=int,nargs='?',metavar='scale down',dest='scaleDown',default='2')
    return p.parse_args()

    
if __name__ == '__main__':    
    args = ParserArgs()
    
    if args.review:
        PreviewWinName = 'Camera Preview'
    g_config_history = args.history
    
        
    if args.ipcam_url:
        url = 'rtsp://%s:%s@%s:554/live2.sdp' % (args.ipcam_name,args.ipcam_pass,args.ipcam_url)
        videocapture = cv2.VideoCapture(url)
    elif args.devid >= 0:
        videocapture = cv2.VideoCapture(args.devid)
        
    if not videocapture.isOpened():
        print("Can't open Camera")
        exit()
    width = int(videocapture.get(3))
    height = int(videocapture.get(4))
    if width < 0 or height < 0:   
        _,im = videocapture.read();
        VideoSize = (im.shape[1],im.shape[0])
    else:
        VideoSize = (width,height)
        
    subtractor = Subtractor(VideoSize,args.history,args.scaleDown)
    if args.mask:
        subtractor.isDrawMask = True;
        
    ReInitSubtractor(videocapture,subtractor,1)
    tick = Tick()
    avg_msec = -1.0;
    while True:
        tick.reset()
        ret, im = videocapture.read()
        if PreviewWinName is not None:
            cv2.imshow(PreviewWinName,im)
            
        ret = subtractor.apply(im)
        if ret is False:
            ReInitSubtractor(videocapture,subtractor)
        else:
            reust = cv2.bitwise_and(im,im,mask=subtractor.fgmask)
            if args.output:
                cv2.imshow("result",reust)
        if avg_msec < 0:
            avg_msec = tick.msec()    
        else:
            avg_msec = (avg_msec*99+tick.msec())/100.0    
        print('\r %.2f %.2f' % (avg_msec,1000/avg_msec),end='')
        sys.stdout.flush()
        #if subtractor.fgmask is not None:
        #    DrawResult(im,subtractor.fgmask)        
                
        key = cv2.waitKey(1)
        if key < 0:
            continue
        
        key &= 0xff
        if key == 27: # esc
            break;
        if key == 81 or key == 113: # Q or q
            break;
        if key == 82 or key == 114 : # R or r
            ReInitSubtractor(videocapture,subtractor)
        
        print(key)
        
