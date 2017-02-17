from __future__ import print_function
import cv2
import threading
import time
import sys

import config
from BaseThread import MyBasicThread
from cvTick import Tick

import logging
log = logging.getLogger(__name__)

class BackGroundThread(MyBasicThread):
    STATE_STOP = 'stop'
    STATE_START = 'start'
    STATE_READY = 'ready'
    STATE_EXCEPTION = 'exception'   
    
    def __init__(self,source):
        MyBasicThread.__init__(self)
        self.args = config.get()
        self.setName('BackGround Subtractor')
        self.history_size = self.args.history  

        self._lock = threading.Lock()
        self._video_source = source;
        self.bgs = cv2.BackgroundSubtractorMOG2(
                                                history=self.history_size,
                                                varThreshold=self.args.background_varThreshold)
        scale = self.args.background_scale
        self._scale_size = (self.args.video_width/scale,self.args.video_height/scale)
        self.kernel3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        
        self.run_status = BackGroundThread.STATE_START
        self.stability=0
        self._lock = threading.Lock()
        self.ready = threading.Event()
    
      
    def run(self):
        tick = Tick()
        
        log.info('Begin BackgroundSubtractorMOG2 first step - %d frame' % self.history_size/2)        
        target_value = self.frameNum() + self.history_size/2;
        while self.frameNum() < target_value:
            self.once()
            if self.is_terminal():
                return
        self.run_status = self.STATE_READY
        log.info('Statue change to [%s]' % self.run_status)
        while self.is_terminal() is False:
            self.once()
            if self.run_status is self.STATE_EXCEPTION:
                time.sleep(0.1)
            elif self.run_status is self.STATE_START:
                tick.reset()
                self.frameNumReset()
                log.info(' begin re-entry start state:')
                while self.frameNum() < self.history_size:
                    self.once()
                    if self.stability > self.args.stability_min:
                        self.frameNumReset()
                    #print('\r %d' % (self.frameNum()*100/self.history_size),end='')
                    #sys.stdout.flush()
                    if self.is_terminal():
                        break;
                self.run_status = self.STATE_READY
                log.info('status change to [%s]'%self.run_status)

   
    def restart(self):
        self.run_status = self.STATE_START
             
    def once(self):
        if self._video_source.ready_event.wait(0.5) is False:
            return       
        image = self._video_source.get()
        self._video_source.ready_event.clear()        
        
        learning = -1
        
        if self.run_status == BackGroundThread.STATE_READY:
            if self.stability > self.args.stability_min and self.stability < self.args.stability_max:
                learning = 1.0/(self.history_size*200)      
        
        self.stability, mask = self._applyBG(image,learning)
        
        if self.stability > self.args.stability_max:
            self.run_status = BackGroundThread.STATE_EXCEPTION
        
        if self.run_status == BackGroundThread.STATE_READY:
            self._lock.acquire()
            self.fgmask = mask.copy() #cv2.resize(mask,(video_width,video_height))
            self._lock.release()
            self.ready.set()
            
        self._incNum()
    
    def get(self):
        if self.fgmask is None:
            return None
        self._lock.acquire()
        mask = self.fgmask.copy()
        self._lock.release()
        return mask
        

    def _applyBG(self,image,learning = -1):
        im = cv2.resize(image,self._scale_size)
        gb = cv2.GaussianBlur(im,(3,3),0)
        mask = self.bgs.apply(gb,learningRate = learning)
        _,mask = cv2.threshold(mask,128.0,255.0,cv2.THRESH_BINARY)
        #self.fgmask = cv2.erode(self.fgmask,self.kernel3x3,iterations=2)
        #self.fgmask = cv2.dilate(self.fgmask,self.kernel3x3,iterations=2)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,self.kernel3x3,iterations=2)
        
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(mask,contours,-1,(255,255,255),cv2.cv.CV_FILLED)
        count = float(cv2.countNonZero(mask));
        size = mask.shape[0]*mask.shape[1]
        count/= size
        count *= 100;                
        return count,mask; 