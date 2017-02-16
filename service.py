#!/usr/bin/env python
'''
Created on 2017/2/8

@author: mark
'''
from __future__ import print_function
import cv2
import threading
import time
import sys
import argparse
import Image
import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import base64
from StringIO import StringIO
#import json
#import numpy as np
 

history_size = 30 * 10
background_scale = 2
opticalflow_scale = 4
stability_max = 40.0
stability_min = 0.3
video_width = 640
video_height = 480
camera_devid = 0
socket_port = 8888

videocapture = None
thread_camera = None
thread_background = None
thread_opticalflow = None

def getCameraVideoSize(videocapture):
    width = int(videocapture.get(3))
    height = int(videocapture.get(4))
    if width < 0 or height < 0:   
        _,im = videocapture.read();
        VideoSize = (im.shape[1],im.shape[0])
    else:
        VideoSize = (width,height)
    return VideoSize
    
class Tick:
    def __init__(self):
        self._tick = cv2.getTickCount()
    def msec(self):
        diff = cv2.getTickCount()-self._tick;
        return (diff/cv2.getTickFrequency())*1000
    def reset(self):
        self._tick = cv2.getTickCount()

class MyBasicThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.isTerminal = False
        self._frameNum = 0
        self._tick = Tick()
    
    def run(self):        
        while self.isTerminal is False:
            self.once()
    
    def once(self):
        raise NotImplementedError()
    
    def Terminal(self):
        self.isTerminal = True
    
    def is_terminal(self):
        return self.isTerminal
    
    def _incNum(self):
        self._frameNum = self._frameNum+1;
        
    def msec(self):
        if self._frameNum == 0:
            return -1
        return self._tick.msec()/self._frameNum
    
    def frameNum(self):
        return self._frameNum
    
    def frameNumReset(self):
        self._frameNum = 0
        self._tick.reset()
        
class CameraThread(MyBasicThread):
    def __init__(self,videocapture):
        MyBasicThread.__init__(self)
        self.cap = videocapture        
        self.setName('CameraThread')
        self._previewImage = None
        self._frameNum = 0
        self._image_lock = threading.Lock()
        self.ready_event = threading.Event()
         
      
    def once(self):
        _,im = self.cap.read()
        self._image_lock.acquire()
        self._previewImage = im.copy()
        self._image_lock.release()        
        self.ready_event.set()
        self._incNum()
    
            
    def get(self):
        if self._previewImage is None:
            return None
        self._image_lock.acquire()
        image = self._previewImage.copy()
        self._image_lock.release()
        return image
class OpticalFlowThread(MyBasicThread):
    def __init__(self,cap):
        MyBasicThread.__init__(self)
        self.cap = cap;
        self._tick_250ms = Tick()
        self.im_pre = None
        self.im_next = None
        self.activity = -1
        self._resize = (video_width/opticalflow_scale,video_height/opticalflow_scale)
        print(self._resize)
    
    def once(self):
        msec = self._tick_250ms.msec()
        if msec < 250:
            time.sleep(float(250-msec)/1000)
        self._tick_250ms.reset()
        im = self.cap.get()
        if im is None:
            return
        ims = cv2.resize(im,self._resize)
        im_g = cv2.cvtColor(ims,cv2.cv.CV_BGRA2GRAY)
        self.im_pre = self.im_next
        self.im_next =cv2.GaussianBlur(im_g,(3,3),0)
        if self.im_pre is None:
            return;
        flow = cv2.calcOpticalFlowFarneback(self.im_pre,self.im_next,0.5,3,15,3,5,1.2,0)
        x,y = cv2.split(abs(flow))
        v = sum(cv2.sumElems(x))/cv2.countNonZero(x)
        v+= sum(cv2.sumElems(y))/cv2.countNonZero(y)
        v *= 100.0
        if self.activity < 0:
            self.activity = v
        else:
            self.activity = (self.activity*5.0+v)/6.0
        self._incNum()        
        
class BackGroundThread(MyBasicThread):
    STATE_STOP = 'stop'
    STATE_START = 'start'
    STATE_READY = 'ready'
    STATE_EXCEPTION = 'exception'   
    
    def __init__(self,source):
        MyBasicThread.__init__(self)
        self.setName('BackGround Subtractor')
        self._lock = threading.Lock()
        self._video_source = source;
        self.bgs = cv2.BackgroundSubtractorMOG2(history=history_size,varThreshold=9)
        self._scale_size = (video_width/background_scale,video_height/background_scale)
        self.kernel3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        
        self.run_status = BackGroundThread.STATE_START
        self.stability=0
        self._lock = threading.Lock()
        self.ready = threading.Event()        
    
      
    def run(self):
        tick = Tick()
        target_value = self.frameNum() + history_size/2;
        while self.frameNum() < target_value:
            self.once()
            if self.is_terminal():
                return
        self.run_status = self.STATE_READY
        while self.is_terminal() is False:
            self.once()
            if self.run_status is self.STATE_EXCEPTION:
                time.sleep(0.1)
            elif self.run_status is self.STATE_START:
                tick.reset()
                self.frameNumReset()
                print(' begin re-entry start state:')
                while self.frameNum() < history_size:
                    self.once()
                    if self.stability > stability_min:
                        self.frameNumReset()
                    print('\r %d' % (self.frameNum()*100/history_size),end='')
                    sys.stdout.flush()
                    if self.is_terminal():
                        break;
                self.run_status = self.STATE_READY

   
    def restart(self):
        self.run_status = self.STATE_START
             
    def once(self):
        if self._video_source.ready_event.wait(0.5) is False:
            return       
        image = self._video_source.get()
        self._video_source.ready_event.clear()        
        
        learning = -1
        
        if self.run_status == BackGroundThread.STATE_READY:
            if self.stability > stability_min and self.stability < stability_max:
                learning = 1.0/(history_size*100)      
        
        self.stability, mask = self._applyBG(image,learning)
        
        if self.stability > stability_max:
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

class WSCameraHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True
    def initialize(self):
        pass
            
    def open(self):
        pass
        #print ('New connection was opened')
        #self.write_message(self.str)

    def on_message(self, message):
        if message == 'camera':
            self.on_camera()
        elif message == 'fg':
            self.on_fg()
        elif message == 'status':
            self.on_status()
        elif message == 'start':
            self.on_start()
        elif message == 'stop':
            self.on_stop()       

    def on_stop(self):
        global videocapture,thread_background,thread_camera,thread_opticalflow
        if thread_opticalflow is not None:
            thread_opticalflow.Terminal()
            thread_opticalflow.join()
            thread_opticalflow = None        
        if thread_background is not None:
            thread_background.Terminal()
            thread_background.join()
            thread_background = None
        if thread_camera is not None:
            thread_camera.Terminal()
            thread_camera.join()
            thread_camera = None
        if videocapture is not None:
            videocapture.release()
            videocapture = None
        self.write_message('{"ret":true}')
                                
    def on_start(self):
        global videocapture,thread_background,thread_camera,thread_opticalflow
        if videocapture is None:
            videocapture = cv2.VideoCapture(camera_devid)
            if not videocapture.isOpened:
                self.write_message('{"ret":false,"reason":"Camera Fail"}')
                videocapture.release()
                videocapture = None
                                
        if thread_camera is None:
            thread_camera = CameraThread(videocapture)
            thread_camera.start();
        
        if thread_opticalflow is None:
            thread_opticalflow = OpticalFlowThread(thread_camera)
            thread_opticalflow.start()
            
        if thread_background is None:
            thread_background = BackGroundThread(thread_camera)
            thread_background.start()        
        else:
            if thread_background.run_status == thread_background.STATE_START:
                self.write_message('{"ret":true}')
            else:
                print("restart...")            
                thread_background.restart()
        self.write_message('{"ret":true}')

        
    def on_status(self):
        if thread_background == None:
            msg = '{"ret":true,"status":"%s"}' % BackGroundThread.STATE_STOP
            self.write_message(msg)
        else:
            msg = '{"ret":true,"status":"%s"}' % thread_background.run_status
        self.write_message(msg)        
    
    def on_fg(self):
        if thread_background == None:
            msg = '{"ret":false,"reason":"%s"}' % BackGroundThread.STATE_STOP
            self.write_message(msg)
            return
        if thread_background.run_status != BackGroundThread.STATE_READY:
            msg = '{"ret":false,"reason":"%s"}' % thread_background.run_status
            self.write_message(msg)
            return
        mask = thread_background.get()
        fgmask = cv2.resize(mask,(video_width,video_height))
        image = thread_camera.get()
        result = cv2.bitwise_and(image,image,mask=fgmask)
        b_channel, g_channel, r_channel = cv2.split(result)                
        img_RGBA = cv2.merge((r_channel, g_channel, b_channel, fgmask))
        pil_im = Image.fromarray(img_RGBA)
        iobuffer = StringIO()
        pil_im.save(iobuffer, format="PNG")
        img_str = base64.b64encode(iobuffer.getvalue())
        v = thread_opticalflow.activity
        if v > 100:
            v = 100
        if thread_background.stability < stability_min:
            v = 0
        
        msg = '{"ret":true,"activity":%d,"img":"data:image/png;base64,%s"}' %(v,img_str)
        self.write_message(msg)
    
    def cvimageToBase64JpegString(self,im):
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(im)
        iobuffer = StringIO()
        pil_im.save(iobuffer, format="JPEG")
        img_str = base64.b64encode(iobuffer.getvalue())
        return img_str
                    
    def on_camera(self):
        if thread_camera == None:
            msg = '{"ret":false,"reason":"%s"}' % BackGroundThread.STATE_STOP
            self.write_message(msg)
            return
        im = thread_camera.get()
        img_str = self.cvimageToBase64JpegString(im)
        msg = '{"ret":true,"stability":%d,"img":"data:image/jpeg;base64,%s"}' % (int(round(thread_background.stability)),img_str) 
        self.write_message(msg)
    def on_close(self):
        pass
        #print ('Connection was closed...')
    
application = tornado.web.Application([
  (r'/ws', WSCameraHandler),

])

class WSThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        http_server = tornado.httpserver.HTTPServer(application)
        http_server.listen(socket_port)    
        tornado.ioloop.IOLoop.instance().start()
                
def ParserArgs():
    p = argparse.ArgumentParser(description='background subtractor')
    p.add_argument('-d','--dev' ,type = int,nargs='?',metavar='devid',dest='devid' , default=0)
    p.add_argument('-r','--review',action="store_true",default = False)
    p.add_argument('-m','--mask',action="store_true",default = False)
    p.add_argument('-o','--output',action='store_true',default=False)
    p.add_argument('--history',type=int,nargs='?',default=30*10)
    p.add_argument('--background_scale',type=int,nargs='?',default='2')
    p.add_argument('--opticalflow_scale',type=int,nargs='?',default='4')
    p.add_argument('--stability_max',type=float, nargs='?',default='40')
    p.add_argument('--stability_min',type=float, nargs='?',default='0.3')
    p.add_argument('--stocket_port',type=int,nargs='?',default='8888')
    p.add_argument('--autostart',action='store_true',default=False)
    return p.parse_args()
    
if __name__ == '__main__':

    args = ParserArgs()

    camera_devid = args.devid
    videocapture = cv2.VideoCapture(camera_devid)
    history_size = args.history
    background_scale = args.background_scale
    opticalflow_scale = args.opticalflow_scale
    stability_max = args.stability_max
    stability_min = args.stability_min
    socket_port = args.stocket_port
    
    if not videocapture.isOpened():
        print('Can\'t open thread_camera')
        exit()
        
    videoSize = getCameraVideoSize(videocapture)
    video_width = videoSize[0]
    video_height = videoSize[1]
    
    if args.review or args.mask or args.output or args.autostart:
        thread_camera = CameraThread(videocapture)
        thread_camera.start();
        number = thread_camera.frameNum()
        thread_background = BackGroundThread(thread_camera)
        thread_background.start()
        thread_opticalflow = OpticalFlowThread(thread_camera)
        thread_opticalflow.start()
        subNum = thread_background.frameNum()
        prestate = thread_background.run_status;
    else:
        number = 0
        subNum = 0
        prestate = BackGroundThread.STATE_STOP
        
    wsThread = WSThread()
    wsThread.start()
     
    image = None
    mask = None
    loop = True
    while loop:
        try:
            key = cv2.waitKey(1) & 0xff
            if key == 27: # ESC
                loop = False
            if thread_background is not None and (key == 83 or key == 115) : # 's' or 'S'                    
                thread_background.restart()

            if thread_background == None:
                time.sleep(0.1)
                continue
                   
            if args.review:
                t = thread_camera.frameNum()
                if number != t:
                    number = t            
                    image = thread_camera.get()
                    if image is not None:
                        cv2.imshow('Preview',image)
            
            if (args.mask or args.output) and thread_background.ready.is_set(): 
                mask = thread_background.get()
                thread_background.ready.clear()
                fgmask = cv2.resize(mask,(video_width,video_height))
                if args.mask:
                    cv2.imshow('mask',mask)
                if not args.review:
                    image = thread_camera.get()
                result = cv2.bitwise_and(image,image,mask=fgmask)
                b_channel, g_channel, r_channel = cv2.split(result)                
                img_RGBA = cv2.merge((b_channel, g_channel, r_channel, fgmask))

                cv2.imshow('Result',img_RGBA)
            if thread_camera is not None:
                print('\r','%.2f' % (1000.0/thread_camera.msec()), '%.2f'%(1000/thread_background.msec()),end=' ')            
                v = thread_opticalflow.activity
                print('%.2f' % (v),end=' ')            
                sys.stdout.flush()
            if prestate != thread_background.run_status:
                prestate = thread_background.run_status
                print('\n state change',prestate)
            
        except KeyboardInterrupt:
            loop = False
            
    
    tornado.ioloop.IOLoop.instance().stop()
    
    if thread_background is not None: 
        thread_background.Terminal();
        thread_background.join()
        print('sub thread terminal')
    
    if thread_opticalflow is not None:
        thread_opticalflow.Terminal()
        thread_opticalflow.join()
        print('Optical Flow thread terminal')
    
    if thread_camera is not None:
        thread_camera.Terminal()
        thread_camera.join()
        print('thread_camera thread terminal')
    
    wsThread.join()
    print('webstocket thread terminal')    
    
