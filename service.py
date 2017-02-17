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
import platform
from StringIO import StringIO
#import json
#import numpy as np
import config 

from background import BackGroundThread
from webcam import CameraThread
from activity import OpticalFlowThread

videocapture = None
thread_camera = None
thread_background = None
thread_opticalflow = None
from BaseThread import MyBasicThread
from cvTick import Tick

def _initial_system():
    global videocapture,thread_background,thread_camera,thread_opticalflow
    args = config.get()
    if videocapture is None:
        videocapture = cv2.VideoCapture(args.devid)
        if not videocapture.isOpened:
            videocapture.release()
            videocapture = None
            return False,'{"ret":false,"reason":"Camera Fail"}'
                            
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
            #return '{"ret":true}'
            pass
        else:
            print("restart...")            
            thread_background.restart()
    
    return True,'{"ret":true}'
                    
def _stop_system():
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
        _stop_system()
        self.write_message('{"ret":true}')
                                
    def on_start(self):
        msg = _initial_system()
        self.write_message(msg)

        
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
        image = thread_camera.get()
        mask = thread_background.get()
        timestamp = thread_background.timestamp()
        fgmask = cv2.resize(mask,(image.shape[1],image.shape[0]))
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
        if thread_background.stability < config.get().stability_min:
            v = 0
        
        msg = '{"ret":true,"activity":%d,"timstamp":%d,"img":"data:image/png;base64,%s"}' %(v,timestamp,img_str)
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
        timestamp = thread_camera.timestamp()
        stability = int(round(thread_background.stability))
        img_str = self.cvimageToBase64JpegString(im)
        msg = '{"ret":true,"stability":%d,"img":"data:image/jpeg;base64,%s"}' % (stability,timestamp,img_str) 
        self.write_message(msg)
    def on_close(self):
        pass
        #print ('Connection was closed...')
    
application = tornado.web.Application([
  (r'/ws', WSCameraHandler),

])

class WSThread(threading.Thread):
    def __init__(self,port):
        threading.Thread.__init__(self)
        self.port = port
    def run(self):
        http_server = tornado.httpserver.HTTPServer(application)
        http_server.listen(self.port)    
        tornado.ioloop.IOLoop.instance().start()
                
    
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    args = config.get()
    
    videocapture = cv2.VideoCapture(args.devid)
    
    if not videocapture.isOpened():
        print('Can\'t open thread_camera')
        exit()        
    
    if args.review or args.mask or args.output or args.autostart:
        _initial_system()
        number = thread_camera.frameNum()
        subNum = thread_background.frameNum()
        prestate = thread_background.run_status;
    else:
        number = 0
        subNum = 0
        prestate = BackGroundThread.STATE_STOP
        
    wsThread = WSThread(args.stocket_port)
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
                if args.mask:
                    cv2.imshow('mask',mask)
                if not args.review:
                    image = thread_camera.get()
                fgmask = cv2.resize(mask,(image.shape[1],image.shape[0]))
                result = cv2.bitwise_and(image,image,mask=fgmask)
                b_channel, g_channel, r_channel = cv2.split(result)                
                img_RGBA = cv2.merge((b_channel, g_channel, r_channel, fgmask))

                cv2.imshow('Result',img_RGBA)
            if thread_camera is not None:
                cfps = int(round((1000.0/thread_camera.msec())))
                bgfps = int(round((1000/thread_background.msec())))                           
                print('\r','%d fps' % cfps,'%d fps' % bgfps,end=' ')            
                v = thread_opticalflow.activity
                print('%d msec' % int(round(thread_opticalflow.msec())),end=' ')
                print('%.1f' % (v),end=' ')            
                sys.stdout.flush()            
        except KeyboardInterrupt:
            loop = False
            
    
    tornado.ioloop.IOLoop.instance().stop()    
    wsThread.join()
    print('webstocket thread terminal')    
    _stop_system()   
    
