import time
import cv2

from BaseThread import MyBasicThread
from cvTick import Tick
import config
class OpticalFlowThread(MyBasicThread):
    def __init__(self,cap):
        MyBasicThread.__init__(self)
        self.args = config.get()
        
        self.cap = cap;
        self._tick_250ms = Tick()
        self.im_pre = None
        self.im_next = None
        self.activity = -1
        scale = self.args.opticalflow_scale
        self._resize = (self.args.video_width/scale,self.args.video_height/scale)
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
        