import threading
import logging
import cv2

import config 
from BaseThread import MyBasicThread
from cvTick import Tick

log = logging.getLogger(__name__)

def _FindandSetId(control,ID,value,id_name,preset = None):
    if preset is None:
        preset = control.get_controls()
    matches = [x for x in preset if x['id'] == ID]
    if len(matches) == 0:
        log.error("No support %s" % id_name)
        return False
    else:
        control.set_control_value(ID,value)
        return True

def getCameraVideoSize(videocapture):
    width = int(videocapture.get(3))
    height = int(videocapture.get(4))
    if width <=0 or height <= 0:
        log.error('videocapture.get return error value',width,height)
        _,im = videocapture.read()
        if im is None:
            log.error('videocapture read false')
            exit()
        width = im.shape[1]
        height = im.shape[0]
    return width,height
def setCameraVideoSize(videocapture,width,height):
    videocapture.set(3,width)
    videocapture.set(4,height)
    
def initialCamera(cap,args):
    log.debug('initialCamera begin')
    import pyv4l2
    # I don't know why first time import will fail 
    # and second import can success    #  
    try:
        import pyv4l2.control 
    except ImportError:
        import pyv4l2.control
    
    from pyv4l2.control import Control    

    control = Control('/dev/video%d'% args.devid)
    preset = control.get_controls()
    _FindandSetId(control,9963788,1,'Auto White Balance Control',preset)
    _FindandSetId(control, 10094849, 3, 'Exposure, Auto',preset)
    _FindandSetId(control,10094851,1,'Exposure, Auto Priority',preset)
    setCameraVideoSize(cap, args.video_width, args.video_height)
    cap.set(5,args.fps)
    tick = Tick()
    while tick.msec() < 1500:
        cap.read()
    
    args.video_width,args.video_height = getCameraVideoSize(cap)
    log.info('initialCamera to video size (%d,%d)' %(args.video_width,args.video_height))
    
    preset = control.get_controls()
    _FindandSetId(control,9963788,0,'Auto White Balance Control',preset)
    _FindandSetId(control, 10094849, 1, 'Exposure, Auto',preset)
    _FindandSetId(control,10094851,0,'Exposure, Auto Priority',preset)
    log.debug('initial camera finish')
    
class CameraThread(MyBasicThread):
    def __init__(self,videocapture):
        MyBasicThread.__init__(self)
        self.cap = videocapture
        self._args = config.get()        
        self.setName('CameraThread')
        self._previewImage = None
        self._frameNum = 0
        self._image_lock = threading.Lock()
        self.ready_event = threading.Event()
        initialCamera(videocapture,self._args)
        self.timestamp = cv2.getCPUTickCount()
        self._tick.reset() 
      
    def once(self):
        _,im = self.cap.read()
        self._image_lock.acquire()
        self._previewImage = im.copy()
        self.timestamp = cv2.getTickCount()
        self._image_lock.release()        
        self.ready_event.set()
        self._incNum()
    
    def width(self):
        self._args.video_width
    
    def heigh(self):
        self._args.video_height
    
    def size(self):    
        'video size (width,height)'
        return (self._args.video_width,self._args.video_height)
    
    def get(self):
        if self._previewImage is None:
            return None
        self._image_lock.acquire()
        image = self._previewImage.copy()
        self._image_lock.release()
        return image
    
    def timestamp(self):
        return self.timestamp