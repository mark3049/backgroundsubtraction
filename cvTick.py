
import cv2

class Tick:
    def __init__(self):
        self._tick = cv2.getTickCount()
    def msec(self):
        diff = cv2.getTickCount()-self._tick;
        return (diff/cv2.getTickFrequency())*1000
    def reset(self):
        self._tick = cv2.getTickCount()

class period_tick:
    def __init__(self):
        self._msec = 0
        self._tick = Tick()
    def msec(self):
        return self._msec
    def start(self):
        self._tick.reset()
    def stop(self):
        self._msec += self._tick.msec()
    def reset(self):
        self.msec = 0       
            
