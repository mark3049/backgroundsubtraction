
import cv2

class Tick:
    def __init__(self):
        self._tick = cv2.getTickCount()
    def msec(self):
        diff = cv2.getTickCount()-self._tick;
        return (diff/cv2.getTickFrequency())*1000
    def reset(self):
        self._tick = cv2.getTickCount()
