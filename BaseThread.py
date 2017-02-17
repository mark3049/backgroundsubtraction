import threading
from cvTick import Tick

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
        