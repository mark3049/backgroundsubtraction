
import platform
import argparse
import logging

log = logging.getLogger(__name__)

_config = None
def _parserArgs():
    if platform.machine() == 'x86_64':
        history = 30*5
    else:
        history = 15*5
    p = argparse.ArgumentParser(description='background subtractor')
    p.add_argument('-d','--dev' ,type = int,nargs='?',metavar='devid',dest='devid' , default=0)
    p.add_argument('-r','--review',action="store_true",default = False)
    p.add_argument('-m','--mask',action="store_true",default = False)
    p.add_argument('-o','--output',action='store_true',default=False)
    p.add_argument('--history',type=int,nargs='?',default=history)
    p.add_argument('--background_scale',type=int,nargs='?',default='2')
    p.add_argument('--opticalflow_scale',type=int,nargs='?',default='4')
    p.add_argument('--stability_max',type=float, nargs='?',default='80')
    p.add_argument('--stability_min',type=float, nargs='?',default='1')
    p.add_argument('--stocket_port',type=int,nargs='?',default='8888')
    p.add_argument('--autostart',action='store_true',default=False)
    p.add_argument('-fps',type=int,nargs='?',default='0')
    p.add_argument('-width',type=int,nargs='?',dest='video_width',default='640')
    p.add_argument('-height',type=int,nargs='?',dest='video_height',default='480')
    p.add_argument('--background_varThreshold',type=int,nargs='?',default='16')
    return p.parse_args()
def get():
    global _config,log
    if _config is None:
        _config = _parserArgs()
        log.info('Current Configure=%s', _config)
        
    return _config

if __name__ == '__main__':
    pass