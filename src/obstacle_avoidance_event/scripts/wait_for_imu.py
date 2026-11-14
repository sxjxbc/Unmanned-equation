#!/usr/bin/env python
"""Wait for the original IMU device; run only its existing read-only driver."""
from __future__ import print_function
import os,time,subprocess,signal


if __name__=='__main__':
    state=dict(stop=False,child=None)
    def shutdown(signum,frame):
        state['stop']=True
        child=state['child']
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGINT)
            except OSError:pass
    signal.signal(signal.SIGINT,shutdown);signal.signal(signal.SIGTERM,shutdown)
    while not state['stop']:
        if not os.path.exists('/dev/imu'):
            print('Waiting for original /dev/imu; no guessed serial port');time.sleep(2.);continue
        child=subprocess.Popen(['rosrun','imu','daoyuan_simple.py'],preexec_fn=os.setsid)
        state['child']=child
        while child.poll() is None and not state['stop']:time.sleep(.2)
        if state['stop']:
            deadline=time.time()+5.
            while child.poll() is None and time.time()<deadline:time.sleep(.1)
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            break
        state['child']=None;time.sleep(2.)
