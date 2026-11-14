import math
from .model import distance, fresh

def pursuit(path, pose, vehicle, progress=0, lookahead=1.5):
    # Restricted progress window prevents global nearest-point branch jumps.
    stop = min(len(path),progress+12)
    index = min(range(progress,stop),key=lambda i:distance(path[i],pose[:2]))
    target = index
    length = 0.
    while target+1 < len(path) and length < lookahead:
        length += distance(path[target],path[target+1])
        target += 1
    dx,dy = path[target][0]-pose[0],path[target][1]-pose[1]
    x = math.cos(pose[2])*dx+math.sin(pose[2])*dy
    y = -math.sin(pose[2])*dx+math.cos(pose[2])*dy
    if x <= .05 or dx*dx+dy*dy < .01:
        raise ValueError('no_forward_target')
    delta = math.atan2(2*vehicle['wheelbase']*y,dx*dx+dy*dy)
    if abs(delta) > vehicle['max_steer']:
        raise ValueError('tracking_steer_limit')
    return delta,index

def speed_limit(path, ks, vehicle, measured_steer, steering, period=.05, check_response=True):
    caps = []
    for k in ks:
        caps.append(min(vehicle['speed_cap'], math.sqrt(vehicle['lateral_accel']/max(abs(k),1e-9))))
    caps[-1] = 0. # finite validated horizon must permit stopping
    for i in range(len(path)-2,-1,-1):
        ds = distance(path[i],path[i+1])
        caps[i] = min(caps[i], math.sqrt(caps[i+1]**2+2*vehicle['brake']*ds))
        if i+1 < len(ks):
            change = abs(math.atan(vehicle['wheelbase']*ks[i+1])-math.atan(vehicle['wheelbase']*ks[i]))
            if change > 1e-8:
                caps[i] = min(caps[i], vehicle['steer_rate']*ds/change)
    available = max(0.,sum(distance(a,b) for a,b in zip(path,path[1:]))-vehicle['stop_margin'])
    a,t = vehicle['brake'],vehicle['latency']
    # Backward propagation already enforces future braking constraints at the
    # current station. Taking the minimum across the whole horizon incorrectly
    # imposes the near-terminal crawl speed even on a long clear straight.
    cap = min(caps[0], math.sqrt((a*t)**2+2*a*available)-a*t)
    if check_response and abs(steering-measured_steer) > vehicle['steer_rate']*period:
        raise ValueError('steering_response_unavailable')
    return max(0.,cap)
