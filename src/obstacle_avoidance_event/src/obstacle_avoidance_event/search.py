import math
from .model import distance, wrap

def oriented_gate(gate, heading):
    """Orient both endpoints relative to LOCAL travel, not initial vehicle yaw."""
    a,b = gate['endpoints']
    dx,dy = a[0]-b[0],a[1]-b[1]
    lateral = -math.sin(heading)*dx+math.cos(heading)*dy
    # Reject longitudinal links that masquerade as cross-track gates.
    if abs(lateral) < .55*gate['width']:
        return None
    return (a,b) if lateral > 0 else (b,a)

def transition_ok(a,b,heading,side_labels):
    sides = oriented_gate(b,heading)
    if sides is None:
        return False
    for point,label in zip(sides,('left','right')):
        if point in side_labels and side_labels[point] != label:
            return False
    # Side connections must not cross each other.
    old = oriented_gate(a,heading)
    if old is None:
        return False
    def cross(p,q,r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    l0,r0 = old
    l1,r1 = sides
    if cross(l0,l1,r0)*cross(l0,l1,r1) < -1e-8 and cross(r0,r1,l0)*cross(r0,r1,l1) < -1e-8:
        return False
    return True

def search(candidates, adjacent, pose, previous=None, beam_width=12, initial_labels=None):
    """Local beam search with persistent side labels and ordered history."""
    pos, yaw = pose[:2], pose[2]
    starts = []
    for e,g in candidates.items():
        dx,dy = g['center'][0]-pos[0],g['center'][1]-pos[1]
        forward = math.cos(yaw)*dx+math.sin(yaw)*dy
        lateral = -math.sin(yaw)*dx+math.cos(yaw)*dy
        if 0.1 < forward < 4 and abs(lateral) < 1:
            if oriented_gate(g,yaw) is not None:
                starts.append((distance(pos,g['center']),e))
    if not starts:
        raise ValueError('no_start_gate')
    starts.sort()
    start = starts[0][1]
    sides = oriented_gate(candidates[start],yaw)
    labels = dict(initial_labels or {})
    for point,label in zip(sides,('left','right')):
        if point in labels and labels[point] != label:
            raise ValueError('historical_side_conflict')
        labels[point] = label
    beam = [(0.,[start],yaw,labels)]
    completed = []
    for unused in range(40):
        expanded = []
        for cost,seq,heading,labels in beam:
            a = candidates[seq[-1]]
            options = []
            for e in sorted(adjacent[seq[-1]]):
                if e in seq:
                    continue
                b = candidates[e]
                ds = distance(a['center'],b['center'])
                angle = math.atan2(b['center'][1]-a['center'][1],b['center'][0]-a['center'][0])
                turn = abs(wrap(angle-heading))
                if ds < .15 or turn > .8 or not transition_ok(a,b,angle,labels):
                    continue
                # A gate should span the corridor, not lie along travel.
                options.append((e,b,ds,angle,turn))
            if not options:
                completed.append((cost,seq,heading,labels))
            for e,b,ds,angle,turn in options:
                hist = 0.
                if previous:
                    nearest = min(range(len(previous)),key=lambda i:distance(a['center'],previous[i]))
                    hist = min(distance(b['center'],p) for p in previous[nearest:min(len(previous),nearest+5)])
                score = cost+turn*turn+abs(b['width']-a['width'])+.2*hist-.1*ds
                new_labels = dict(labels)
                new_labels.update(zip(oriented_gate(b,angle),('left','right')))
                expanded.append((score,seq+[e],angle,new_labels))
        if not expanded:
            break
        beam = sorted(expanded,key=lambda x:x[0])[:beam_width]
    completed.extend(beam)
    completed = [b for b in completed if len(b[1]) >= 3]
    if not completed:
        raise ValueError('insufficient_corridor')
    completed.sort(key=lambda b:(-len(b[1]),b[0]))
    best = completed[0]
    for alt in completed[1:]:
        if len(alt[1]) == len(best[1]) and abs(alt[0]-best[0]) < .05:
            if set(alt[1]) != set(best[1]):
                raise ValueError('ambiguous_corridor')
    return [candidates[e]['center'] for e in best[1]], best[1]
