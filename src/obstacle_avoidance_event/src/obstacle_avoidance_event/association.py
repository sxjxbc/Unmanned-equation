"""Conservative one-to-one cone identity association in a fixed map frame."""
from .model import distance

class ConeTracker(object):
    def __init__(self, radius=.4, ambiguity=.05):
        self.radius,self.ambiguity = radius,ambiguity
        self.points = {}
        self.next_id = 0
        self.version = None
        self.stamp = None

    def update(self, points, stamp, version):
        if self.version is not None and version != self.version:
            raise ValueError('map_version_changed')
        if self.stamp is not None and stamp <= self.stamp:
            raise ValueError('non_monotonic_observation')
        assignments,used = [],set()
        for point in points:
            matches = sorted((distance(point,p),i) for i,p in self.points.items()
                             if distance(point,p) <= self.radius)
            if len(matches)>1 and matches[1][0]-matches[0][0] < self.ambiguity:
                raise ValueError('ambiguous_cone_identity')
            identity = matches[0][1] if matches else None
            if identity is not None and identity in used:
                raise ValueError('non_unique_cone_identity')
            assignments.append(identity)
            if identity is not None:used.add(identity)
        # Commit only after the entire frame is unambiguous.
        updated = {}
        for point,identity in zip(points,assignments):
            if identity is None:
                identity = self.next_id
                self.next_id += 1
            updated[identity] = point
        self.points,self.version,self.stamp = updated,version,stamp
        return dict(updated)
