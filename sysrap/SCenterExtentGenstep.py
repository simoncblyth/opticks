#!/usr/bin/env python
from opticks.ana.fold import Fold

class SCenterExtentGenstep(object):
    def __init__(self, path):
        self.fold = Fold.Load(path)
   

if __name__ == '__main__':
    path = os.path.expandvars("$CFBASE/CSGIntersectSolidTest/$GEOM")
    cegs = SCenterExtentGenstep(path)



 
