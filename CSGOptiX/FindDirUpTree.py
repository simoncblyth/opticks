#!/usr/bin/env python

import os, logging 
log = logging.getLogger(__name__)

def FindDirUpTree(origpath, name="CSGFoundry"): 
    elem = origpath.split("/")
    found = None
    for i in range(len(elem),0,-1):
        path = "/".join(elem[:i])
        cand = os.path.join(path, name)
        log.debug(cand) 
        if os.path.isdir(cand):
            found = cand
            break 
        pass  
    pass
    return found 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    path = os.path.expandvars("$HOME/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1/CSG_GGeo/CSGFoundry")
    fold = FindDirUpTree(path, "CSGFoundry")

    print(fold)

