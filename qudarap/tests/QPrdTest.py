#!/usr/bin/env python

import os, numpy as np

if __name__ == '__main__':
     path = os.path.expandvars("/tmp/$USER/opticks/QPrdTest/prd.npy")
     a = np.load(path)
     print("PRD %s : %s " % (str(a.shape), path))
     print("mock PRD isect : surface normal and distance " )
     print(a[:,:,0,:].view(np.float32)) 

     print(" mock PRD identity and boundary" ) 
     print(a[:,:,1,:].view(np.uint32)) 



