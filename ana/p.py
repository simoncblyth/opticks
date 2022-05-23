#!/usr/bin/env python

import numpy as np
import hashlib
from opticks.ana.hismask import HisMask   
from opticks.ana.histype import HisType  

hm = HisMask()
ht = HisType()

seqhis_ = lambda s:ht.label(s)

digest_ = lambda a:hashlib.md5(a.data.tobytes()).hexdigest()  


boundary__ = lambda p:p.view(np.uint32)[:,3,0] >> 16
boundary_  = lambda p:p.view(np.uint32)[3,0] >> 16

flag__    = lambda p:p.view(np.uint32)[:,3,0] & 0xffff
flag_     = lambda p:p.view(np.uint32)[3,0] & 0xffff

identity_ = lambda p:p.view(np.uint32)[3,1]   
primIdx_   = lambda p:identity_(p) >> 16 
instanceId_  = lambda p:identity_(p) & 0xffff  

idx_      = lambda p:p.view(np.uint32)[3,2] & 0x7fffffff
orient_   = lambda p:p.view(np.uint32)[3,2] >> 31

flagmask_ = lambda p:p.view(np.uint32)[3,3]

flagdesc_ = lambda p:" %6d prd(%3d %4d %5d %1d)  %3s  %15s " % ( idx_(p),  boundary_(p),primIdx_(p),instanceId_(p), orient_(p),  hm.label(flag_(p)),hm.label( flagmask_(p) ))


flagmask__ = lambda p:p.view(np.uint32)[:,3,3]
hit__      = lambda p,msk:p[np.where( ( flagmask__(p) & msk ) == msk)]    



## TO PICK THE GEOMETRY APPROPRIATE TO THE RESULT ARRAYS SET CFBASE envvar 
from opticks.CSG.CSGFoundry import CSGFoundry 
cf = CSGFoundry.Load()

bflagdesc_ = lambda p:"%s : %40s : %s " % ( flagdesc_(p), cf.sim.bndnamedict[boundary_(p)], digest_(p[:3]) )


ridiff_ = lambda ri:ri[1:,:3] - ri[:-1,:3]     






