#!/usr/bin/env python

import numpy as np
from opticks.ana.hismask import HisMask   
hm = HisMask()


boundary__ = lambda p:p.view(np.uint32)[:,3,0] >> 16
boundary_  = lambda p:p.view(np.uint32)[3,0] >> 16

flag__    = lambda p:p.view(np.uint32)[:,3,0] & 0xffff
flag_     = lambda p:p.view(np.uint32)[3,0] & 0xffff

identity_ = lambda p:p.view(np.uint32)[3,1]   
idx_      = lambda p:p.view(np.uint32)[3,2] & 0x7fffffff
orient_   = lambda p:p.view(np.uint32)[3,2] >> 31

flagmask_ = lambda p:p.view(np.uint32)[3,3]

flagdesc_ = lambda p:" %6d prd(%3d %3d %1d)  %3s  %10s " % ( idx_(p),  boundary_(p),identity_(p),orient_(p),  hm.label(flag_(p)),hm.label( flagmask_(p) ))


flagmask__ = lambda p:p.view(np.uint32)[:,3,3]
hit__      = lambda p,msk:p[np.where( ( flagmask__(p) & msk ) == msk)]    


## hmm need to use opticks-switch-key when analysing event from remote 
## actually should be using the specific CSGFoundry persisted and grabbed 

from opticks.CSG.CSGFoundry import CSGFoundry 
cf = CSGFoundry.Load()

bflagdesc_ = lambda p:"%s : %s " % ( flagdesc_(p), cf.bndnamedict[boundary_(p)] )





