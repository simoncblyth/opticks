#!/usr/bin/env python
"""
p.py : HMM perhaps rename to sysrap/sphoton.py beside sphoton.h
===========================================================================

NB locations and packing used here must match sysrap/sphoton.h::

    050 struct sphoton
     51 {
     52     float3 pos ;
     53     float  time ;
     54 
     55     float3 mom ;
     56     float  weight ;
     57 
     58     float3 pol ;
     59     float  wavelength ;
     60 
     61     unsigned boundary_flag ;  // p.view(np.uint32)[3,0] 
     62     unsigned identity ;       // p.view(np.uint32)[3,1]
     63     unsigned orient_idx ;     // p.view(np.uint32)[3,2]
     64     unsigned flagmask ;       // p.view(np.uint32)[3,3]
     65 
     66 
     67     SPHOTON_METHOD void set_prd( unsigned  boundary, unsigned  identity, float  orient );


     72     SPHOTON_METHOD void set_orient(float orient){ orient_idx = ( orient_idx & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; } // clear orient bit and then set it 
     73     SPHOTON_METHOD void set_idx( unsigned idx ){  orient_idx = ( orient_idx & 0x80000000u ) | ( 0x7fffffffu & idx ) ; }   // retain bit 31 asis 
     74 
    105 SPHOTON_METHOD void sphoton::set_prd( unsigned  boundary_, unsigned  identity_, float  orient_ )
    106 {
    107     set_boundary(boundary_);
    108     identity = identity_ ;
    109     set_orient( orient_ );
    110 }



"""
import numpy as np
import hashlib
from opticks.ana.hismask import HisMask   
from opticks.ana.histype import HisType  
from opticks.ana.nibble import count_nibbles 


hm = HisMask()
ht = HisType()

seqhis_ = lambda s:ht.label(s)
seqnib_ = lambda s:count_nibbles(s)

#seqdesc_ = lambda s:"seqdesc_ %20s " % 


digest_ = lambda a:hashlib.md5(a.data.tobytes()).hexdigest()  


boundary__ = lambda p:p.view(np.uint32)[:,3,0] >> 16
boundary_  = lambda p:p.view(np.uint32)[3,0] >> 16

flag__    = lambda p:p.view(np.uint32)[:,3,0] & 0xffff
flag_     = lambda p:p.view(np.uint32)[3,0] & 0xffff

identity__ = lambda p:p.view(np.uint32)[:,3,1]   
primIdx__   = lambda p:identity__(p) >> 16 
instanceId__  = lambda p:identity__(p) & 0xffff  


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






