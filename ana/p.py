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
import hashlib, builtins
from opticks.ana.hismask import HisMask   
from opticks.ana.histype import HisType  
from opticks.ana.nibble import count_nibbles 
from opticks.ana.nbase import cus


hm = HisMask()
ht = HisType()

seqhis_ = lambda s:ht.label(s)
cseqhis_ = lambda l:ht.code(l)
seqnib_ = lambda s:count_nibbles(s)

#seqdesc_ = lambda s:"seqdesc_ %20s " % 



class CUSS(np.ndarray): pass

def cuss(s,w=None):
    """
    CountUniqueSubSelections 

    :param s: seqhis array with same shape as w that provides the histories of the selection 
    :param w: None OR array of indices expressing a selection of eg deviant photons created for example using np.unique np.where
    :return o: CUSS instance view of cus count-unique-sorted array listing unique seqhis 
               and their counts in descending frequency order 

    This aims to make it easier to investigate seqhis sub-selections of selections (eg deviants)
    CAUTION: using this auto-populates via builtins the calling scope with np.where arrays
    for each of the sub-selections::

          w0, w1, w2, ... 

    Usage example::

        w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0])  ## unique indices selection 
        s = a.seq[w,0]     # seqhis histories of the selection 
        o = cuss(s,w)    
        print(o)
        print(w1)
    
    """
    if w is None:
        w = np.arange(len(s))    
    pass
    assert w.shape == s.shape
    cu = cus(s)   ## count unique sorted 
    for i in range(len(cu)): setattr( builtins, "w%d"%i, w[s == cu[i,0]] )

    o = np.zeros( (cu.shape[0], cu.shape[1]+2), dtype=np.object )
    o[:,0] = list(map(lambda _:"w%d" % _ , list(range(len(cu))) )) 
    o[:,1] = list(map(lambda _:"%30s" % _, seqhis_(cu[:,0]) ))  
    o[:,2] = list(map(lambda _:"%16s" % _, cu[:,0] ))
    o[:,3] = list(map(lambda _:"%16s" % _, cu[:,1] ))

    ret = o.view(CUSS)
    ret.cu = cu 
    return ret




digest_ = lambda a:hashlib.md5(a.data.tobytes()).hexdigest()  

## using ellipsis avoids having to duplicate for photons and records 
#3,0
boundary___ = lambda p:p.view(np.uint32)[...,3,0] >> 16
boundary__ = lambda p:p.view(np.uint32)[:,3,0] >> 16
boundary_  = lambda p:p.view(np.uint32)[3,0] >> 16

flag___    = lambda p:p.view(np.uint32)[...,3,0] & 0xffff
flag__    = lambda p:p.view(np.uint32)[:,3,0] & 0xffff
flag_     = lambda p:p.view(np.uint32)[3,0] & 0xffff

#3,1
identity___   = lambda p:p.view(np.uint32)[...,3,1]   
identity__    = lambda p:p.view(np.uint32)[:,3,1]   
identity_     = lambda p:p.view(np.uint32)[3,1] 
  
primIdx___    = lambda p:p.view(np.uint32)[...,3,1] >> 16 
primIdx__     = lambda p:p.view(np.uint32)[:,3,1] >> 16 

instanceId___ = lambda p:p.view(np.uint32)[...,3,1] & 0xffff  
instanceId__  = lambda p:p.view(np.uint32)[:,3,1] & 0xffff  

primIdx_      = lambda p:identity_(p) >> 16 
instanceId_   = lambda p:identity_(p) & 0xffff  



ident_ = lambda p:p.view(np.uint32)[...,3,1]
prim_  = lambda p:ident_(p) >> 16
inst_  = lambda p:ident_(p) & 0xffff

iindex_ = lambda p:p.view(np.uint32)[...,1,3] 

#3,2
idx_      = lambda p:p.view(np.uint32)[3,2] & 0x7fffffff
orient___  = lambda p:p.view(np.uint32)[...,3,2] >> 31
orient_   = lambda p:p.view(np.uint32)[3,2] >> 31

#3,3
flagmask_ = lambda p:p.view(np.uint32)[3,3]

flagdesc_ = lambda p:" idx(%6d) prd(b%3d p%4d i%5d o%1d ii:%5d) %3s  %15s " % ( idx_(p),  boundary_(p),primIdx_(p),instanceId_(p), orient_(p), iindex_(p), hm.label(flag_(p)),hm.label( flagmask_(p) ))


flagmask__ = lambda p:p.view(np.uint32)[:,3,3]
hit__      = lambda p,msk:p[np.where( ( flagmask__(p) & msk ) == msk)]    



## TO PICK THE GEOMETRY APPROPRIATE TO THE RESULT ARRAYS SET CFBASE envvar TO DIRECTORY CONTAINING CSGFoundry dir 
from opticks.CSG.CSGFoundry import CSGFoundry 
cf = CSGFoundry.Load()

if not cf is None:
    cf_bnd_  = lambda p:cf.sim.bndnamedict.get(boundary_(p),"cf_bnd_ERR")
    cf_prim_ = lambda p:cf.primIdx_meshname_dict.get(prim_(p),"cf_prim_ERR")
else:
    cf_bnd_  = lambda p:"no_cf"
    cf_prim_  = lambda p:"no_cf"
pass

bflagdesc_ = lambda p:"%s : %60s : %s : %s " % ( flagdesc_(p), cf_prim_(p) , digest_(p[:3])[:8], cf_bnd_(p) )


ridiff_ = lambda ri:ri[1:,:3] - ri[:-1,:3]     

# rdist_(a,i)/rtime_(a,i)/rspeed_(a,i) : distance/time/speed between record point i and i+1 
rdist_ = lambda a,i:np.sqrt(np.sum( (a.record[:,i+1,0,:3]-a.record[:,i,0,:3])*(a.record[:,i+1,0,:3]-a.record[:,i,0,:3]) , axis=1 ))
rtime_ = lambda a,i:a.record[:,i+1,0,3] - a.record[:,i,0,3]  
rspeed_ = lambda a,i:rdist_(a,i)/rtime_(a,i)




