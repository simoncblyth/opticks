#!/usr/bin/env python 
"""
U4RecorderTest.py
==================

::

     18 struct spho
     19 {
     20     int gs ; // 0-based genstep index within the event
     21     int ix ; // 0-based photon index within the genstep
     22     int id ; // 0-based photon identity index within the event 
     23     int gn ; // 0-based reemission index incremented at each reemission 


"""
import numpy as np
from opticks.ana.fold import Fold


def check_pho_labels(l):
     """ 
     :param l: spho labels 

     When reemission is enabled this would fail for pho0 (push_back labels)
     but should pass for pho (labels slotted in by event photon id)
     """

     gs = l[:,0]
     ix = l[:,1]
     id_ = l[:,2]
     gn = l[:,3]

     gs_u, gs_c = np.unique(gs, return_counts=True ) 
     np.all( np.arange( len(gs_u) ) == gs_u )       
     # not expecting gaps in list of unique genstep index as there should always be at least one photon per genstep

     id_u, id_c = np.unique( id_, return_counts=True  )  
     assert np.all( id_c == 1 )  
     # expecting the photon identity index to be unique within event, so these should all be 1 
     # if not then that points to problem with labelling 

     ix_u, ix_c = np.unique( ix, return_counts=True )  

     gn_u, gn_c = np.unique( gn, return_counts=True )  
     print(gn_u)
     print(gn_c)


if __name__ == '__main__':

     t = Fold.Load() 
     print(t)

     # pho: labels are collected within U4Recorder::PreUserTrackingAction 
     check_pho_labels(t.pho)

     #l = t.pho0   # push_back labels collected asis
     l = t.pho     # slotted in using spho::id

     gs = l[:,0]
     ix = l[:,1]
     id_ = l[:,2]
     gn = l[:,3]


     idx = t.p.view(np.uint32)[:,3,2] 
     assert np.all( np.arange( len(t.p) ) == idx ) 

     flagmask_u, flagmask_c = np.unique(t.p.view(np.uint32)[:,3,3], return_counts=True)    
     print("flagmask_u:%s " % str(flagmask_u))
     print("flagmask_c:%s " % str(flagmask_c))



