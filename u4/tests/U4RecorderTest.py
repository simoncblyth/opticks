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

if __name__ == '__main__':

     t = Fold.Load() 
     print(t)

     l = t.labels

     gs = l[:,0]
     ix = l[:,1]
     id_ = l[:,2]
     gn = l[:,3]


     gs_u, gs_c = np.unique(gs, return_counts=True ) 
     np.all( np.arange( len(gs_u) ) == gs_u )       # not expecting gaps in the genstep index

     id_u, id_c = np.unique( id_, return_counts=True  )  
     assert np.all( id_c == 1 )  # expecring the photon identity index to be unique within event, so these should all be 1 
     # TODO: check when are collecting both Cerenkov and Scintillation 

     ix_u, ix_c = np.unique( ix, return_counts=True )  

     assert np.all( gn  == 0 )     # reemission is switched off currently, so all 0





