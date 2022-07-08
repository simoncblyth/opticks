cxs_check_following_SEvt_rejig
=================================

::

    cx
    ./cxs_Hama.sh         # workstation
    ./cxs_Hama.sh grab    # laptop
    ./cxs_Hama.sh ana     # laptop


After upping the SEvt logging see that is writing to::

    /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGOptiXSimtraceTest



gxt.sh : g4cx/tests/G4CXSimtraceTest.py
-----------------------------------------

issue 1 : default genstep grid is tiny around origin : why ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


FIXED : issue 2 : MASK=t making half PMT disappear ? Was due to simtrace "photon" array layout change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   MASK=t ./gxt.sh ana
       curious MASK=t is unexpected making half the hama_body_log disappear ? 



ana/simtrace_positions.py::

    128     def apply_t_mask(self):
    129         """
    130         t_mask restricts the intersect distance t to be greater than zero
    131         this excludes misses 
    132         """
    133         log.info("apply_t_mask")
    134         t = self.p[:,2,2]
    135         mask = t > 0.
    136         self.apply_mask( mask)


Looks like not using the real distance ?::

    In [3]: t_pos.p[:,2,2]
    Out[3]: array([-5., -5., -5., -5., -5., ...,  5.,  5.,  5.,  5.,  5.], dtype=float32)

    In [4]: t_pos.p[:,2,2].shape
    Out[4]: (31506,)

    In [5]: np.unique( t_pos.p[:,2,2], return_counts=True )
    Out[5]: 
    (array([-5., -4., -3., -2., -1.,  1.,  2.,  3.,  4.,  5.], dtype=float32),
     array([2403, 2736, 3071, 3530, 4154, 4136, 3466, 2963, 2698, 2349]))




