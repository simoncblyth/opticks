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

::

    In [2]: t.sframe
    Out[2]: 
    sframe       : 
    path         : /tmp/blyth/opticks/G4CXSimtraceTest/hama_body_log/sframe.npy
    meta         : {'creator': 'sframe::save', 'frs': '-1'}
    ce           : array([   0.,    0.,    0., 2000.], dtype=float32)
    grid         : ix0   -5 ix1    5 iy0    0 iy1    0 iz0   -5 iz1    5 num_photon 1000 gridscale     1.0000
    bbox         : array([[-10000.,      0., -10000.],
           [ 10000.,      0.,  10000.]], dtype=float32)
    target       : midx      0 mord      0 iidx      0       inst       0   
    qat4id       : ins_idx     -1 gas_idx   -1   -1 
    m2w          : 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    w2m          : 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    id           : 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)
    ins_gas_ias  :  ins      0 gas    0 ias    0 


The -10000. comes from sframe.py -5.*2000*1. but seems not used ?


Grid comes from default CEGS and GRIDSCALE::

    102 NP* SFrameGenstep::MakeCenterExtentGensteps(sframe& fr)
    103 {
    104     const float4& ce = fr.ce ; 
    105     float gridscale = SSys::getenvfloat("GRIDSCALE", 1.0 ) ;
    106 
    107     // CSGGenstep::init
    108     std::vector<int> cegs ; 
    109     SSys::getenvintvec("CEGS", cegs, ':', "5:0:5:1000" );
    110 
    111     StandardizeCEGS(ce, cegs, gridscale );  // ce is informational here 
    112     assert( cegs.size() == 7 );
    113 
    114     fr.set_grid(cegs, gridscale); 
    115 
    116 
    117     std::vector<float3> ce_offset ; 
    118     CE_OFFSET(ce_offset, ce); 
    119 
    120     LOG(info) 
    121         << " ce " << ce 
    122         << " ce_offset.size " << ce_offset.size() 
    123         ;
    124 
    125 
    126     int ce_scale = SSys::getenvint("CE_SCALE", 1) ; // TODO: ELIMINATE AFTER RTP CHECK 
    127     if(ce_scale == 0) LOG(fatal) << "warning CE_SCALE is not enabled : NOW THINK THIS SHOULD ALWAYS BE ENABLED " ;    

    128  
    129 
    130     Tran<double>* geotran = Tran<double>::FromPair( &fr.m2w, &fr.w2m, 1e-6 );
    131 
    132     NP* gs = MakeCenterExtentGensteps(ce, cegs, gridscale, geotran, ce_offset, ce_scale );
    133 
    134     //gs->set_meta<std::string>("moi", moi );
    135     gs->set_meta<int>("midx", fr.midx() );
    136     gs->set_meta<int>("mord", fr.mord() );
    137     gs->set_meta<int>("iidx", fr.iidx() );
    138     gs->set_meta<float>("gridscale", fr.gridscale() );
    139     gs->set_meta<int>("ce_scale", int(ce_scale) ); 
    140     
    141     return gs ; 
    142 }




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




