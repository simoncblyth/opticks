cxt_min_simtrace_frame_targeting_not_working
==============================================


Workflow
---------

::

    LOG=1 ~/o/cxt_min.sh

    ~/o/sysrap/tests/SFrameGenstep_MakeCenterExtentGensteps_Test.sh

    ~/o/CSG/tests/CSGFoundry_MakeCenterExtentGensteps_Test.sh

    ~/o/cxt_min.sh          
    ~/o/cxt_min.sh grab 
    ~/o/cxt_min.sh ana


Overview
-----------

* changed QEvent back to using NP gs so have gensteps to debug 
* find that with adhoc GRIDSCALE of 100 can get OK traces for PMTs
* but that doesnt work for the Chimney, grid is still mid-CD

  * presumably because Chimney is global not instanced

YES: getting identity transform is actually correct, which 
means must special case handle global geometry being 
offset from the origin 


sframe with PMT
------------------

::

    In [1]: sf                                                                                                                                 
    Out[1]: 
    sframe       : 
    path         : /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXTMTest/NNVT:0:1000/A000/sframe.npy
    meta         : creator:sframe::getFrameArray
    frs:NNVT:0:1000
    ek:MOI
    ev:NNVT:0:1000
    ekvid:sframe_MOI_NNVT_0_1000
    ce           : array([-3156.737, 10406.367, 16012.954,   348.289], dtype=float32)
    grid         : ix0  -16 ix1   16 iy0    0 iy1    0 iz0   -9 iz1    9 num_photon 2000 gridscale   100.0000
    bbox         : array([[-557262.5 ,       0.  , -313460.16],
           [ 557262.5 ,       0.  ,  313460.16]], dtype=float32)
    target       : midx    109 mord      0 iidx   1000       inst       0   
    qat4id       : ins_idx     -1 gas_idx   -1   -1 
    m2w          : 
    array([[    0.24 ,    -0.792,     0.562,     0.   ],
           [   -0.957,    -0.29 ,     0.   ,     0.   ],
           [    0.163,    -0.538,    -0.827,     0.   ],
           [-3169.384, 10448.06 , 16077.108,     1.   ]], dtype=float32)

    w2m          : 
    array([[    0.24 ,    -0.957,     0.163,     0.   ],
           [   -0.792,    -0.29 ,    -0.538,     0.   ],
           [    0.562,    -0.   ,    -0.827,     0.   ],
           [   -0.005,     0.001, 19434.   ,     1.   ]], dtype=float32)

    id           : 
    array([[ 1.   , -0.   , -0.   ,  0.   ],
           [ 0.   ,  1.   , -0.   ,  0.   ],
           [-0.   , -0.   ,  1.   ,  0.   ],
           [ 0.001,  0.   ,  0.   ,  1.   ]], dtype=float32)
    ins_gas_ias  :  ins      0 gas    0 ias    0 

    In [2]:                               




sframe with chimney
----------------------

::

    sframe       : 
    path         : /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXTMTest/sChimneyAcrylic:0:0/A000/sframe.npy
    meta         : creator:sframe::getFrameArray
    frs:sChimneyAcrylic:0:0
    ek:MOI
    ev:sChimneyAcrylic:0:0
    ekvid:sframe_MOI_sChimneyAcrylic_0_0
    ce           : array([    0.,     0., 18124.,   524.], dtype=float32)
    grid         : ix0  -16 ix1   16 iy0    0 iy1    0 iz0   -9 iz1    9 num_photon 2000 gridscale   100.0000
    bbox         : array([[-838400.,       0., -471600.],
           [ 838400.,       0.,  471600.]], dtype=float32)
    target       : midx    123 mord      0 iidx      0       inst       0   
    qat4id       : ins_idx     -1 gas_idx   -1   -1 
    m2w          : 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    w2m          : 
    array([[ 1., -0.,  0.,  0.],
           [-0.,  1., -0.,  0.],
           [ 0., -0.,  1.,  0.],
           [-0.,  0., -0.,  1.]], dtype=float32)

    id           : 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)
    ins_gas_ias  :  ins      0 gas    0 ias    0 

    In [2]:                             


::

    3351 int CSGFoundry::getFrame(sframe& fr, int midx, int mord, int iidxg) const
    3352 {
    3353     int rc = 0 ;
    3354     if( midx == -1 )
    3355     {
    3356         unsigned long long emm = 0ull ;   // hmm instance var ?
    3357         iasCE(fr.ce, emm);
    3358     }
    3359     else
    3360     {
    3361         rc = target->getFrame( fr, midx, mord, iidxg );
    3362     }
    3363     return rc ;
    3364 }

    135 int CSGTarget::getFrame(sframe& fr,  int midx, int mord, int iidxg ) const
    136 {
    137     fr.set_midx_mord_iidx( midx, mord, iidxg );
    138     int rc = getFrameComponents( fr.ce, midx, mord, iidxg, &fr.m2w , &fr.w2m );
    139     LOG(LEVEL) << " midx " << midx << " mord " << mord << " iidxg " << iidxg << " rc " << rc ;
    140     return rc ;
    141 }



getFrameComponents_called_twice
----------------------------------

:doc:`getFrameComponents_called_twice`

Issue with cxt_min.sh 
-------------------------


MOI=ALL
   gives expected simtrace of whole detector, showing 2D slice thru all detector

MOI=sChimneyAcrylic:0:0 
   gives unexpected simtrace with just a circle and blip looking like the simtrace 
   grid is at center of CD rather than in the throat of the chimney as intended

::

    N[blyth@localhost ~]$ LOG=1 BP=SFrameGenstep::MakeCenterExtentGenstep ~/o/cxt_min.sh

    ...

    2023-12-13 10:05:15.695 INFO  [65161] [CSGOptiX::setFrame@796]  ce [ 0 0 18124 524] sglm.TMIN 0.1 sglm.tmin_abs 52.4 sglm.m2w.is_zero 0 sglm.w2m.is_zero 0
    2023-12-13 10:05:15.695 INFO  [65161] [CSGOptiX::setFrame@804] m2w ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000) 
    2023-12-13 10:05:15.695 INFO  [65161] [CSGOptiX::setFrame@805] w2m ( 1.000,-0.000, 0.000, 0.000) (-0.000, 1.000,-0.000, 0.000) ( 0.000,-0.000, 1.000, 0.000) (-0.000, 0.000,-0.000, 1.000) 
    2023-12-13 10:05:15.695 INFO  [65161] [CSGOptiX::setFrame@807] ]
    2023-12-13 10:05:15.695 INFO  [65161] [CSGOptiX::init@457] ]
    2023-12-13 10:05:15.695 INFO  [65161] [CSGOptiX::Create@370] ]

    (gdb) bt
    #0  0x00007ffff7177b40 in SFrameGenstep::MakeCenterExtentGenstep(sframe&)@plt ()
       from /data/blyth/junotop/ExternalLibs/opticks/head/lib/../lib64/libSysRap.so
    #1  0x00007ffff72350f2 in SEvt::addInputGenstep (this=0x69835c0) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:727
    #2  0x00007ffff72388a4 in SEvt::beginOfEvent (this=0x69835c0, eventID=0) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:1563
    #3  0x00007ffff74a2f64 in QSim::simtrace (this=0xacc2b60, eventID=0) at /home/blyth/junotop/opticks/qudarap/QSim.cc:396
    #4  0x00007ffff7e5c513 in CSGOptiX::simtrace (this=0xacd16b0, eventID=0) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:671
    #5  0x00007ffff7e5950d in CSGOptiX::SimtraceMain () at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:170
    #6  0x0000000000405b15 in main (argc=1, argv=0x7fffffff23b8) at /home/blyth/junotop/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) 


::

     630 void CSGOptiX::initFrame()
     631 {
     632     sframe _fr = foundry->getFrameE() ;
     633     LOG(LEVEL) << _fr ;
     634     SEvt::SetFrame(_fr) ;
     635     setFrame(_fr);
     636 }

