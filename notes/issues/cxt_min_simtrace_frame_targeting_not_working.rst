cxt_min_simtrace_frame_targeting_not_working
==============================================


Workflow
---------

::

    LOG=1 ~/o/cxt_min.sh

    ~/o/sysrap/tests/SFrameGenstep_MakeCenterExtentGensteps_Test.sh

    ~/o/CSG/tests/CSGFoundry_MakeCenterExtentGensteps_Test.sh



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

