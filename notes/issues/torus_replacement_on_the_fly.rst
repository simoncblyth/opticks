torus_replacement_on_the_fly
=============================

context
---------

* :doc:`quartic_solve_optix_600_misaligned_address_exception`


thoughts
-----------

As only have TITAN RTX for a few more days, and want to 
make some full geometry benchmarks with and without OptiX_600 
RTX better to:

1. just skip the torus guide tube (--csgskiplv ??)
2. on-the-fly change PMT neck to use hyperboloid OR polycone


geocache-j1808 
-----------------

:: 

   geocache-j1808 () 
    { 
        type $FUNCNAME;
        opticksdata-;
        OKX4Test --gdmlpath $(opticksdata-j) --g4codegen --csgskiplv 22
    }


okg4/tests/OKX4Test.cc
-------------------------

1. pure G4 parses GDML
2. X4PhysicalVolume traverses the G4 tree and populates a GGeo 
3. wraps the GGeo in OKMgr and visualizes


csgskiplv : added int list handling  
-----------------------------------------

::

    [blyth@localhost opticks]$ opticks-findl csgskiplv
    ./ana/geocache.bash
    ./ggeo/GInstancer.cc
    ./optickscore/OpticksCfg.cc
    ./ggeo/GInstancer.hh
    ./optickscore/OpticksCfg.hh

    [blyth@localhost optickscore]$ opticks-f getCSGSkipLV
    ./ggeo/GGeo.cc:    m_instancer->setCSGSkipLV(m_ok->getCSGSkipLV()) ;  
    ./optickscore/Opticks.hh:       int   getCSGSkipLV() const ;
    ./optickscore/Opticks.cc:int Opticks::getCSGSkipLV() const 
    ./optickscore/Opticks.cc:   return m_cfg->getCSGSkipLV();
    ./optickscore/OpticksCfg.cc:int OpticksCfg<Listener>::getCSGSkipLV() const 
    ./optickscore/OpticksCfg.hh:     int          getCSGSkipLV() const ;  


need to be able to skip more than one lv
--------------------------------------------

::

    --csgskiplv 22,32,33

    ## 22: lMaskVirtual0x4c803b0 a misused polycone (actually a tubs) 
           used for technical G4 performance reasons to contain the PMT : but that obscures it so skip

    ## 32,33 : lvacSurftube0x5b3c020, lSurftube0x5b3ac50


::

    2019-04-15 16:47:13.683 INFO  [306341] [GInstancer::dump@625] GGeo::prepareVolumes
    2019-04-15 16:47:13.683 INFO  [306341] [GInstancer::dumpMeshset@569]  numRepeats 5 numRidx 6 (slot 0 for global non-instanced) 
     ridx 1 ms 5 ( 23 24 25 26 27  ) 
     ridx 2 ms 6 ( 17 18 19 20 21 22  ) 
     ridx 3 ms 4 ( 4 5 6 7  ) 
     ridx 4 ms 1 ( 15  ) 
     ridx 5 ms 1 ( 16  ) 
    2019-04-15 16:47:13.683 INFO  [306341] [GInstancer::dumpCSGSkips@601] 
     lvIdx 22 skip total : 20046 nodeIdx ( 63555 63561 63567 63573 63579 63585 63591 63597 63603 63609 63615 63621 63627 63633 63639 63645 63651 63657 63663 63669  ...  ) 


* hmm the skipping not working for globals 32 and 33 (guide tube)


after add traverseGlobals to GInstancer
---------------------------------------------

::

    2019-04-15 17:19:53.344 INFO  [357943] [GInstancer::dump@651] GGeo::prepareVolumes
    2019-04-15 17:19:53.344 INFO  [357943] [GInstancer::dumpMeshset@595]  numRepeats 5 numRidx 6 (slot 0 for global non-instanced) 
     ridx 0 ms 23 ( 0 1 2 3 8 9 10 11 12 13 14 28 29 30 31 32 33 34 35 36 37 38 39  ) 
     ridx 1 ms 5 ( 23 24 25 26 27  ) 
     ridx 2 ms 6 ( 17 18 19 20 21 22  ) 
     ridx 3 ms 4 ( 4 5 6 7  ) 
     ridx 4 ms 1 ( 15  ) 
     ridx 5 ms 1 ( 16  ) 
    2019-04-15 17:19:53.345 INFO  [357943] [GInstancer::dumpCSGSkips@627] 
     lvIdx 22 skip total : 20046 nodeIdx ( 63555 63561 63567 63573 63579 63585 63591 63597 63603 63609 63615 63621 63627 63633 63639 63645 63651 63657 63663 63669  ...  ) 
     lvIdx 32 skip total : 1 nodeIdx ( 352854  ) 
     lvIdx 33 skip total : 1 nodeIdx ( 352853  ) 
    2019-04-15 17:19:53.345 INFO  [357943] [GGeo::prepare@683] prepareVertexColors


segv after getting global volumes skipped
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    ## with gdb --args OKX4Test --gdmlpath $(opticksdata-j) --g4codegen --csgskiplv 22,32,33   
    ## no segv with --csgskiplv 22
    ## still segv with --csgskiplv 22,32
    
 
    (gdb) bt
    #0  0x00007fffe20cf3b1 in __strlen_sse2 () from /lib64/libc.so.6
    #1  0x00007fffe20cf0be in strdup () from /lib64/libc.so.6
    #2  0x00007ffff7548583 in RBuf::RBuf (this=0x18850bb30, num_items_=0, num_bytes_=1062129359, num_elements_=1057458056, ptr_=0x3f4ececf3f078788, name_=0x3f7afafb3f4ececf <Address 0x3f7afafb3f4ececf out of bounds>)
        at /home/blyth/opticks/oglrap/RBuf.cc:26
    #3  0x00007ffff756ffa2 in Renderer::setDrawable (this=0x18636f530, drawable=0x1121b0470) at /home/blyth/opticks/oglrap/Renderer.cc:286
    #4  0x00007ffff756f9a2 in Renderer::upload (this=0x18636f530, mm=0x1121b0470) at /home/blyth/opticks/oglrap/Renderer.cc:257
    #5  0x00007ffff75618db in Scene::uploadGeometryGlobal (this=0x186369d90, mm=0x1121b0470) at /home/blyth/opticks/oglrap/Scene.cc:553
    #6  0x00007ffff756216f in Scene::uploadGeometry (this=0x186369d90) at /home/blyth/opticks/oglrap/Scene.cc:634
    #7  0x00007ffff757871f in OpticksViz::uploadGeometry (this=0x186368af0) at /home/blyth/opticks/oglrap/OpticksViz.cc:326
    #8  0x00007ffff757790b in OpticksViz::init (this=0x186368af0) at /home/blyth/opticks/oglrap/OpticksViz.cc:141
    #9  0x00007ffff75774f1 in OpticksViz::OpticksViz (this=0x186368af0, hub=0x18634b540, idx=0x186367620, immediate=true) at /home/blyth/opticks/oglrap/OpticksViz.cc:98
    #10 0x00007ffff79cb92e in OKMgr::OKMgr (this=0x7fffffffcb60, argc=6, argv=0x7fffffffda18, argforced=0x0) at /home/blyth/opticks/ok/OKMgr.cc:49
    #11 0x000000000040521f in main (argc=6, argv=0x7fffffffda18) at /home/blyth/opticks/okg4/tests/OKX4Test.cc:121
    (gdb) f 5
    #5  0x00007ffff75618db in Scene::uploadGeometryGlobal (this=0x186369d90, mm=0x1121b0470) at /home/blyth/opticks/oglrap/Scene.cc:553
    553             m_global_renderer->upload(mm);  
    (gdb) f 4
    #4  0x00007ffff756f9a2 in Renderer::upload (this=0x18636f530, mm=0x1121b0470) at /home/blyth/opticks/oglrap/Renderer.cc:257
    257     setDrawable(mm);
    (gdb) f 3
    #3  0x00007ffff756ffa2 in Renderer::setDrawable (this=0x18636f530, drawable=0x1121b0470) at /home/blyth/opticks/oglrap/Renderer.cc:286
    286     m_cbuf = MAKE_RBUF(m_drawable->getColorsBuffer());
    (gdb) f 2
    #2  0x00007ffff7548583 in RBuf::RBuf (this=0x18850bb30, num_items_=0, num_bytes_=1062129359, num_elements_=1057458056, ptr_=0x3f4ececf3f078788, name_=0x3f7afafb3f4ececf <Address 0x3f7afafb3f4ececf out of bounds>)
        at /home/blyth/opticks/oglrap/RBuf.cc:26
    26      debug_index(-1)
    (gdb) f 6
    #6  0x00007ffff756216f in Scene::uploadGeometry (this=0x186369d90) at /home/blyth/opticks/oglrap/Scene.cc:634
    634            uploadGeometryGlobal(mm);
    (gdb) f 7
    #7  0x00007ffff757871f in OpticksViz::uploadGeometry (this=0x186368af0) at /home/blyth/opticks/oglrap/OpticksViz.cc:326
    326     m_scene->uploadGeometry();
    (gdb) 





finding lv with torus 
-----------------------

::

    2019-04-15 15:59:04.897 INFO  [220717] [X4PhysicalVolume::convertSolid@500]  ] 39
    2019-04-15 15:59:04.898 INFO  [220717] [X4PhysicalVolume::dumpTorusLV@557]  num_afflicted 6
     lvIdx ( 18 19 20 21 32 33  ) 
    18 PMT_20inch_inner1_log0x4cb3cc0
    19 PMT_20inch_inner2_log0x4c9a6e0
    20 PMT_20inch_body_log0x4cb3aa0
    21 PMT_20inch_log0x4cb3bb0
    32 lvacSurftube0x5b3c020
    33 lSurftube0x5b3ac50


for fast cycle need to write out GDML for a single PMT ? 
----------------------------------------------------------

* hmm investigate g4codegen, G4 code would do just as well as GDML





