mar2022geom_structural_transform_debug 
==========================================

JUNO Renders cxr OptiX 7 renders now looking to use wrong global transforms : chimney in middle of LS
--------------------------------------------------------------------------------------------------------

I have seen something similar before which was due to the --gparts_transform_offset being 
omitted. But that seems not to be the case this time. 


Test with simple geometry : a grid of PMTs : so far no problem
-----------------------------------------------------------------

To setup a simpler environment to debug structural transforms and their conversion, 
get PMTSim::GetPV operational and use it from a new test 

* g4ok/tests/G4OKPMTSimTest.cc
* g4ok/G4OKPMTSimTest.sh

::


     01 #!/bin/bash -l 
      2 
      3 export X4PhysicalVolume=INFO
      4 export GInstancer=INFO
      5 
      6 # comment the below to create an all global geometry, uncomment to instance the PMT volume 
      7 export GInstancer_instance_repeat_min=25
      8 
      9 G4OKPMTSimTest
     10 
     11 
     12 


* used this to create a geocache
* grabbed the OPTICKS_KEY and then created CSG_GGeo CSGFoundry geometry
* rendered it with CSGOptiX cxr_overview.sh : no surprises get a grid of PMTs within the world box

::

    PUB=simple_transform_check EYE=1,0,0 ZOOM=1 ./cxr_overview.sh 


* Using "Six" on laptop with both an instanced and an all global geometry get the expected render of a grid of PMTs in a box. 
* Using OKTest and OTracerTest on laptop (OpenGL rasterized) shows expected 27 PMTs in box

  * NB have to open the Scene menu and select the instances to get the PMTs to appear
  * checking "O" flipping gets the same modulo y-inversion between the ray traced and rasterized 

    * TODO: fix the y-inversion (not urgent)


* checking on P and setting::

    export OPTICKS_KEYDIR_GRABBED=.opticks/geocache/G4OKPMTSimTest_World_pv_g4live/g4ok_gltf/64f67f47f00b6e9831b16c54495f0bdd/1 

Then grabbing from laptop::

   cx 
   ./cxr_grab.sh jpg   

* no surprises the box of PMTs looks as expected 



DONE: OTracerTest check with dayabay gdml geometry : it looks fine after avoid a few asserts, no transform troubles 
-----------------------------------------------------------------------------------------------------------------------

::

   geocache-create    # geocache-dx1 


::

    2022-03-04 00:14:50.886 INFO  [337494] [X4PhysicalVolume::convertSolid@951]  lvname /dd/Geometry/CalibrationSources/lvDiffuserBall0xc3074400x3f225a0 soname DiffuserBall0xc3073d00x3e9b4b0 [--x4skipsolidname] n
    2022-03-04 00:14:50.889 INFO  [337494] [X4PhysicalVolume::convertSolid@951]  lvname /dd/Geometry/CalibrationSources/lvLedSourceShell0xc3066b00x3f226d0 soname led-source-shell0xc3068f00x3e9bd80 [--x4skipsolidname] n
    2022-03-04 00:14:50.891 INFO  [337494] [ncylinder::increase_z2@122]  _z2 14.865 dz 1 new_z2 15.865
    OKX4Test: /data/blyth/junotop/opticks/npy/NZSphere.cpp:134: void nzsphere::check() const: Assertion `fabs(z2()) <= radius()' failed.
    Aborted (core dumped)
    === geocache-create- : logs are in tmp:/tmp/blyth/opticks/geocache-create-
    N[blyth@localhost opticks]$ 
    N[blyth@localhost opticks]$ 



::

    ame] n
    2022-03-04 00:27:56.876 INFO  [353849] [X4PhysicalVolume::convertSolid@951]  lvname /dd/Geometry/CalibrationSources/lvLedSourceShell0xc3066b00x3f226d0 soname led-source-shell0xc3068f00x3e9bd80 [--x4skipsolidname] n
    2022-03-04 00:27:56.878 INFO  [353849] [ncylinder::increase_z2@122]  _z2 14.865 dz 1 new_z2 15.865
    2022-03-04 00:27:56.878 FATAL [353849] [nzsphere::check@137]  NOT z2_lt_radius  z2 11.035 radius 10.035
    2022-03-04 00:27:56.878 FATAL [353849] [nzsphere::check@159]  tmp skip assert 
    OKX4Test: /data/blyth/junotop/opticks/npy/NNodeNudger.cpp:501: void NNodeNudger::znudge_union_maxmin(NNodeCoincidence*): Assertion `join2 != JOIN_COINCIDENT' failed.
    Aborted (core dumped)
    === geocache-create- : logs are in tmp:/tmp/blyth/opticks/geocache-create-
    N[blyth@localhost npy]$ 


    2022-03-03 16:31:49.711 INFO  [9583911] [ncylinder::increase_z2@122]  _z2 14.865 dz 1 new_z2 15.865
    2022-03-03 16:31:49.711 FATAL [9583911] [nzsphere::check@137]  NOT z2_lt_radius  z2 11.035 radius 10.035
    2022-03-03 16:31:49.711 FATAL [9583911] [nzsphere::check@159]  tmp skip assert 
    Assertion failed: (join2 != JOIN_COINCIDENT), function znudge_union_maxmin, file /Users/blyth/opticks/npy/NNodeNudger.cpp, line 501.
    Process 90800 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff53f5db66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff53f5db66 <+10>: jae    0x7fff53f5db70            ; <+20>
        0x7fff53f5db68 <+12>: movq   %rax, %rdi
        0x7fff53f5db6b <+15>: jmp    0x7fff53f54ae9            ; cerror_nocancel
        0x7fff53f5db70 <+20>: retq   
    Target 0: (OKX4Test) stopped.

    Process 90800 launched: '/usr/local/opticks/lib/OKX4Test' (x86_64)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff53f5db66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff54128080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff53eb91ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff53e811ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010a5c3514 libNPY.dylib`NNodeNudger::znudge_union_maxmin(this=0x000000011ed55690, coin=0x000000011ed54f30) at NNodeNudger.cpp:501
        frame #5: 0x000000010a5c2a31 libNPY.dylib`NNodeNudger::znudge(this=0x000000011ed55690, coin=0x000000011ed54f30) at NNodeNudger.cpp:298
        frame #6: 0x000000010a5c188f libNPY.dylib`NNodeNudger::uncoincide(this=0x000000011ed55690) at NNodeNudger.cpp:285
        frame #7: 0x000000010a5c0b39 libNPY.dylib`NNodeNudger::init(this=0x000000011ed55690) at NNodeNudger.cpp:92
        frame #8: 0x000000010a5c07f7 libNPY.dylib`NNodeNudger::NNodeNudger(this=0x000000011ed55690, root_=0x000000011ed549e0, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:66
        frame #9: 0x000000010a5c0e8d libNPY.dylib`NNodeNudger::NNodeNudger(this=0x000000011ed55690, root_=0x000000011ed549e0, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:64
        frame #10: 0x000000010a628c9d libNPY.dylib`NCSG::MakeNudger(msg="Adopt root ctor", root=0x000000011ed549e0, surface_epsilon=0.00000999999974) at NCSG.cpp:278
        frame #11: 0x000000010a628df2 libNPY.dylib`NCSG::NCSG(this=0x000000011ed55280, root=0x000000011ed549e0) at NCSG.cpp:309
        frame #12: 0x000000010a6280cd libNPY.dylib`NCSG::NCSG(this=0x000000011ed55280, root=0x000000011ed549e0) at NCSG.cpp:324
        frame #13: 0x000000010a627e6d libNPY.dylib`NCSG::Adopt(root=0x000000011ed549e0, config=0x000000011ed54ec0, soIdx=0, lvIdx=0) at NCSG.cpp:173
        frame #14: 0x000000010a627abd libNPY.dylib`NCSG::Adopt(root=0x000000011ed549e0, config_="", soIdx=0, lvIdx=0) at NCSG.cpp:147
        frame #15: 0x000000010a627938 libNPY.dylib`NCSG::Adopt(root=0x000000011ed549e0) at NCSG.cpp:140
        frame #16: 0x00000001038f15ed libExtG4.dylib`X4CSG::X4CSG(this=0x00007ffeefbf9378, solid_=0x000000011644e450, ok_=0x0000000115c3db40) at X4CSG.cc:128
        frame #17: 0x00000001038ef765 libExtG4.dylib`X4CSG::X4CSG(this=0x00007ffeefbf9378, solid_=0x000000011644e450, ok_=0x0000000115c3db40) at X4CSG.cc:132
        frame #18: 0x00000001038f02d7 libExtG4.dylib`X4CSG::GenerateTest(solid=0x000000011644e450, ok=0x0000000115c3db40, prefix="/usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/f9225f882628d01e0303b3609013324e/1/g4codegen", lvidx=100) at X4CSG.cc:78
        frame #19: 0x00000001039df094 libExtG4.dylib`X4PhysicalVolume::GenerateTestG4Code(ok=0x0000000115c3db40, lvIdx=100, solid=0x000000011644e450, raw=0x000000011ed45fa0) at X4PhysicalVolume.cc:1198
        frame #20: 0x00000001039dedf0 libExtG4.dylib`X4PhysicalVolume::ConvertSolid_(ok=0x0000000115c3db40, lvIdx=100, soIdx=100, solid=0x000000011644e450, soname="led-source-shell0xc3068f00x3e9bd80", lvname="/dd/Geometry/CalibrationSources/lvLedSourceShell0xc3066b00x3f226d0", balance_deep_tree=true) at X4PhysicalVolume.cc:1115
        frame #21: 0x00000001039ddf4d libExtG4.dylib`X4PhysicalVolume::ConvertSolid(ok=0x0000000115c3db40, lvIdx=100, soIdx=100, solid=0x000000011644e450, soname="led-source-shell0xc3068f00x3e9bd80", lvname="/dd/Geometry/CalibrationSources/lvLedSourceShell0xc3066b00x3f226d0") at X4PhysicalVolume.cc:1015
        frame #22: 0x00000001039dca08 libExtG4.dylib`X4PhysicalVolume::convertSolid(this=0x00007ffeefbfdea8, lv=0x000000011523a100) at X4PhysicalVolume.cc:962
        frame #23: 0x00000001039db115 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x000000011523aa90, depth=13) at X4PhysicalVolume.cc:923
        frame #24: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x000000011523fa70, depth=12) at X4PhysicalVolume.cc:917
        frame #25: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x00000001152404f0, depth=11) at X4PhysicalVolume.cc:917
        frame #26: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x0000000115240870, depth=10) at X4PhysicalVolume.cc:917
        frame #27: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x00000001152479b0, depth=9) at X4PhysicalVolume.cc:917
        frame #28: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x0000000115249b10, depth=8) at X4PhysicalVolume.cc:917
        frame #29: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x00000001162f4060, depth=7) at X4PhysicalVolume.cc:917
        frame #30: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x00000001162f67e0, depth=6) at X4PhysicalVolume.cc:917
        frame #31: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x000000010f07eac0, depth=5) at X4PhysicalVolume.cc:917
        frame #32: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x000000010f07f840, depth=4) at X4PhysicalVolume.cc:917
        frame #33: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x000000010f080e50, depth=3) at X4PhysicalVolume.cc:917
        frame #34: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x000000010f081c50, depth=2) at X4PhysicalVolume.cc:917
        frame #35: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x000000010f081a10, depth=1) at X4PhysicalVolume.cc:917
        frame #36: 0x00000001039dae34 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfdea8, pv=0x000000010f084200, depth=0) at X4PhysicalVolume.cc:917
        frame #37: 0x00000001039d69c8 libExtG4.dylib`X4PhysicalVolume::convertSolids(this=0x00007ffeefbfdea8) at X4PhysicalVolume.cc:879
        frame #38: 0x00000001039d57af libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfdea8) at X4PhysicalVolume.cc:202
        frame #39: 0x00000001039d545f libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfdea8, ggeo=0x0000000115c60f80, top=0x000000010f084200) at X4PhysicalVolume.cc:181
        frame #40: 0x00000001039d45f5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfdea8, ggeo=0x0000000115c60f80, top=0x000000010f084200) at X4PhysicalVolume.cc:172
        frame #41: 0x00000001000156e6 OKX4Test`main(argc=13, argv=0x00007ffeefbfe6b8) at OKX4Test.cc:108
        frame #42: 0x00007fff53e0d015 libdyld.dylib`start + 1
        frame #43: 0x00007fff53e0d015 libdyld.dylib`start + 1
    (lldb) 


    (lldb) f 4
    frame #4: 0x000000010a5c3514 libNPY.dylib`NNodeNudger::znudge_union_maxmin(this=0x000000011ed55690, coin=0x000000011ed54f30) at NNodeNudger.cpp:501
       498 	    float zj2 = jbb2.min.z ;
       499 	 
       500 	    NNodeJoinType join2 = NNodeEnum::JoinClassify( zi2, zj2, epsilon );
    -> 501 	    assert(join2 != JOIN_COINCIDENT);
       502 	
       503 	    coin->fixed = true ; 
       504 	
    (lldb) 


Huh : having to add lots of x4nudgeskip to get the conversion thru::

    geocache-;DEBUG=1 geocache-create


::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff53f5db66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff54128080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff53eb91ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff53e811ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000109ae4ccf libGGeo.dylib`GBndLib::add(this=0x0000000116478790, omat_="/dd/Materials/Vacuum", osur_=0x0000000000000000, isur_=0x0000000000000000, imat_="/dd/Materials/Vacuum") at GBndLib.cc:508
        frame #5: 0x0000000109ae478e libGGeo.dylib`GBndLib::addBoundary(this=0x0000000116478790, omat="/dd/Materials/Vacuum", osur=0x0000000000000000, isur=0x0000000000000000, imat="/dd/Materials/Vacuum") at GBndLib.cc:470
        frame #6: 0x00000001039e260d libExtG4.dylib`X4PhysicalVolume::addBoundary(this=0x00007ffeefbfde88, pv=0x0000000116651000, pv_p=0x0000000000000000) at X4PhysicalVolume.cc:1598
        frame #7: 0x00000001039e028a libExtG4.dylib`X4PhysicalVolume::convertNode(this=0x00007ffeefbfde88, pv=0x0000000116651000, parent=0x0000000000000000, depth=0, pv_p=0x0000000000000000, recursive_select=0x00007ffeefbfd083) at X4PhysicalVolume.cc:1674
        frame #8: 0x00000001039e003d libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfde88, pv=0x0000000116651000, parent=0x0000000000000000, depth=0, parent_pv=0x0000000000000000, recursive_select=0x00007ffeefbfd083) at X4PhysicalVolume.cc:1409
        frame #9: 0x00000001039d6dcc libExtG4.dylib`X4PhysicalVolume::convertStructure(this=0x00007ffeefbfde88) at X4PhysicalVolume.cc:1336
        frame #10: 0x00000001039d57bb libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfde88) at X4PhysicalVolume.cc:203
        frame #11: 0x00000001039d545f libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfde88, ggeo=0x00000001164784a0, top=0x0000000116651000) at X4PhysicalVolume.cc:181
        frame #12: 0x00000001039d45f5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfde88, ggeo=0x00000001164784a0, top=0x0000000116651000) at X4PhysicalVolume.cc:172
        frame #13: 0x00000001000156e6 OKX4Test`main(argc=14, argv=0x00007ffeefbfe698) at OKX4Test.cc:108
        frame #14: 0x00007fff53e0d015 libdyld.dylib`start + 1
        frame #15: 0x00007fff53e0d015 libdyld.dylib`start + 1
    (lldb) 



cg with dx1
-------------

::

    2022-03-03 17:40:11.116 INFO  [9727616] [*CSG_GGeo_Convert::convertNode@570]  primIdx 1641 partIdxRel   14 tag     ze tc       zero tranIdx    0 is_list not_list subNum   -1 subOffset   -1
    2022-03-03 17:40:11.116 INFO  [9727616] [*CSG_GGeo_Convert::convertNode@570]  primIdx 1641 partIdxRel   15 tag     cy tc   cylinder tranIdx 1528 is_list not_list subNum   -1 subOffset   -1
    2022-03-03 17:40:11.116 INFO  [9727616] [*CSG_GGeo_Convert::convertNode@570]  primIdx 1641 partIdxRel   16 tag     cy tc   cylinder tranIdx 1529 is_list not_list subNum   -1 subOffset   -1
    2022-03-03 17:40:11.116 INFO  [9727616] [*CSG_GGeo_Convert::convertNode@570]  primIdx 1641 partIdxRel   17 tag     co tc convexpolyhedron tranIdx 1530 is_list not_list subNum   -1 subOffset   -1
    2022-03-03 17:40:11.116 FATAL [9727616] [*CSG_GGeo_Convert::GetPlanes@662]  unexpected pl_buf 672,4
    Assertion failed: (pl_expect), function GetPlanes, file /Users/blyth/opticks/CSG_GGeo/CSG_GGeo_Convert.cc, line 666.
    ./run.sh: line 98:  8660 Abort trap: 6           $GDB $bin $GDBDIV $*
    epsilon:CSG_GGeo blyth$ 
    epsilon:CSG_GGeo blyth$ 


    (lldb) f 5
    frame #5: 0x00000001000fd286 libCSG_GGeo.dylib`CSG_GGeo_Convert::convertNode(this=0x00007ffeefbfded8, comp=0x0000000101ffb610, primIdx=1641, partIdxRel=17) at CSG_GGeo_Convert.cc:585
       582 	    bool complement = comp->getComplement(partIdx);
       583 	
       584 	    bool has_planes = CSG::HasPlanes(tc); 
    -> 585 	    std::vector<float4>* planes = has_planes ? GetPlanes(comp, primIdx, partIdxRel) : nullptr ; 
       586 	
       587 	    const float* aabb = nullptr ;  
       588 	    CSGNode nd = CSGNode::Make(tc, param6, aabb ) ; 
    (lldb) p tc
    (unsigned int) $0 = 112
    (lldb) p CSG::Name(tc)
    (const char *) $1 = 0x00000001001283fd "convexpolyhedron"
    (lldb) p primIdx
    (unsigned int) $2 = 1641
    (lldb) p partIdxRel
    (unsigned int) $3 = 17





TODO : check with an old JUNO gdml
-------------------------------------




review gparts_transform_offset
----------------------------------

::

    epsilon:CSG_GGeo blyth$ opticks-f gparts_transform_offset
    ./okop/OpMgr.cc:    bool is_gparts_transform_offset = m_ok->isGPartsTransformOffset()  ; 
    ./okop/OpMgr.cc:    LOG(info) << " is_gparts_transform_offset " << is_gparts_transform_offset ; 

    ./opticksgeo/OpticksHub.cc:    bool is_gparts_transform_offset = m_ok->isGPartsTransformOffset()  ; 
    ./opticksgeo/OpticksHub.cc:    LOG(info) << "[ " << m_ok->getIdPath() << " isGPartsTransformOffset " << is_gparts_transform_offset  ; 

    ./GeoChain/tests/GeoChainNodeTest.cc:    const char* argforced = "--allownokey --gparts_transform_offset" ; 
    ./GeoChain/tests/GeoChainVolumeTest.cc:    const char* argforced = "--allownokey --gparts_transform_offset" ; 
    ./GeoChain/tests/GeoChainVolumeTest.cc:    // see notes/issues/PMT_body_phys_bizarre_innards_confirmed_fixed_by_using_gparts_transform_offset_option.rst
    ./GeoChain/tests/GeoChainSolidTest.cc:    const char* argforced = "--allownokey --gparts_transform_offset" ; 

    ./ggeo/GParts.cc:Notice the --gparts_transform_offset option which 
    ./ggeo/GParts.cc:are handled separately, hence --gparts_transform_offset
    ./ggeo/GParts.cc:    if(m_ok && m_ok->isGPartsTransformOffset())  // --gparts_transform_offset
    ./ggeo/GParts.cc:        if(dump) LOG(info) << " --gparts_transform_offset IS ENABLED, COUNT  " << COUNT  ; 
    ./ggeo/GParts.cc:        if(dump) LOG(info) << " NOT ENABLED --gparts_transform_offset, COUNT  " << COUNT  ; 

    ./optickscore/tests/OpticksTest.cc:    bool is_gparts_transform_offset = ok->isGPartsTransformOffset(); 
    ./optickscore/tests/OpticksTest.cc:    LOG(info) << " is_gparts_transform_offset " << is_gparts_transform_offset ; 

    ./optickscore/Opticks.hh:       bool isGPartsTransformOffset() const ; // --gparts_transform_offset
    ./optickscore/OpticksCfg.cc:       ("gparts_transform_offset",  "see GParts::add") ;
    ./optickscore/Opticks.cc:    return m_cfg->hasOpt("gparts_transform_offset") ;  

    ./optixrap/OGeo.cc:    bool is_gparts_transform_offset = m_ok->isGPartsTransformOffset()  ;   
    ./optixrap/OGeo.cc:    LOG(info) << " is_gparts_transform_offset " << is_gparts_transform_offset ; 
    ./optixrap/OGeo.cc:    if( is_gparts_transform_offset )
    ./optixrap/OGeo.cc:           << " using the old pre7 optixrap machinery with option --gparts_transform_offset enabled will result in mangled transforms " ; 
    ./optixrap/OGeo.cc:           << " the --gparts_transform_offset is only appropriate when using the new optix7 machinery, eg CSG/CSGOptiX/CSG_GGeo/.. " ; 

    ./CSG_GGeo/tests/CSG_GGeoTest.cc:    const char* argforced = "--gparts_transform_offset" ; 
    ./CSG_GGeo/run.sh:--gparts_transform_offset
    ./CSG_GGeo/run.sh:Hmm without "--gparts_transform_offset" get messed up geometry 
    ./CSG_GGeo/run.sh:    epsilon:CSG_GGeo blyth$ opticks-f gparts_transform_offset 
    ./CSG_GGeo/run.sh:    ./ggeo/GParts.cc:    if(m_ok && m_ok->isGPartsTransformOffset())  // --gparts_transform_offset
    ./CSG_GGeo/run.sh:    ./ggeo/GParts.cc:        LOG(LEVEL) << " --gparts_transform_offset " ; 
    ./CSG_GGeo/run.sh:    ./ggeo/GParts.cc:        LOG(LEVEL) << " NOT --gparts_transform_offset " ; 
    ./CSG_GGeo/run.sh:    ./optickscore/Opticks.hh:       bool isGPartsTransformOffset() const ; // --gparts_transform_offset
    ./CSG_GGeo/run.sh:    ./optickscore/OpticksCfg.cc:       ("gparts_transform_offset",  "see GParts::add") ;
    ./CSG_GGeo/run.sh:    ./optickscore/Opticks.cc:    return m_cfg->hasOpt("gparts_transform_offset") ;  
    ./CSG_GGeo/run.sh:    1266 Notice the --gparts_transform_offset option which 
    ./CSG_GGeo/run.sh:    1272 are handled separately, hence --gparts_transform_offset
    ./CSG_GGeo/run.sh:    1297     if(m_ok && m_ok->isGPartsTransformOffset())  // --gparts_transform_offset
    ./CSG_GGeo/run.sh:    1299         LOG(LEVEL) << " --gparts_transform_offset " ;
    ./CSG_GGeo/run.sh:    1307         LOG(LEVEL) << " NOT --gparts_transform_offset " ;
    ./CSG_GGeo/CSG_GGeo_Convert.cc:    bool gparts_transform_offset = ok->isGPartsTransformOffset() ; 
    ./CSG_GGeo/CSG_GGeo_Convert.cc:    if(!gparts_transform_offset)
    ./CSG_GGeo/CSG_GGeo_Convert.cc:            << " GParts geometry requires use of --gparts_transform_offset "
    ./CSG_GGeo/CSG_GGeo_Convert.cc:    assert(gparts_transform_offset); 
    ./CSG_GGeo/run1.sh:./run.sh --gparts_transform_offset 
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 


::

    GGeo::deferredCreateGParts
    GParts::Create




Huh thats unreasonable the tranOffset should not go down::

    2022-03-03 20:33:35.570 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4471 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.570 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4472 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.570 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4473 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.570 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4474 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.570 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4475 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.570 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4476 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.570 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4477 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.570 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4478 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.571 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4479 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.571 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4480 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.571 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4481 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.571 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4482 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.571 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4483 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.571 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4484 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.571 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4485 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.572 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4486 ridx 0 tranOffset 5740
    2022-03-03 20:33:35.856 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4487 ridx 1 tranOffset 0
    2022-03-03 20:33:35.856 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4488 ridx 2 tranOffset 0
    2022-03-03 20:33:35.856 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4489 ridx 3 tranOffset 0
    2022-03-03 20:33:35.857 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4490 ridx 4 tranOffset 0
    2022-03-03 20:33:35.857 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4491 ridx 5 tranOffset 0
    2022-03-03 20:33:35.857 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4492 ridx 5 tranOffset 4
    2022-03-03 20:33:35.858 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4493 ridx 5 tranOffset 8
    2022-03-03 20:33:35.858 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4494 ridx 5 tranOffset 10
    2022-03-03 20:33:35.858 INFO  [9919390] [GParts::add@1322]  --gparts_transform_offset IS ENABLED, COUNT  4495 ridx 5 tranOffset 11
    2022-03-03 20:33:35.859 ERROR [9919390] [main@26] ] load ggeo 
    epsilon:CSG_GGeo blyth$ 


Actually it is OK, see CSG_GGeo_Convert::convertNode.::


    542 CSGNode* CSG_GGeo_Convert::convertNode(const GParts* comp, unsigned primIdx, unsigned partIdxRel )
    543 {
    544     unsigned repeatIdx = comp->getRepeatIndex();  // set in GGeo::deferredCreateGParts
    545     unsigned partOffset = comp->getPartOffset(primIdx) ;
    546     unsigned partIdx = partOffset + partIdxRel ;
    547     unsigned idx = comp->getIndex(partIdx);
    548     assert( idx == partIdx );
    549     unsigned boundary = comp->getBoundary(partIdx); // EXPT
    550 
    551     std::string tag = comp->getTag(partIdx);
    552     unsigned tc = comp->getTypeCode(partIdx);
    553     bool is_list = CSG::IsList((OpticksCSG_t)tc) ;
    554     int subNum = is_list ? comp->getSubNum(partIdx) : -1 ;
    555     int subOffset = is_list ? comp->getSubOffset(partIdx) : -1 ;
    556 
    557 
    558     // TODO: transform handling in double, narrowing to float at the last possible moment 
    559     const Tran<float>* tv = nullptr ;
    560     unsigned gtran = comp->getGTransform(partIdx);  // 1-based index, 0 means None
    561     if( gtran > 0 )
    562     {
    563         glm::mat4 t = comp->getTran(gtran-1,0) ;
    564         glm::mat4 v = comp->getTran(gtran-1,1);
    565         tv = new Tran<float>(t, v);
    566     }
    567 
    568     unsigned tranIdx = tv ?  1 + foundry->addTran(tv) : 0 ;   // 1-based index referencing foundry transforms
    569 




::

    epsilon:ggeo blyth$ ./GPartsTest.sh 
    Fold : loading from base /tmp/blyth/opticks/GParts/0 setting globals False globals_prefix  
                   GParts :           14700 : 2022-03-03 22:03:19.280775 :  GParts.txt 
                idxBuffer :       (4486, 4) : 2022-03-03 22:03:19.277043 :  idxBuffer.npy 
               planBuffer :        (672, 4) : 2022-03-03 22:03:19.276789 :  planBuffer.npy 
               partBuffer :   (14700, 4, 4) : 2022-03-03 22:03:19.275552 :  partBuffer.npy 
               tranBuffer : (5745, 3, 4, 4) : 2022-03-03 22:03:19.276539 :  tranBuffer.npy 
               primBuffer :       (4486, 4) : 2022-03-03 22:03:19.277272 :  primBuffer.npy 
     min_stamp : 2022-03-03 22:03:19.275552 
     max_stamp : 2022-03-03 22:03:19.280775 
     dif_stamp : 0:00:00.005223 
    Fold : loading from base /tmp/blyth/opticks/GParts/1 setting globals False globals_prefix  
                   GParts :               1 : 2022-03-03 22:03:19.281851 :  GParts.txt 
                idxBuffer :          (1, 4) : 2022-03-03 22:03:19.281489 :  idxBuffer.npy 
               partBuffer :       (1, 4, 4) : 2022-03-03 22:03:19.281333 :  partBuffer.npy 
               primBuffer :          (1, 4) : 2022-03-03 22:03:19.281656 :  primBuffer.npy 
     min_stamp : 2022-03-03 22:03:19.281333 
     max_stamp : 2022-03-03 22:03:19.281851 
     dif_stamp : 0:00:00.000518 
    Fold : loading from base /tmp/blyth/opticks/GParts/2 setting globals False globals_prefix  
                   GParts :               1 : 2022-03-03 22:03:19.282868 :  GParts.txt 
                idxBuffer :          (1, 4) : 2022-03-03 22:03:19.282533 :  idxBuffer.npy 
               partBuffer :       (1, 4, 4) : 2022-03-03 22:03:19.282359 :  partBuffer.npy 
               primBuffer :          (1, 4) : 2022-03-03 22:03:19.282699 :  primBuffer.npy 
     min_stamp : 2022-03-03 22:03:19.282359 
     max_stamp : 2022-03-03 22:03:19.282868 
     dif_stamp : 0:00:00.000509 
    Fold : loading from base /tmp/blyth/opticks/GParts/3 setting globals False globals_prefix  
                   GParts :               1 : 2022-03-03 22:03:19.283844 :  GParts.txt 
                idxBuffer :          (1, 4) : 2022-03-03 22:03:19.283465 :  idxBuffer.npy 
               partBuffer :       (1, 4, 4) : 2022-03-03 22:03:19.283295 :  partBuffer.npy 
               primBuffer :          (1, 4) : 2022-03-03 22:03:19.283631 :  primBuffer.npy 
     min_stamp : 2022-03-03 22:03:19.283295 
     max_stamp : 2022-03-03 22:03:19.283844 
     dif_stamp : 0:00:00.000549 
    Fold : loading from base /tmp/blyth/opticks/GParts/4 setting globals False globals_prefix  
                   GParts :               1 : 2022-03-03 22:03:19.284859 :  GParts.txt 
                idxBuffer :          (1, 4) : 2022-03-03 22:03:19.284510 :  idxBuffer.npy 
               partBuffer :       (1, 4, 4) : 2022-03-03 22:03:19.284321 :  partBuffer.npy 
               primBuffer :          (1, 4) : 2022-03-03 22:03:19.284678 :  primBuffer.npy 
     min_stamp : 2022-03-03 22:03:19.284321 
     max_stamp : 2022-03-03 22:03:19.284859 
     dif_stamp : 0:00:00.000538 
    Fold : loading from base /tmp/blyth/opticks/GParts/5 setting globals False globals_prefix  
                   GParts :              41 : 2022-03-03 22:03:19.287721 :  GParts.txt 
                idxBuffer :          (5, 4) : 2022-03-03 22:03:19.287413 :  idxBuffer.npy 
               partBuffer :      (41, 4, 4) : 2022-03-03 22:03:19.287077 :  partBuffer.npy 
               tranBuffer :   (11, 3, 4, 4) : 2022-03-03 22:03:19.287241 :  tranBuffer.npy 
               primBuffer :          (5, 4) : 2022-03-03 22:03:19.287564 :  primBuffer.npy 
     min_stamp : 2022-03-03 22:03:19.287077 
     max_stamp : 2022-03-03 22:03:19.287721 
     dif_stamp : 0:00:00.000644 

    In [1]:                                                         



::

    //    partOffset, numParts, tranOffset, planOffset 

    In [2]: g[0].primBuffer                                                                                                                                                                                   
    Out[2]: 
    array([[    0,     1,     0,     0],
           [    1,     3,     0,     0],
           [    4,     3,     2,     0],
           [    7,    15,     4,     0],
           [   22,     1,     9,     0],
           ...,
           [14681,     1,  5740,   672],
           [14682,     1,  5740,   672],
           [14683,     1,  5740,   672],
           [14684,     1,  5740,   672],
           [14685,    15,  5740,   672]], dtype=int32)





::

    EYE=-1.1,0,0 ./cxr_debug.sh 



P EMM check : after rerun with online-data updated
-----------------------------------------------------

::

    EMM=1, ./cxr_overview.sh   # looks normal 3 inch PMTs : no trouble with instance transforms
    EMM=2, ./cxr_overview.sh   # looks normal hatboxes 
    EMM=3, ./cxr_overview.sh   # looks normal hatboxes 




Havest names
---------------

::

    sChimneyAcrylic0x71a6010
    sChimneyLS0x71a61f0
    sChimneySteel0x71a63d0



::

   MOI=sChimneyAcrylic ./cxr_view.sh 

   MOI=sChimneyAcrylic EYE=-1,0,0 TMIN=0 ./cxr_view.sh 
       looks centrally pointed
 
   MOI=sChimneyAcrylic EYE=-10,0,0 TMIN=0 ./cxr_view.sh 
        this is OK, at top of sphere 

   MOI=sChimneyLS EYE=-10,0,0 TMIN=0 ./cxr_view.sh 
        confusing view

   MOI=sChimneyLS EYE=0,0,1 UP=0,1,0  TMIN=0.1 ./cxr_view.sh 
        try to look down inside the mid-chimney 



    epsilon:offline blyth$ jgr sChimney 
    ./Simulation/DetSimV2/Chimney/src/LowerChimney.cc:    G4Tubs* solidChimneyAcrylic = new G4Tubs("sChimneyAcrylic",
    ./Simulation/DetSimV2/Chimney/src/LowerChimney.cc:    G4Tubs* solidChimneyLS = new G4Tubs("sChimneyLS",
    ./Simulation/DetSimV2/Chimney/src/LowerChimney.cc:    G4Tubs* solidChimneySteel = new G4Tubs("sChimneySteel",





From geocache-02mar2022-key
-------------------------------

Start new session to get the bashrc defined OPTICKS_KEY 


::


    epsilon:~ blyth$ CSGTargetTest
    2022-03-04 19:48:43.239 INFO  [490190] [CSGTargetTest::CSGTargetTest@56] cfbase /usr/local/opticks/geocache/OKX4Test_lWorld0x5780b30_PV_g4live/g4ok_gltf/5303cd587554cb16682990189831ae83/1/CSG_GGeo
    2022-03-04 19:48:43.240 INFO  [490190] [CSGTargetTest::CSGTargetTest@57] foundry CSGFoundry  total solids 10 STANDARD 10 ONE_PRIM 0 ONE_NODE 0 DEEP_COPY 0 KLUDGE_BBOX 0 num_prim 3248 num_node 23518 num_plan 0 num_tran 7228 num_itra 7228 num_inst 48477 ins 0 gas 0 ias 0 meshname 141 mmlabel 10
    2022-03-04 19:48:43.240 INFO  [490190] [CSGTargetTest::dumpALL@119]  fd.getNumPrim 3248 fd.meshname.size 141
     primIdx    0 lce (        0.00       0.00       0.00   60000.00 ) lce.w/1000        60.00 meshIdx  138 sWorld0x577e4d0
     primIdx    1 lce (        0.00       0.00       0.00   31125.00 ) lce.w/1000        31.12 meshIdx   17 sTopRock0x578c0a0
     primIdx    2 lce (        0.00       0.00       0.00   31125.00 ) lce.w/1000        31.12 meshIdx    2 sDomeRockBox0x578c210
     primIdx    3 lce (     3125.00       0.00   21990.00   31125.00 ) lce.w/1000        31.12 meshIdx    1 sTopRock_dome0x578c520
     primIdx    4 lce (     3125.00       0.00   21990.00   28125.00 ) lce.w/1000        28.12 meshIdx    0 sTopRock_domeAir0x578ca70
     primIdx    5 lce (        0.00       0.00       0.00   31125.00 ) lce.w/1000        31.12 meshIdx   16 sExpRockBox0x578ce00
     primIdx    6 lce (        0.00       0.00   27250.00   31250.00 ) lce.w/1000        31.25 meshIdx   15 sExpHall0x578d4f0
     primIdx    7 lce (        0.00       0.00   21751.00   21800.00 ) lce.w/1000        21.80 meshIdx    3 PoolCoversub0x578d9b0

     primIdx    8 lce (        0.00       0.00       0.00    1750.00 ) lce.w/1000         1.75 meshIdx    7 Upper_Chimney0x71a3800
     primIdx    9 lce (        0.00       0.00       0.00    1750.00 ) lce.w/1000         1.75 meshIdx    4 Upper_LS_tube0x71a38f0

     primIdx   10 lce (        0.00       0.00   21750.00    1750.00 ) lce.w/1000         1.75 meshIdx    5 Upper_Steel_tube0x71a39e0
     primIdx   11 lce (        0.00       0.00   21750.00    1750.00 ) lce.w/1000         1.75 meshIdx    6 Upper_Tyvek_tube0x71a3af0
     primIdx   12 lce (        0.00       0.00   25952.00   24000.00 ) lce.w/1000        24.00 meshIdx   14 sAirTT0x71a76a0
     primIdx   13 lce (        0.00       0.00       0.00    3430.60 ) lce.w/1000         3.43 meshIdx   13 sWall0x71a8b30
     primIdx   14 lce (        0.00       0.00       0.00    3430.60 ) lce.w/1000         3.43 meshIdx   12 sPlane0x71a8bb0
     primIdx   15 lce (        0.00       0.00       0.00    3430.60 ) lce.w/1000         3.43 meshIdx   12 sPlane0x71a8bb0
     primIdx   16 lce (        0.00       0.00       0.00    3430.60 ) lce.w/1000         3.43 meshIdx   13 sWall0x71a8b30
     primIdx   17 lce (        0.00       0.00       0.00    3430.60 ) lce.w/1000         3.43 meshIdx   12 sPlane0x71a8bb0




::

    epsilon:~ blyth$ CSGTargetTest | grep Chimney
     primIdx    8 lce (        0.00       0.00       0.00    1750.00 ) lce.w/1000         1.75 meshIdx    7 Upper_Chimney0x71a3800
     primIdx 3086 lce (        0.00       0.00   18124.00     524.00 ) lce.w/1000         0.52 meshIdx  123 sChimneyAcrylic0x71a6010
     primIdx 3087 lce (        0.00       0.00       0.00    1963.00 ) lce.w/1000         1.96 meshIdx  124 sChimneyLS0x71a61f0
     primIdx 3088 lce (        0.00       0.00   20087.00    1663.00 ) lce.w/1000         1.66 meshIdx  125 sChimneySteel0x71a63d0
    epsilon:~ blyth$ 




CSG/tests/CSGPrimTest.py 
--------------------------


::

    prim_numNode = cf.prim.view(np.int32)[:,0,0]

    In [30]: np.unique( primNode, return_counts=True )                                                                                                                                                        
    Out[30]: 
    (array([  1,   3,   7,  15,  31, 127], dtype=int32),
     array([ 931,  118, 2130,   12,    1,   56]))




    In [43]: np.unique(prim_repeatIdx , return_counts=True )                                                                                                                                                  
    Out[43]: 
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32),
     array([3089,    5,    7,    7,    6,    1,    1,    1,    1,  130]))


    In [56]: cf.solid.shape                                                                                                                                                                                   
    Out[56]: (10, 3, 4)

    In [58]: solid_numPrim = cf.solid[:,1,0]                                                                                                                                                                  

    In [59]: solid_numPrim                                                                                                                                                                                    
    Out[59]: array([3089,    5,    7,    7,    6,    1,    1,    1,    1,  130], dtype=int32)




    In [53]: cf.prim.shape                                                                                                                                                                                    
    Out[53]: (3248, 4, 4)

    In [54]: cf.node.shape                                                                                                                                                                                    
    Out[54]: (23518, 4, 4)

    In [51]: prim_primIdx[3080:]                                                                                                                                                                              
    Out[51]: 
    array([3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088,    0,    1,    2,    3,    4,    0,    1,    2,    3,    4,    5,    6,    0,    1,    2,    3,    4,    5,    6,    0,    1,    2,    3,
              4,    5,    0,    0,    0,    0,    0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,
             26,   27,   28,   29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,
             58,   59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   84,   85,   86,   87,   88,   89,
             90,   91,   92,   93,   94,   95,   96,   97,   98,   99,  100,  101,  102,  103,  104,  105,  106,  107,  108,  109,  110,  111,  112,  113,  114,  115,  116,  117,  118,  119,  120,  121,
            122,  123,  124,  125,  126,  127,  128,  129], dtype=int32)



ridx 0 
--------


::


    In [5]: midx = prim_meshIdx_(ridx_prims)                                                                                                                                                                  

    In [6]: midx                                                                                                                                                                                              
    Out[6]: array([138,  17,   2, ..., 123, 124, 125], dtype=int32)

    In [8]: np.unique(midx, return_counts=True)                                                                                                                                                               
    Out[8]: 
    (array([  0,   1,   2,   3,   4,   5,   6,   7,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
             42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
             80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97, 102, 103, 123, 124, 125, 126, 127, 128, 135, 136, 137, 138], dtype=int32),
     array([  1,   1,   1,   1,   1,   1,   1,   1, 126,  63,   1,   1,   1,   1,  10,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  10,  30,  30,
             30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,
             30,  30,  30,  30,  30,  30,  30,  30,  30,  30,   2,  36,   8,   8,   1,   1, 370, 220,  56,  56,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]))

    In [9]:                    



This bbox looks wrong::

    In [17]: midx_prims_(8).reshape(-1,16)                                                                                                                                                                    
    Out[17]: 
    array([[    0.,     0.,     0.,     0.,     0.,     0.,     0.,     0., -3430.,   -13.,    -5.,  3430.,    13.,     5.,     0.,     0.],
           [    0.,     0.,     0.,     0.,     0.,     0.,     0.,     0., -3430.,   -13.,    -5.,  3430.,    13.,     5.,     0.,     0.],
           [    0.,     0.,     0.,     0.,     0.,     0.,     0.,     0., -3430.,   -13.,    -5.,  3430.,    13.,     5.,     0.,     0.],
           [    0.,     0.,     0.,     0.,     0.,     0.,     0.,     0., -3430.,   -13.,    -5.,  3430.,    13.,     5.,     0.,     0.],
           [    0.,     0.,     0.,     0.,     0.,     0.,     0.,     0., -3430.,   -13.,    -5.,  3430.,    13.,     5.,     0.,     0.],
           [    0.,     0.,     0.,     0.,     0.,     0.,     0.,     0., -3430.,   -13.,    -5.,  3430.,    13.,     5.,     0.,     0.],
           [    0.,     0.,     0.,     0.,     0.,     0.,     0.,     0., -3430.,   -13.,    -5.,  3430.,    13.,     5.,     0.,     0.],





   92    8 :                                     solidSJReceiver0x5964f50 : (8, 4, 4) 
   93    8 :                              solidSJReceiverFastern0x5969570 : (8, 4, 4) 
   94    1 :                                             sTarget0x5829bb0 : (1, 4, 4) 
   95    1 :                                            sAcrylic0x5829590 : (1, 4, 4) 
   96  370 :                                              sStrut0x582c680 : (370, 4, 4) 
   97  220 :                                              sStrut0x5880150 : (220, 4, 4) 
  102   56 :                                       solidXJanchor0x5933100 : (56, 4, 4) 
  103   56 :                                      solidXJfixture0x595f360 : (56, 4, 4) 
  123    1 :                                     sChimneyAcrylic0x71a6010 : (1, 4, 4) 
  124    1 :                                          sChimneyLS0x71a61f0 : (1, 4, 4) 
  125    1 :                                       sChimneySteel0x71a63d0 : (1, 4, 4) 
  126    1 :                                          sWaterTube0x71a5e30 : (1, 4, 4) 
  127    1 :                                         sInnerWater0x5828f70 : (1, 4, 4) 
  128    1 :                                      sReflectorInCD0x58289b0 : (1, 4, 4) 
  135    1 :                                     sOuterWaterPool0x5792bd0 : (1, 4, 4) 
  136    1 :                                         sPoolLining0x57924c0 : (1, 4, 4) 
  137    1 :                                         sBottomRock0x578e0e0 : (1, 4, 4) 
  138    1 :                                              sWorld0x577e4d0 : (1, 4, 4) 



These supposedly global bbox look local


In [3]: midx_prims_(96).reshape(-1,16)                                                                                                                                                                    
Out[3]: 
array([[   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],

In [4]: midx_prims_(97).reshape(-1,16)                                                                                                                                                                    
Out[4]: 
array([[   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       ...,
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ],
       [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,  -30.   ,  -30.   , -774.039,   30.   ,   30.   ,  774.039,    0.   ,    0.   ]], dtype=float32)

In [5]:                                              



Compare with an old geometry::


    In [6]: midx_prims_(8)[:,2:].reshape(-1,8)                                                                                                                                                               
    Out[6]: 
    array([[ 16703.   , -10096.35 ,  23483.2  ,  23564.2  ,  -3326.05 ,  23496.5  ,      0.   ,      0.   ],
           [ 16748.45 , -10141.801,  23497.5  ,  23518.75 ,  -3280.6  ,  23510.8  ,      0.   ,      0.   ],
           [  9991.801, -10096.35 ,  23433.2  ,  16853.   ,  -3326.05 ,  23446.5  ,      0.   ,      0.   ],
           [ 10037.25 , -10141.801,  23447.5  ,  16807.55 ,  -3280.6  ,  23460.8  ,      0.   ,      0.   ],
           [  3280.6  , -10096.35 ,  23483.2  ,  10141.801,  -3326.05 ,  23496.5  ,      0.   ,      0.   ],
           [  3326.05 , -10141.801,  23497.5  ,  10096.35 ,  -3280.6  ,  23510.8  ,      0.   ,      0.   ],
           [ -3430.6  , -10096.35 ,  23433.2  ,   3430.6  ,  -3326.05 ,  23446.5  ,      0.   ,      0.   ],
           [ -3385.15 , -10141.801,  23447.5  ,   3385.15 ,  -3280.6  ,  23460.8  ,      0.   ,      0.   ],
           [-10141.801, -10096.35 ,  23483.2  ,  -3280.6  ,  -3326.05 ,  23496.5  ,      0.   ,      0.   ],
           [-10096.35 , -10141.801,  23497.5  ,  -3326.05 ,  -3280.6  ,  23510.8  ,      0.   ,      0.   ],
           [-16853.   , -10096.35 ,  23433.2  ,  -9991.801,  -3326.05 ,  23446.5  ,      0.   ,      0.   ],
           [-16807.55 , -10141.801,  23447.5  , -10037.25 ,  -3280.6  ,  23460.8  ,      0.   ,      0.   ],
           [-23564.2  , -10096.35 ,  23483.2  , -16703.   ,  -3326.05 ,  23496.5  ,      0.   ,      0.   ],
           [-23518.75 , -10141.801,  23497.5  , -16748.45 ,  -3280.6  ,  23510.8  ,      0.   ,      0.   ],
           [ 16703.   ,  -3385.15 ,  23383.2  ,  23564.2  ,   3385.15 ,  23396.5  ,      0.   ,      0.   ],
           [ 16748.45 ,  -3430.6  ,  23397.5  ,  23518.75 ,   3430.6  ,  23410.8  ,      0.   ,      0.   ],
           ...,
           [-23564.2  ,  -3385.15 ,  26383.2  , -16703.   ,   3385.15 ,  26396.5  ,      0.   ,      0.   ],
           [-23518.75 ,  -3430.6  ,  26397.5  , -16748.45 ,   3430.6  ,  26410.8  ,      0.   ,      0.   ],
           [ 16703.   ,   3326.05 ,  26483.2  ,  23564.2  ,  10096.35 ,  26496.5  ,      0.   ,      0.   ],
           [ 16748.45 ,   3280.6  ,  26497.5  ,  23518.75 ,  10141.801,  26510.8  ,      0.   ,      0.   ],
           [  9991.801,   3326.05 ,  26433.2  ,  16853.   ,  10096.35 ,  26446.5  ,      0.   ,      0.   ],
           [ 10037.25 ,   3280.6  ,  26447.5  ,  16807.55 ,  10141.801,  26460.8  ,      0.   ,      0.   ],
           [  3280.6  ,   3326.05 ,  26483.2  ,  10141.801,  10096.35 ,  26496.5  ,      0.   ,      0.   ],
           [  3326.05 ,   3280.6  ,  26497.5  ,  10096.35 ,  10141.801,  26510.8  ,      0.   ,      0.   ],
           [ -3430.6  ,   3326.05 ,  26433.2  ,   3430.6  ,  10096.35 ,  26446.5  ,      0.   ,      0.   ],
           [ -3385.15 ,   3280.6  ,  26447.5  ,   3385.15 ,  10141.801,  26460.8  ,      0.   ,      0.   ],
           [-10141.801,   3326.05 ,  26483.2  ,  -3280.6  ,  10096.35 ,  26496.5  ,      0.   ,      0.   ],
           [-10096.35 ,   3280.6  ,  26497.5  ,  -3326.05 ,  10141.801,  26510.8  ,      0.   ,      0.   ],
           [-16853.   ,   3326.05 ,  26433.2  ,  -9991.801,  10096.35 ,  26446.5  ,      0.   ,      0.   ],
           [-16807.55 ,   3280.6  ,  26447.5  , -10037.25 ,  10141.801,  26460.8  ,      0.   ,      0.   ],
           [-23564.2  ,   3326.05 ,  26483.2  , -16703.   ,  10096.35 ,  26496.5  ,      0.   ,      0.   ],
           [-23518.75 ,   3280.6  ,  26497.5  , -16748.45 ,  10141.801,  26510.8  ,      0.   ,      0.   ]], dtype=float32)

In [7]:                                                                                                               




only 4 prim have tranOffset 0 
--------------------------------

::

    In [7]: np.unique(prim_tranOffset, return_counts=True)                                                                                                                                                    
    Out[7]: 
    (array([   0,    2,    4,    6,    8,   10,   12,   14,   17,   20,   23,   26,   29,   32,   35,   38, ..., 7167, 7173, 7176, 7180, 7186, 7190, 7194, 7197, 7201, 7203, 7205, 7208, 7211, 7221, 7225,
            7228], dtype=int32),
     array([  4,   1,   2,   1,   3,   1,   1, 193,   1,   1,   1,   1,   1,   1,   1,   1, ...,   1,   5,   1,   1,   1,   1,   2,   2,   1,   1,   1,   1,   2,   1,   1, 130]))

    In [8]: toff_prims_ = lambda toff:cf.prim[prim_tranOffset_(cf.prim) == toff]                                                                                                                              

    In [9]: toff_prims_(0)                                                                                                                                                                                    
    Out[9]: 
    array([[[     0.,      0.,      0.,      0.],
            [     0.,      0.,      0.,      0.],
            [-60000., -60000., -60000.,  60000.],
            [ 60000.,  60000.,      0.,      0.]],

           [[     0.,      0.,      0.,      0.],
            [     0.,      0.,      0.,      0.],
            [-31125., -27500., -15000.,  31125.],
            [ 27500.,  15000.,      0.,      0.]],

           [[     0.,      0.,      0.,      0.],
            [     0.,      0.,      0.,      0.],
            [-31125., -27500.,  -9500.,  31125.],
            [ 27500.,   9500.,      0.,      0.]],

           [[     0.,      0.,      0.,      0.],
            [     0.,      0.,      0.,      0.],
            [-28000., -29760.,  -7770.,  34250.],
            [ 29760.,  51750.,      0.,      0.]]], dtype=float32)

    In [10]: prim_meshIdx_(toff_prims_(0))                                                                                                                                                                    
    Out[10]: array([138,  17,   2,   1], dtype=int32)


    In [18]: cf.meshname[prim_meshIdx_(toff_prims_(0))]                                                                                                                                                       
    Out[18]: array(['sWorld0x577e4d0', 'sTopRock0x578c0a0', 'sDomeRockBox0x578c210', 'sTopRock_dome0x578c520'], dtype=object)


sStrut bbox all the same
----------------------------

::

    In [4]: midx_prims_(96)[:,2:].reshape(-1,8)                                                                                                                                                               
    Out[4]: 
    array([[ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           ...,
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ],
           [ -42.5  ,  -42.5  , -774.027,   42.5  ,   42.5  ,  774.027,    0.   ,    0.   ]], dtype=float32)

    In [5]: cf.meshname[96]                                                                                                                                                                                   
    Out[5]: 'sStrut0x582c680'


And tranOffset all same too::

    In [3]: prim_tranOffset_(midx_prims_(96))                                                                                                                                                                 
    Out[3]: 
    array([6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538,
           6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538,
           6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538,
           6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538,
           6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538,
           6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538,
           6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538,
           6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538,
           6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538,
           6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538,
           6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538,
           6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538, 6538], dtype=int32)




Compare with old geometry which has different bbox and a variety of tranOffset::


     c
     ./CSGPrimTest.sh old 


    In [2]: midx_prims_(91)[:,2:].reshape(-1,8)                                                                                                                                                               
    Out[2]: 
    array([[   977.606,   1367.903,  17986.479,   1185.118,   1608.832,  19539.42 ,      0.   ,      0.   ],
           [ -1185.118,   1367.903,  17986.479,   -977.606,   1608.832,  19539.42 ,      0.   ,      0.   ],
           [ -1875.062,   -645.397,  17986.479,  -1624.3  ,   -491.614,  19539.42 ,      0.   ,      0.   ],
           [   -42.5  ,  -1957.747,  17986.479,     42.5  ,  -1721.699,  19539.42 ,      0.   ,      0.   ],
           [  1624.3  ,   -645.397,  17986.479,   1875.062,   -491.614,  19539.42 ,      0.   ,      0.   ],
           [  3569.107,    333.328,  17699.139,   3971.174,    459.187,  19236.584,      0.   ,      0.   ],
           [  3090.581,   1759.811,  17699.139,   3475.463,   2031.097,  19236.584,      0.   ,      0.   ],
           [  2077.667,   2882.006,  17699.139,   2378.812,   3251.811,  19236.584,      0.   ,      0.   ],


    In [3]: prim_tranOffset_(midx_prims_(91))                                                                                                                                                                 
    Out[3]: 
    array([6775, 6776, 6777, 6778, 6779, 6780, 6781, 6782, 6783, 6784, 6785, 6786, 6787, 6788, 6789, 6790, 6791, 6792, 6793, 6794, 6795, 6796, 6797, 6798, 6799, 6800, 6801, 6802, 6803, 6804, 6805, 6806,
           6807, 6808, 6809, 6810, 6811, 6812, 6813, 6814, 6815, 6816, 6817, 6818, 6819, 6820, 6821, 6822, 6823, 6824, 6825, 6826, 6827, 6828, 6829, 6830, 6831, 6832, 6833, 6834, 6835, 6836, 6837, 6838,
           6839, 6840, 6841, 6842, 6843, 6844, 6845, 6846, 6847, 6848, 6849, 6850, 6851, 6852, 6853, 6854, 6855, 6856, 6857, 6858, 6859, 6860, 6861, 6862, 6863, 6864, 6865, 6866, 6867, 6868, 6869, 6870,
           6871, 6872, 6873, 6874, 6875, 6876, 6877, 6878, 6879, 6880, 6881, 6882, 6883, 6884, 6885, 6886, 6887, 6888, 6889, 6890, 6891, 6892, 6893, 6894, 6895, 6896, 6897, 6898, 6899, 6900, 6901, 6902,
           6903, 6904, 6905, 6906, 6907, 6908, 6909, 6910, 6911, 6912, 6913, 6914, 6915, 6916, 6917, 6918, 6919, 6920, 6921, 6922, 6923, 6924, 6925, 6926, 6927, 6928, 6929, 6930, 6931, 6932, 6933, 6934,
           6935, 6936, 6937, 6938, 6939, 6940, 6941, 6942, 6943, 6944, 6945, 6946, 6947, 6948, 6949, 6950, 6951, 6952, 6953, 6954, 6955, 6956, 6957, 6958, 6959, 6960, 6961, 6962, 6963, 6964, 6965, 6966,
           6967, 6968, 6969, 6970, 6971, 6972, 6973, 6974, 6975, 6976, 6977, 6978, 6979, 6980, 6981, 6982, 6983, 6984, 6985, 6986, 6987, 6988, 6989, 6990, 6991, 6992, 6993, 6994, 6995, 6996, 6997, 6998,
           6999, 7000, 7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010, 7011, 7012, 7013, 7014, 7015, 7016, 7017, 7018, 7019, 7020, 7021, 7022, 7023, 7024, 7025, 7026, 7027, 7028, 7029, 7030,
           7031, 7032, 7033, 7034, 7035, 7036, 7037, 7038, 7039, 7040, 7041, 7042, 7043, 7044, 7045, 7046, 7047, 7048, 7049, 7050, 7051, 7052, 7053, 7054, 7055, 7056, 7057, 7058, 7059, 7060, 7061, 7062,
           7063, 7064, 7065, 7066, 7067, 7068, 7069, 7070, 7071, 7072, 7073, 7074, 7075, 7076, 7077, 7078, 7079, 7080, 7081, 7082, 7083, 7084, 7085, 7086, 7087, 7088, 7089, 7090, 7091, 7092, 7093, 7094,
           7095, 7096, 7097, 7098, 7099, 7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109, 7110, 7111, 7112, 7113, 7114, 7115, 7116, 7117, 7118, 7119, 7120, 7121, 7122, 7123, 7124, 7125, 7126,
           7127, 7128, 7129, 7130, 7131, 7132, 7133, 7134, 7135, 7136, 7137, 7138, 7139, 7140, 7141, 7142, 7143, 7144], dtype=int32)



                                                  

Was considering to remove tranOffset. They get set automatically on addPrim::

    1103 CSGPrim* CSGFoundry::addPrim(int num_node, int nodeOffset_ )
    1104 {
    1105     if(!last_added_solid) LOG(fatal) << "must addSolid prior to addPrim" ;
    1106     assert( last_added_solid );
    1107 
    1108     unsigned primOffset = last_added_solid->primOffset ;
    1109     unsigned numPrim = last_added_solid->numPrim ;
    1110 
    1111     unsigned globalPrimIdx = prim.size();
    1112     unsigned localPrimIdx = globalPrimIdx - primOffset ;
    1113 
    1114     bool in_range = localPrimIdx < numPrim ;
    1115     if(!in_range) LOG(fatal)
    1116         << " TOO MANY addPrim FOR SOLID "
    1117         << " localPrimIdx " << localPrimIdx
    1118         << " numPrim " << numPrim
    1119         << " globalPrimIdx " << globalPrimIdx
    1120         << " (must addPrim only up to to the declared numPrim from the addSolid call) "
    1121         ;
    1122 
    1123     assert( in_range );
    1124 
    1125     int nodeOffset = nodeOffset_ < 0 ? int(node.size()) : nodeOffset_ ;
    1126 
    1127     CSGPrim pr = {} ;
    1128 
    1129     pr.setNumNode(num_node) ;
    1130     pr.setNodeOffset(nodeOffset);
    1131     //pr.setSbtIndexOffset(globalPrimIdx) ;  // <--- bug ?
    1132     pr.setSbtIndexOffset(localPrimIdx) ;
    1133 
    1134     pr.setMeshIdx(-1) ;                // metadata, that may be set by caller 
    1135 
    1136     pr.setTranOffset(tran.size());     // HMM are tranOffset and planOffset used now that use global referencing  ?
    1137     pr.setPlanOffset(plan.size());
    1138 
    1139     assert( globalPrimIdx < IMAX );
    1140     prim.push_back(pr);
    1141 

     
This means that the tran vector is not being populated so much during the conversion.

* that suggests problem with GParts::getGTransform 

::


    541 CSGNode* CSG_GGeo_Convert::convertNode(const GParts* comp, unsigned primIdx, unsigned partIdxRel )
    542 {
    543     unsigned repeatIdx = comp->getRepeatIndex();  // set in GGeo::deferredCreateGParts
    544     unsigned partOffset = comp->getPartOffset(primIdx) ;
    545     unsigned partIdx = partOffset + partIdxRel ;
    546     unsigned idx = comp->getIndex(partIdx);
    547     assert( idx == partIdx );
    548     unsigned boundary = comp->getBoundary(partIdx); // EXPT
    549 
    550     std::string tag = comp->getTag(partIdx);
    551     unsigned tc = comp->getTypeCode(partIdx);
    552     bool is_list = CSG::IsList((OpticksCSG_t)tc) ;
    553     int subNum = is_list ? comp->getSubNum(partIdx) : -1 ;
    554     int subOffset = is_list ? comp->getSubOffset(partIdx) : -1 ;
    555 
    556 
    557     // TODO: transform handling in double, narrowing to float at the last possible moment 
    558     const Tran<float>* tv = nullptr ;
    559     unsigned gtran = comp->getGTransform(partIdx);  // 1-based index, 0 means None
    560     if( gtran > 0 )
    561     {
    562         glm::mat4 t = comp->getTran(gtran-1,0) ;
    563         glm::mat4 v = comp->getTran(gtran-1,1);
    564         tv = new Tran<float>(t, v);
    565     }
    566 
    567     unsigned tranIdx = tv ?  1 + foundry->addTran(tv) : 0 ;   // 1-based index referencing foundry transforms
    568 


So need to compare at GParts level (actually GPts).

::

    099     #g[0].plc[np.where( g[0].ipt[:,0] == 96 )[0]]   
    100     
    101     # GPts placement transforms of the ridx solid selected by midx (aka lvIdx)
    102     ridx_midx_plc_ = lambda ridx,midx:g[ridx].plc[np.where( g[ridx].ipt[:,0] == midx)[0]]
    103     



ggeo/GPtsTest.py : compare plc transforms for strut370 : shows now changes
------------------------------------------------------------------------------

::


    Fold : loading from base /usr/local/opticks/geocache/OKX4Test_lWorld0x5780b30_PV_g4live/g4ok_gltf/5303cd587554cb16682990189831ae83/1/GPts/9 setting globals False globals_prefix  
                iptBuffer :        (130, 4) : 2022-03-03 13:40:04.017415 :  iptBuffer.npy 
                plcBuffer :     (130, 4, 4) : 2022-03-03 13:40:04.017593 :  plcBuffer.npy 
                     GPts :             130 : 2022-03-03 13:40:04.017807 :  GPts.txt 
     min_stamp : 2022-03-03 13:40:04.017415 
     max_stamp : 2022-03-03 13:40:04.017807 
     dif_stamp : 0:00:00.000392 
     age_stamp : 1 day, 22:40:34.128125 
    arg:new strut370_midx:96 
    ridx_midx_plc_(0,96).reshape(-1,16)
    [[    -0.947     -0.308      0.098      0.        -0.309      0.951      0.         0.        -0.093     -0.03      -0.995      0.      1749.866    568.566  18764.94       1.   ]
     [     0.        -0.995      0.098      0.        -1.         0.         0.         0.        -0.        -0.098     -0.995      0.         0.      1839.918  18764.94       1.   ]
     [     0.947     -0.308      0.098      0.        -0.309     -0.951     -0.         0.         0.093     -0.03      -0.995      0.     -1749.866    568.566  18764.94       1.   ]
     [     0.585      0.805      0.098      0.         0.809     -0.588     -0.         0.         0.057      0.079     -0.995      0.     -1081.477  -1488.525  18764.94       1.   ]
     [    -0.585      0.805      0.098      0.         0.809      0.588      0.         0.        -0.057      0.079     -0.995      0.      1081.477  -1488.525  18764.94       1.   ]
     [    -0.932     -0.303      0.201      0.        -0.309      0.951     -0.         0.        -0.191     -0.062     -0.98       0.      3605.75    1171.579  18469.82       1.   ]
     [    -0.728     -0.655      0.201      0.        -0.669      0.743     -0.         0.        -0.149     -0.135     -0.98       0.      2817.492   2536.881  18469.82       1.   ]
     [    -0.398     -0.895      0.201      0.        -0.914      0.407      0.         0.        -0.082     -0.184     -0.98       0.      1542.064   3463.534  18469.82       1.   ]
     [    -0.        -0.98       0.201      0.        -1.        -0.         0.         0.        -0.        -0.201     -0.98       0.         0.      3791.31   18469.82       1.   ]

    age_stamp : 1 day, 22:40:34.128125 
    arg:new strut370_midx:96 
    ridx_midx_plc_(0,96).reshape(-1,16)
    [[    -0.947     -0.308      0.098      0.        -0.309      0.951      0.         0.        -0.093     -0.03      -0.995      0.      1749.866    568.566  18764.94       1.   ]
     [     0.        -0.995      0.098      0.        -1.         0.         0.         0.        -0.        -0.098     -0.995      0.         0.      1839.918  18764.94       1.   ]
     [     0.947     -0.308      0.098      0.        -0.309     -0.951     -0.         0.         0.093     -0.03      -0.995      0.     -1749.866    568.566  18764.94       1.   ]
     [     0.585      0.805      0.098      0.         0.809     -0.588     -0.         0.         0.057      0.079     -0.995      0.     -1081.477  -1488.525  18764.94       1.   ]
     [    -0.585      0.805      0.098      0.         0.809      0.588      0.         0.        -0.057      0.079     -0.995      0.      1081.477  -1488.525  18764.94       1.   ]
     [    -0.932     -0.303      0.201      0.        -0.309      0.951     -0.         0.        -0.191     -0.062     -0.98       0.      3605.75    1171.579  18469.82       1.   ]
     [    -0.728     -0.655      0.201      0.        -0.669      0.743     -0.         0.        -0.149     -0.135     -0.98       0.      2817.492   2536.881  18469.82       1.   ]
     [    -0.398     -0.895      0.201      0.        -0.914      0.407      0.         0.        -0.082     -0.184     -0.98       0.      1542.064   3463.534  18469.82       1.   ]




Old chimney placements::


    In [3]: ridx_midx_plc_(0,117)                                                                                                                                                                                                          
    Out[3]: 
    array([[[    1.,     0.,     0.,     0.],
            [    0.,     1.,     0.,     0.],
            [    0.,     0.,     1.,     0.],
            [    0.,     0., 18120.,     1.]]], dtype=float32)

    In [1]: ridx_midx_plc_(0,118)                                                                                                                                                                                                          
    Out[1]: 
    array([[[    1.,     0.,     0.,     0.],
            [    0.,     1.,     0.,     0.],
            [    0.,     0.,     1.,     0.],
            [    0.,     0., 19785.,     1.]]], dtype=float32)

    In [2]: ridx_midx_plc_(0,119)                                                                                                                                                                                                          
    Out[2]: 
    array([[[    1.,     0.,     0.,     0.],
            [    0.,     1.,     0.,     0.],
            [    0.,     0.,     1.,     0.],
            [    0.,     0., 20085.,     1.]]], dtype=float32)


New chimney placements, slight shifts out in Z::

    In [1]: ridx_midx_plc_(0,123)                                                                                                                                                                                                          
    Out[1]: 
    array([[[    1.,     0.,     0.,     0.],
            [    0.,     1.,     0.,     0.],
            [    0.,     0.,     1.,     0.],
            [    0.,     0., 18124.,     1.]]], dtype=float32)

    In [2]: ridx_midx_plc_(0,124)                                                                                                                                                                                                          
    Out[2]: 
    array([[[    1.,     0.,     0.,     0.],
            [    0.,     1.,     0.,     0.],
            [    0.,     0.,     1.,     0.],
            [    0.,     0., 19787.,     1.]]], dtype=float32)

    In [3]: ridx_midx_plc_(0,125)                                                                                                                                                                                                          
    Out[3]: 
    array([[[    1.,     0.,     0.,     0.],
            [    0.,     1.,     0.,     0.],
            [    0.,     0.,     1.,     0.],
            [    0.,     0., 20087.,     1.]]], dtype=float32)

    In [4]:                                                                                                         





CSG/CSGPrimTest.sh  old and new : clear lack of global placement
------------------------------------------------------------------

::

     all_ridxs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   ridxs:(0,)   nmame_skip:Flange   geocache_hookup_arg:new 

     ridx :  0   ridx_prims.shape (3089, 4, 4) 
     u_mx c_mx :                        unique midx prim counts and meshname  : prs.shape 
        0    1 :                                    sTopRock_domeAir0x578ca70 :            (1, 4, 4) : [-25000. -26760.  -4770.  31250.  26760.  48750.      0.      0.]  
        1    1 :                                       sTopRock_dome0x578c520 :            (1, 4, 4) : [-28000. -29760.  -7770.  34250.  29760.  51750.      0.      0.]  
        2    1 :                                        sDomeRockBox0x578c210 :            (1, 4, 4) : [-31125. -27500.  -9500.  31125.  27500.   9500.      0.      0.]  
        3    1 :                                        PoolCoversub0x578d9b0 :            (1, 4, 4) : [-21800. -21800.  21749.  21800.  21800.  21753.      0.      0.]  
        4    1 :                                       Upper_LS_tube0x71a38f0 :   PLC?     (1, 4, 4) : [ -400.  -400. -1750.   400.   400.  1750.     0.     0.]  
        5    1 :                                    Upper_Steel_tube0x71a39e0 :            (1, 4, 4) : [ -407.  -407. 20000.   407.   407. 23500.     0.     0.]  
        6    1 :                                    Upper_Tyvek_tube0x71a3af0 :            (1, 4, 4) : [ -402.  -402. 20000.   402.   402. 23500.     0.     0.]  
        7    1 :                                       Upper_Chimney0x71a3800 :   PLC?     (1, 4, 4) : [ -412.  -412. -1750.   412.   412.  1750.     0.     0.]  
       12  126 :                                              sPlane0x71a8bb0 :   PLC?   (126, 4, 4) : [-3430.6  -3385.15    -6.65  3430.6   3385.15     6.65     0.       0.  ]  
       13   63 :                                               sWall0x71a8b30 :   PLC?    (63, 4, 4) : [-3430.6 -3430.6   -13.8  3430.6  3430.6    13.8     0.      0. ]  
       14    1 :                                              sAirTT0x71a76a0 :            (1, 4, 4) : [-24000. -24000.  21752.  24000.  24000.  30152.      0.      0.]  
       15    1 :                                            sExpHall0x578d4f0 :            (1, 4, 4) : [-31250. -24500.  21750.  31250.  24500.  32750.      0.      0.]  
       16    1 :                                         sExpRockBox0x578ce00 :            (1, 4, 4) : [-31125. -27500.  -5500.  31125.  27500.   5500.      0.      0.]  
       17    1 :                                            sTopRock0x578c0a0 :            (1, 4, 4) : [-31125. -27500. -15000.  31125.  27500.  15000.      0.      0.]  
       90    2 :                                    solidSJCLSanchor0x5962500 :            (2, 4, 4) : [   -60.    -12773.921  12245.275     60.    -12683.368  12338.543      0.         0.   ]  
       91   36 :                                      solidSJFixture0x5966940 :           (36, 4, 4) : [-15179.856  -8778.528   2559.474 -15138.459  -8726.628   2610.399      0.         0.   ]  
       92    8 :                                     solidSJReceiver0x5964f50 :            (8, 4, 4) : [-15358.562   8792.987    -60.    -15289.901   8901.911     60.         0.         0.   ]  
       93    8 :                              solidSJReceiverFastern0x5969570 :            (8, 4, 4) : [-15352.4     8777.157    -62.    -15265.26    8901.24      62.         0.         0.   ]  
       94    1 :                                             sTarget0x5829bb0 :            (1, 4, 4) : [-17700. -17700. -17700.  17700.  17700.  17824.      0.      0.]  
       95    1 :                                            sAcrylic0x5829590 :            (1, 4, 4) : [-17824. -17824. -17824.  17824.  17824.  17824.      0.      0.]  
       96  370 :                                              sStrut0x582c680 :  PLC?    (370, 4, 4) : [ -42.5    -42.5   -774.027   42.5     42.5    774.027    0.       0.   ]  
       97  220 :                                              sStrut0x5880150 :  PLC?    (220, 4, 4) : [ -30.     -30.    -774.039   30.      30.     774.039    0.       0.   ]  
      102   56 :                                       solidXJanchor0x5933100 :           (56, 4, 4) : [ -141.592 -1800.876 17733.078   -16.254 -1675.153 17764.264     0.        0.   ]  
      103   56 :                                      solidXJfixture0x595f360 :           (56, 4, 4) : [ -144.657 -1801.932    -0.        0.        0.    17790.158     0.        0.   ]  
      123    1 :                                     sChimneyAcrylic0x71a6010 :            (1, 4, 4) : [ -524.  -524. 17824.   524.   524. 18424.     0.     0.]  
      124    1 :                                          sChimneyLS0x71a61f0 :  PLC?      (1, 4, 4) : [ -400.  -400. -1963.   400.   400.  1963.     0.     0.]  
      125    1 :                                       sChimneySteel0x71a63d0 :            (1, 4, 4) : [ -405.  -405. 18424.   405.   405. 21750.     0.     0.]  
      126    1 :                                          sWaterTube0x71a5e30 :  PLC?      (1, 4, 4) : [ -524.  -524. -1963.   524.   524.  1963.     0.     0.]  
      127    1 :                                         sInnerWater0x5828f70 :            (1, 4, 4) : [-19629. -19629. -19629.  19629.  19629.  21750.      0.      0.]  
      128    1 :                                      sReflectorInCD0x58289b0 :            (1, 4, 4) : [-19631. -19631. -19631.  19631.  19631.  21750.      0.      0.]  
      135    1 :                                     sOuterWaterPool0x5792bd0 :            (1, 4, 4) : [-21750. -21750. -21750.  21750.  21750.  21750.      0.      0.]  
      136    1 :                                         sPoolLining0x57924c0 :            (1, 4, 4) : [-21755. -21755. -21755.  21755.  21755.  21750.      0.      0.]  
      137    1 :                                         sBottomRock0x578e0e0 :            (1, 4, 4) : [-24750. -24750. -24750.  24750.  24750.  21750.      0.      0.]  
      138    1 :                                              sWorld0x577e4d0 :            (1, 4, 4) : [-60000. -60000. -60000.  60000.  60000.  60000.      0.      0.]  
     skip:72  nmame_skip:Flange 

    In [1]:                                                                    



::

     all_ridxs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   ridxs:(0,)   nmame_skip:Flange   geocache_hookup_arg:old 

     ridx :  0   ridx_prims.shape (3084, 4, 4) 
     u_mx c_mx :                        unique midx prim counts and meshname  : prs.shape 
        0    1 :                                       Upper_LS_tube0x7170e10 :            (1, 4, 4) : [ -400.  -400. 20000.   400.   400. 23500.     0.     0.]  
        1    1 :                                    Upper_Steel_tube0x7170f00 :            (1, 4, 4) : [ -407.  -407. 20000.   407.   407. 23500.     0.     0.]  
        2    1 :                                    Upper_Tyvek_tube0x7171010 :            (1, 4, 4) : [ -402.  -402. 20000.   402.   402. 23500.     0.     0.]  
        3    1 :                                       Upper_Chimney0x7170d20 :            (1, 4, 4) : [ -412.  -412. 20000.   412.   412. 23500.     0.     0.]  
        8  126 :                                              sPlane0x71760f0 :          (126, 4, 4) : [ 16703.   -10096.35  23483.2   23564.2   -3326.05  23496.5       0.        0.  ]  
        9   63 :                                               sWall0x7176070 :           (63, 4, 4) : [ 16703.    -10141.801  23483.2    23564.2    -3280.6    23510.8        0.         0.   ]  
       10    1 :                                              sAirTT0x7174be0 :            (1, 4, 4) : [-24000. -24000.  21752.  24000.  24000.  30152.      0.      0.]  
       11    1 :                                            sExpHall0x5759fe0 :            (1, 4, 4) : [-24000. -24000.  21750.  24000.  24000.  40350.      0.      0.]  
       12    1 :                                            sTopRock0x5759b00 :            (1, 4, 4) : [-27000. -27000.  21750.  27000.  27000.  43350.      0.      0.]  
       85    2 :                                    solidSJCLSanchor0x592ed50 :            (2, 4, 4) : [-17700.   -25027.77 -25027.77  17700.    25027.77  25027.77      0.        0.  ]  
       86   36 :                                      solidSJFixture0x59331e0 :           (36, 4, 4) : [-2281.22  -1334.383 17484.422 -2197.285 -1251.283 17524.371     0.        0.   ]  
       87    8 :                                     solidSJReceiver0x5932540 :            (8, 4, 4) : [  -99.72    -99.72  17676.938    99.72     99.72  17709.938     0.        0.   ]  
       88   64 :                                      solidXJfixture0x592ba10 :           (64, 4, 4) : [  -61.471   -63.792 17676.938    61.471    63.792 17716.938     0.        0.   ]  
       89    1 :                                             sTarget0x57f5fb0 :            (1, 4, 4) : [-17700. -17700. -17700.  17700.  17700.  17820.      0.      0.]  
       90    1 :                                            sAcrylic0x57f5990 :            (1, 4, 4) : [-17820. -17820. -17820.  17820.  17820.  17820.      0.      0.]  
       91  370 :                                              sStrut0x57f6680 :          (370, 4, 4) : [  977.606  1367.903 17986.479  1185.118  1608.832 19539.42      0.        0.   ]  
       92  220 :                                              sStrut0x584c3a0 :          (220, 4, 4) : [16251.026  1678.546 -8384.757 17679.373  1887.683 -7669.625     0.        0.   ]  
       97   56 :                                       solidXJanchor0x58ff7a0 :           (56, 4, 4) : [-17743.13   -1811.087   -881.747 -17696.334  -1663.188   -734.99       0.         0.   ]  
      117    1 :                                     sChimneyAcrylic0x7173530 :            (1, 4, 4) : [ -520.  -520. 17820.   520.   520. 18420.     0.     0.]  
      118    1 :                                          sChimneyLS0x7173710 :            (1, 4, 4) : [ -400.  -400. 17820.   400.   400. 21750.     0.     0.]  
      119    1 :                                       sChimneySteel0x71738f0 :            (1, 4, 4) : [ -405.  -405. 18420.   405.   405. 21750.     0.     0.]  
      120    1 :                                          sWaterTube0x7173350 :            (1, 4, 4) : [ -520.  -520. 17820.   520.   520. 21750.     0.     0.]  
      121    1 :                                         sInnerWater0x57f5370 :            (1, 4, 4) : [-19629. -19629. -19629.  19629.  19629.  21750.      0.      0.]  
      122    1 :                                      sReflectorInCD0x57f4db0 :            (1, 4, 4) : [-19631. -19631. -19631.  19631.  19631.  21750.      0.      0.]  
      129    1 :                                     sOuterWaterPool0x575efd0 :            (1, 4, 4) : [-21750. -21750. -21750.  21750.  21750.  21750.      0.      0.]  
      130    1 :                                         sPoolLining0x575e8c0 :            (1, 4, 4) : [-21753. -21753. -21753.  21753.  21753.  21750.      0.      0.]  
      131    1 :                                         sBottomRock0x575a4e0 :            (1, 4, 4) : [-24750. -24750. -24750.  24750.  24750.  21750.      0.      0.]  
      132    1 :                                              sWorld0x574c190 :            (1, 4, 4) : [-60000. -60000. -60000.  60000.  60000.  60000.      0.      0.]  
     skip:72  nmame_skip:Flange 




GGeo/GPartsTest.sh new 
-------------------------

::

  123    1 :                                     sChimneyAcrylic0x71a6010 :            (1, 4, 4) : [ -524.  -524. 17824.   524.   524. 18424.     0.     0.]  
  124    1 :                                          sChimneyLS0x71a61f0 :            (1, 4, 4) : [ -400.  -400. -1963.   400.   400.  1963.     0.     0.]      ## LACKS PLACEMENT
  125    1 :                                       sChimneySteel0x71a63d0 :            (1, 4, 4) : [ -405.  -405. 18424.   405.   405. 21750.     0.     0.]  

::

    In [1]: ridx_midx_prim_ = lambda ridx,midx:g[ridx].prim[np.where( g[ridx].idx[:,1] == midx )]                                                                                                         

    In [2]: ridx_midx_prim_(0,123)                                                                                                                                                                        
    Out[2]: array([[23200,     3,  7087,     0]], dtype=int32)

    In [3]: ridx_midx_prim_(0,124)                                                                                                                                                                        
    Out[3]: array([[23203,     1,  7088,     0]], dtype=int32)   ## numParts 1 : could be issue with placement of leaves ?

    In [4]: ridx_midx_prim_(0,125)                                                                                                                                                                        
    Out[4]: array([[23204,     3,  7088,     0]], dtype=int32)




ok/ott.sh OptiX 5 render : shows the placement issue is there at OptiXRap level
---------------------------------------------------------------------------------

::

    ok
    ./ott.sh new 

    ## used --enabledmergedmesh 0,   to cope with ancient laptop GPU 
    ## get geometry to show up, navigate to center of CD, flip the O switch : see lots of simple prim floating near origin  



EUREKA : succeed to reproduce issue in a small scale test
------------------------------------------------------------

::

    g4ok
    GEOM=JustOrbGrid ./G4OKVolumeTest.sh     
         ## using large repeat_min to great pure global geometry 

    ok
    ./ott.sh      

        ## grid of 9 balls appears in the rasterized view 
        ## flip the O switch and get only one ball 



Looks like failure to GParts::applyPlacementTransform could be due to empty tranbuf for leaves
-------------------------------------------------------------------------------------------------

::

    2034 NPY<float>*    NCSG::getTransformBuffer() const {  return m_csgdata->getTransformBuffer() ; }
    2035 NPY<float>*    NCSG::getGTransformBuffer() const { return m_csgdata->getGTransformBuffer() ; }




::

     608 GParts* GParts::Make( const NCSG* tree, const char* spec, unsigned ndIdx )
     609 {
     610     assert(spec);
     611 
     612     bool usedglobally = tree->isUsedGlobally() ;   // see opticks/notes/issues/subtree_instances_missing_transform.rst
     613     assert( usedglobally == true );  // always true now ?   
     614 
     615     NPY<unsigned>* tree_idxbuf = tree->getIdxBuffer() ;   // (1,4) identity indices (index,soIdx,lvIdx,height)
     616     NPY<float>*   tree_tranbuf = tree->getGTransformBuffer() ;
     617     NPY<float>*   tree_planbuf = tree->getPlaneBuffer() ;
     618     assert( tree_tranbuf );
     619 
     620     NPY<unsigned>* idxbuf = tree_idxbuf->clone()  ;   // <-- lacking this clone was cause of the mystifying repeated indices see notes/issues/GPtsTest             
     621     NPY<float>* nodebuf = tree->getNodeBuffer();       // serialized binary tree
     622     NPY<float>* tranbuf = usedglobally                 ? tree_tranbuf->clone() : tree_tranbuf ;
     623     NPY<float>* planbuf = usedglobally && tree_planbuf ? tree_planbuf->clone() : tree_planbuf ;
     624 
     625     


NCSG::import has changed substantially for list nodes
---------------------------------------------------------


::

     462 void NCSG::import()
     463 {
     464     if(m_verbosity > 1)
     465         LOG(info)
     466                   << " verbosity(>1) " << m_verbosity
     467                   << " treedir " << m_treedir
     468                   << " smry " << smry()
     469                   ;
     470 
     471     if(m_verbosity > 1)
     472     {
     473         LOG(info)
     474                   << " verbosity(>0) " << m_verbosity
     475                   << " importing buffer into CSG node tree "
     476                   << " num_nodes " << getNumNodes()
     477                   ;
     478     }
     479 
     480     m_csgdata->prepareForImport() ;  // from m_srctransforms to m_transforms, and get m_gtransforms ready to collect
     481 
     482     // need type of first node to distinguish tree from list   
     483     OpticksCSG_t root_type  = (OpticksCSG_t)m_csgdata->getTypeCode(0);
     484     LOG(LEVEL) << " root_type " << CSG::Name(root_type) ;
     485 
     486     if(CSG::IsTree(root_type))
     487     {
     488         import_tree();
     489     }
     490     else if(CSG::IsList(root_type))
     491     {
     492         import_list();
     493     }
     494     else if(CSG::IsLeaf(root_type))
     495     {
     496         import_leaf();
     497     }
     498     else
     499     {
     500         LOG(fatal) << " UNEXPECTED root_type " << root_type << "  CSG::Name(root_type) " <<  CSG::Name(root_type) ;
     501         assert(0) ; // unexpected root_type 
     502     }
     503 
     504 
     505     m_root->set_treedir(m_treedir) ;
     506     m_root->set_treeidx(getTreeNameIdx()) ;  //


::

     944 /**
     945 nnode::set_placement
     946 -----------------------
     947 
     948 The placement transform is used by nnode::global_transform 
     949 when a placement transform is present on the root node, that 
     950 is included into the global transforms.  Hence this provides
     951 a way to bake in a placing transform to all the global transforms 
     952 of a node tree. 
     953 
     954 Formerly when this was called apply_placement
     955 the placement was nullified after update_gtransforms()
     956 but it appears that methods such as NCSG::collect_global_transforms 
     957 are going to invoke node->global_transform() again on all 
     958 primitives ... so need to leave the placement in place.
     959 
     960 **/
     961 
     962 void nnode::set_placement( const nmat4triple* plc )
     963 {
     964     assert( is_root() ) ;
     965     placement = plc ;
     966     update_gtransforms();
     967 
     968     // placement = NULL ;   
     969 }
     970 

::

     982 void NCSG::collect_global_transforms()
     983 {
     984     bool locked(false) ;  // with locked=true m_csgdata asserts if called more than once
     985     m_csgdata->prepareForGTransforms(locked);
     986 
     987     if(m_root->is_tree())
     988     {
     989         collect_global_transforms_r( m_root ) ;
     990     }
     991     else if(m_root->is_list())
     992     {
     993         collect_global_transforms_list(m_root) ;
     994     }
     995     else
     996     {
     997          assert(0);  // WHA WHA OOPS : THIS COULD BE THE CAUSE OF THE LEAF PLACEMENT BUG 
     998     }
     999 }


CONFIRMED FIX
-----------------

::

    .NCSG* NCSG::Load(const char* treedir)
     {
         const char* config_ = "polygonize=0" ; 
    @@ -990,10 +988,23 @@ void NCSG::collect_global_transforms()
         {
             collect_global_transforms_r( m_root ) ; 
         }
    +    else if(m_root->is_leaf())  // WHA WHA OOPS : PRIOR OMISSION OF THIS MAY BE THE CAUSE OF THE LEAF PLACEMENT BUG 
    +    {
    +        collect_global_transforms_leaf(m_root) ; 
    +    }
         else if(m_root->is_list())
         {
             collect_global_transforms_list(m_root) ; 
         }
    +    else
    +    {
    +         assert(0); 
    +    }
    +}
    +
    +void NCSG::collect_global_transforms_leaf(nnode* node) 
    +{
    +    collect_global_transforms_node(node);
     }
     





