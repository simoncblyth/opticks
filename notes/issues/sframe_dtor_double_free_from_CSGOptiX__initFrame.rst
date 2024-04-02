sframe_dtor_double_free_from_CSGOptiX__initFrame
=================================================


sframe leaking no problem ? as much fewer instances in use after change from SEvt::getLocalHit_LEAKY to SEvt::getLocalHit
--------------------------------------------------------------------------------------------------------------------------

* now at level of 1 or 2 sframe instances per SEvt 
* aim to reduce more, switching from sframe.h to sfr.h in SEvt and



::

    4318 void SEvt::getLocalHit(sphit& ht, sphoton& lp, unsigned idx) const
    4319 {
    4320     getHit(lp, idx);   // copy *idx* hit from NP array into sphoton& lp struct 
    4321     int iindex = lp.iindex ;
    4322 
    4323     const glm::tmat4x4<double>* tr = tree ? tree->get_iinst(iindex) : nullptr ;
    4324 
    4325     LOG_IF(fatal, tr == nullptr)
    4326          << " FAILED TO GET INSTANCE TRANSFORM : WHEN TESTING NEEDS SSim::Load NOT SSim::Create"
    4327          << " iindex " << iindex
    4328          << " tree " << ( tree ? "YES" : "NO " )
    4329          << " tree.desc_inst " << ( tree ? tree->desc_inst() : "-" )
    4330          ;
    4331     assert( tr );
    4332 
    4333     bool normalize = true ;
    4334     lp.transform( *tr, normalize );
    4335 
    4336     glm::tvec4<int64_t> col3 = {} ;
    4337     strid::Decode( *tr, col3 );
    4338 
    4339     ht.iindex = col3[0] ;
    4340     ht.sensor_identifier = col3[2] ;  // NB : NO "-1" HERE : SEE ABOVE COMMENT 
    4341     ht.sensor_index = col3[3] ;
    4342 }




HMM, tis tedious to debug this... as need to rebuild sysrap+CSG
-----------------------------------------------------------------

Plan for fix:

1. just leak transforms for now
2. add an stree based equivalent frame creation 
 
   * wanted to do this anyhow, as no need to get that info from CSG level  

3. ensure they give equivalent results
4. port to the stree based frame access
5. reenable transform cleaning : if still double free
   can debug with much faster cycle profiting from the move to stree. 


Issue
------

(March 13, 2024)
    Hans points out a double free error in sframe dtor, a reversion in Opticks HEAD not in recent tags like v0.2.7::


::

    (gdb) bt
    #0  0x00007fffe7040387 in raise () from /lib64/libc.so.6
    #1  0x00007fffe7041a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffe7082f67 in __libc_message () from /lib64/libc.so.6
    #3  0x00007fffe708b329 in _int_free () from /lib64/libc.so.6
    #4  0x00007fffe9f7fbdf in sframe::~sframe (this=0x7ffffffe1f90, __in_chrg=<optimized out>) at /data2/wenzel/hybrid6/local/opticks/include/SysRap/sframe.h:188
    #5  0x00007fffe9f5f428 in CSGOptiX::initFrame (this=0x178d4f0) at /data2/wenzel/hybrid6/opticks/CSGOptiX/CSGOptiX.cc:653
    #6  0x00007fffe9f5dd0c in CSGOptiX::init (this=0x178d4f0) at /data2/wenzel/hybrid6/opticks/CSGOptiX/CSGOptiX.cc:476
    #7  0x00007fffe9f5d83c in CSGOptiX::CSGOptiX (this=0x178d4f0, foundry_=0xe96860) at /data2/wenzel/hybrid6/opticks/CSGOptiX/CSGOptiX.cc:445
    #8  0x00007fffe9f5cf1b in CSGOptiX::Create (fd=0xe96860) at /data2/wenzel/hybrid6/opticks/CSGOptiX/CSGOptiX.cc:367
    #9  0x00007ffff78a95e1 in G4CXOpticks::setGeometry_ (this=0xd684a0, fd_=0xe96860) at /data2/wenzel/hybrid6/opticks/g4cx/G4CXOpticks.cc:299
    #10 0x00007ffff78a93ff in G4CXOpticks::setGeometry (this=0xd684a0, fd_=0xe96860) at /data2/wenzel/hybrid6/opticks/g4cx/G4CXOpticks.cc:266
    #11 0x00007ffff78a91a6 in G4CXOpticks::setGeometry (this=0xd684a0, world=0xd61150) at /data2/wenzel/hybrid6/opticks/g4cx/G4CXOpticks.cc:240
    #12 0x00007ffff78a7965 in G4CXOpticks::SetGeometry (world=0xd61150) at /data2/wenzel/hybrid6/opticks/g4cx/G4CXOpticks.cc:58
    #13 0x0000000000427973 in DetectorConstruction::ReadGDML() ()
    #14 0x0000000000428a23 in DetectorConstruction::Construct() ()
    #15 0x00007ffff4b2a1de in G4RunManager::InitializeGeometry (this=0xb2f0f0)
        at /scratch/workspace/geant4-release-build/v4_11_1_p01ba/e20/SLF7/prof/build/geant4/v4_11_1_p01ba/source/geant4-v11.1.1/source/run/src/G4RunManager.cc:711
    #16 0x00007ffff4b29c13 in G4RunManager::Initialize (this=0xb2f0f0)



Checking opticks-t reveals lots of CSG fails and a few others::

    FAILS:  20  / 213   :  Wed Mar 13 09:46:44 2024   
      1  /42  Test #1  : CSGTest.CSGNodeTest                           ***Failed                      2.51   
      5  /42  Test #5  : CSGTest.CSGPrimSpecTest                       ***Failed                      2.46   
      6  /42  Test #6  : CSGTest.CSGPrimTest                           ***Failed                      2.48   
      8  /42  Test #8  : CSGTest.CSGFoundryTest                        ***Failed                      2.43   
      10 /42  Test #10 : CSGTest.CSGFoundry_getCenterExtent_Test       ***Failed                      2.46   
      11 /42  Test #11 : CSGTest.CSGFoundry_findSolidIdx_Test          ***Failed                      2.47   
      13 /42  Test #13 : CSGTest.CSGNameTest                           ***Failed                      2.42   
      14 /42  Test #14 : CSGTest.CSGTargetTest                         ***Failed                      2.49   
      15 /42  Test #15 : CSGTest.CSGTargetGlobalTest                   ***Failed                      2.47   
      16 /42  Test #16 : CSGTest.CSGFoundry_MakeCenterExtentGensteps_Test ***Failed                      2.48   
      17 /42  Test #17 : CSGTest.CSGFoundry_getFrame_Test              ***Failed                      2.45   
      18 /42  Test #18 : CSGTest.CSGFoundry_getFrameE_Test             ***Failed                      2.49   
      19 /42  Test #19 : CSGTest.CSGFoundry_getMeshName_Test           ***Failed                      2.42   
      22 /42  Test #22 : CSGTest.CSGFoundryLoadTest                    ***Failed                      2.47   
      28 /42  Test #28 : CSGTest.CSGSimtraceTest                       ***Failed                      2.50   
      29 /42  Test #29 : CSGTest.CSGSimtraceRerunTest                  ***Failed                      2.46   
      30 /42  Test #30 : CSGTest.CSGSimtraceSampleTest                 ***Failed                      2.42   
      3  /4   Test #3  : CSGOptiXTest.CSGOptiXRenderTest               ***Failed                      2.48   
      1  /2   Test #1  : G4CXTest.G4CXRenderTest                       ***Failed                      3.10   
      2  /2   Test #2  : G4CXTest.G4CXOpticks_setGeometry_Test         ***Failed                      2.66   
    om-test-help




This is with::

    // dtor
    inline sframe::~sframe()
    {
        delete tr_m2w ; 
        delete tr_w2m ; 
    }


Commenting those deletes prevents the errors, 
but also may mean leaking of transforms. 
That doesnt matter for most usage of sframe as there 
are not many of them, but they are also used for 
each hit so its not acceptable to just leak transforms.

* sframe is intended more as  "interactive" struct with few 
  intances,  not as a critical one with instances for every hit

While hit transformation needs rework to avoid use of sframe.h
it will take a while before doing that.  So need a fix 
for sframe that doesnt leak. 

* ACTUALLY stree CAN DIRECTLY PROVIDE THE TRANSFORMS ALREADY 

Looking at how sframe.h is being used in the failing cases, 
I see that often copies are done with::

    sframe b = a ; 

But that causes double ownership of the transform pointers
due to use of default copy ctor. 

Have two options:

1. implement copy ctor, that allocates new Tran<double> avoiding double ownership
2. change transforms from pointers to glm::tmat4<double> values 

For now opt for first, as less disruptive. 
But sframe needs overhaul, so could switch to a pure value 
struct in future. 

 


HUH still failing after added copy ctor::

    FAILS:  19  / 213   :  Wed Mar 13 11:22:41 2024   
      1  /42  Test #1  : CSGTest.CSGNodeTest                           ***Failed                      2.29   
      5  /42  Test #5  : CSGTest.CSGPrimSpecTest                       ***Failed                      2.29   
      6  /42  Test #6  : CSGTest.CSGPrimTest                           ***Failed                      2.24   
      8  /42  Test #8  : CSGTest.CSGFoundryTest                        ***Failed                      2.24   
      10 /42  Test #10 : CSGTest.CSGFoundry_getCenterExtent_Test       ***Failed                      2.23   
      11 /42  Test #11 : CSGTest.CSGFoundry_findSolidIdx_Test          ***Failed                      2.25   
      13 /42  Test #13 : CSGTest.CSGNameTest                           ***Failed                      2.27   
      14 /42  Test #14 : CSGTest.CSGTargetTest                         ***Failed                      2.27   
      15 /42  Test #15 : CSGTest.CSGTargetGlobalTest                   ***Failed                      2.36   
      16 /42  Test #16 : CSGTest.CSGFoundry_MakeCenterExtentGensteps_Test ***Failed                      2.39   
      17 /42  Test #17 : CSGTest.CSGFoundry_getFrame_Test              ***Failed                      2.35   
      18 /42  Test #18 : CSGTest.CSGFoundry_getFrameE_Test             ***Failed                      2.29   
      19 /42  Test #19 : CSGTest.CSGFoundry_getMeshName_Test           ***Failed                      2.27   
      22 /42  Test #22 : CSGTest.CSGFoundryLoadTest                    ***Failed                      2.24   
      28 /42  Test #28 : CSGTest.CSGSimtraceTest                       ***Failed                      2.36   
      29 /42  Test #29 : CSGTest.CSGSimtraceRerunTest                  ***Failed                      2.29   
      30 /42  Test #30 : CSGTest.CSGSimtraceSampleTest                 ***Failed                      2.26   
      1  /2   Test #1  : G4CXTest.G4CXRenderTest                       ***Failed                      3.30   
      2  /2   Test #2  : G4CXTest.G4CXOpticks_setGeometry_Test         ***Failed                      2.42   





Look at fail sites
-------------------

CSGNodeTest::

    3460 void CSGFoundry::AfterLoadOrCreate() // static
    3461 {
    3462     CSGFoundry* fd = CSGFoundry::Get();
    3463 
    3464     SEvt::CreateOrReuse() ;   // creates 1/2 SEvt depending on OPTICKS_INTEGRATION_MODE
    3465 
    3466     if(!fd) return ;
    3467 
    3468     sframe fr = fd->getFrameE() ;
    3469     LOG(LEVEL) << fr ;
    3470     SEvt::SetFrame(fr); // now only needs to be done once to transform input photons
    3471 
    3472 }




stree access to inst transforms
-----------------------------------

Full precision with identity extras added here::

    3034 inline void stree::add_inst(
    3035     glm::tmat4x4<double>& tr_m2w,
    3036     glm::tmat4x4<double>& tr_w2m,
    3037     int gas_idx,
    3038     int nidx )
    3039 {
    3040     assert( nidx > -1 && nidx < int(nds.size()) );
    3041     const snode& nd = nds[nidx];    // structural volume node
    3042 
    3043     int ins_idx = int(inst.size()); // follow sqat4.h::setIdentity
    3044 
    3045     glm::tvec4<int64_t> col3 ;   // formerly uint64_t 
    3046 
    3047     col3.x = ins_idx ;            // formerly  +1 
    3048     col3.y = gas_idx ;            // formerly  +1 
    3049     col3.z = nd.sensor_id ;       // formerly ias_idx + 1 (which was always 1)
    3050     col3.w = nd.sensor_index ;
    3051 
    3052     strid::Encode(tr_m2w, col3 );
    3053     strid::Encode(tr_w2m, col3 );
    3054 
    3055     inst.push_back(tr_m2w);
    3056     iinst.push_back(tr_w2m);
    3057 
    3058     inst_nidx.push_back(nidx);
    3059 }


    3165 inline const glm::tmat4x4<double>* stree::get_inst(int idx) const
    3166 {
    3167     return idx > -1 && idx < int(inst.size()) ? &inst[idx] : nullptr ;
    3168 }
    3169 inline const glm::tmat4x4<double>* stree::get_iinst(int idx) const
    3170 {
    3171     return idx > -1 && idx < int(iinst.size()) ? &iinst[idx] : nullptr ;
    3172 }
    3173 
    3174 inline const glm::tmat4x4<float>* stree::get_inst_f4(int idx) const
    3175 {
    3176     return idx > -1 && idx < int(inst_f4.size()) ? &inst_f4[idx] : nullptr ;
    3177 }
    3178 inline const glm::tmat4x4<float>* stree::get_iinst_f4(int idx) const
    3179 {
    3180     return idx > -1 && idx < int(iinst_f4.size()) ? &iinst_f4[idx] : nullptr ;
    3181 }




sframe not so prolific : it can be replaced fairly easily
------------------------------------------------------------

::

    epsilon:opticks blyth$ opticks-fl sframe.h 
    ./ana/framegensteps.py

    ./CSG/CSGFoundry.cc
    ./CSG/CSGTarget.cc
    ./CSG/CSGSimtrace.hh

    ./CSG/tests/CSGTargetTest.cc
    ./CSG/tests/CSGFoundry_MakeCenterExtentGensteps_Test.cc
    ./CSG/tests/CSGFoundry_getFrame_Test.cc
    ./CSG/tests/CSGFoundry_getFrameE_Test.cc

    ./CSGOptiX/CSGOptiX.h
    ./CSGOptiX/CSGOptiX.cc
    ./CSGOptiX/cxr_min.sh
    ./CSGOptiX/tests/CSGOptiXSimtraceTest.cc

    ./sysrap/CMakeLists.txt
    ./sysrap/SFrameGenstep.hh
    ./sysrap/SFrameGenstep.cc
    ./sysrap/CheckGeo.cc
    ./sysrap/tests/CheckGeoTest.cc
    ./sysrap/tests/sframe_test.cc
    ./sysrap/tests/SFrameGenstep_MakeCenterExtentGensteps_Test.cc
    ./sysrap/SGLM.h
    ./sysrap/tests/SGLM_set_frame_test.sh
    ./sysrap/sframe.h
    ./sysrap/tests/sframeTest.cc
    ./sysrap/SEvt.hh
    ./sysrap/SEvt.cc
    ./sysrap/SEvent.hh
    ./sysrap/SEvent.cc
    ./sysrap/SSimtrace.h

    ./u4/U4App.h
    ./cxr_min.sh
    ./g4cx/tests/G4CXApp.h

    ./examples/UseGeometryShader/UseGeometryShader.cc

    ./ggeo/GGeo.cc
    ./extg4/X4Simtrace.hh

    epsilon:opticks blyth$ 




TODO : frame from tree matching with frame from foundry 
--------------------------------------------------------

1. manual iteration to get close
2. script/executable to load persisted frames and compare fully


::

    IIDX=20000 ~/o/sysrap/tests/stree_load_test.sh
    OIPF=20000 ~/o/CSG/tests/CSGFoundry_getFrameE_Test.sh







