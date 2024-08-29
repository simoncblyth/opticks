flexible_forced_triangulation
================================

Issue
------

Previously geometry code used simple layout::

    ridx:0  global analytic compound solid 
    ridx:1,2,3,...  instanced analytic solids

To provide flexible triangulation need:

1. generalize ana/tri type of each ridx slot  
2. user control to make an lv become triangulated 

Relevant context
-------------------

::

    sysrap/stree.h 
    u4/U4Tree.h 


Q: is the triangulation action post cache ? if not, could/should it be ? 
---------------------------------------------------------------------------

A: NO

stree::collectGlobalNodes invoked at tail of stree::factorize acts on the 
envvar to control the nodes going to the tri/rem vectors


Test with forced triangulation 
-------------------------------

In jok.bash configure what to triangulate via filepath::

    154 export stree__force_triangulate_solid='filepath:$HOME/.opticks/GEOM/${GEOM}_meshname_stree__force_triangulate_solid.txt'

Run one event::

    jok-tds-gdb 

Render::

    ~/o/cxr_min.sh 

    ~/o/sysrap/tests/ssst1.sh 



FIXED : Seventh issue : running from loaded stree does not retain the force triangulate 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





FIXED : Sixth issue : torus PLACEHOLDER logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    sn::setAABB_LeafFrame   typecode 116 CSG::Name(typecode) torus DO NOTHING PLACEHOLDER 
    sn::setAABB_LeafFrame   typecode 116 CSG::Name(typecode) torus DO NOTHING PLACEHOLDER 
    sn::setAABB_LeafFrame   typecode 116 CSG::Name(typecode) torus DO NOTHING PLACEHOLDER 
    sn::setAABB_LeafFrame   typecode 116 CSG::Name(typecode) torus DO NOTHING PLACEHOLDER 
    sn::setAABB_LeafFrame   typecode 116 CSG::Name(typecode) torus DO NOTHING PLACEHOLDER 
    sn::setAABB_LeafFrame   typecode 116 CSG::Name(typecode) torus DO NOTHING PLACEHOLDER 
    sn::setAABB_LeafFrame   typecode 116 CSG::Name(typecode) torus DO NOTHING PLACEHOLDER 


Hmm sgeomtools.h is used to find torus bbox somewhere ? why not here ? 

::

    4198     else if( typecode == CSG_HYPERBOLOID )
    4199     {
    4200         double r0, zf, z1, z2, a, b ;
    4201         getParam_(r0, zf, z1, z2, a, b ) ;
    4202         assert( a == 0. && b == 0. );
    4203         assert( z1 < z2 );
    4204         const double rr0 = r0*r0 ;
    4205         const double z1s = z1/zf ;
    4206         const double z2s = z2/zf ;
    4207 
    4208         const double rr1 = rr0 * ( z1s*z1s + 1. ) ;
    4209         const double rr2 = rr0 * ( z2s*z2s + 1. ) ;
    4210         const double rmx = sqrtf(fmaxf( rr1, rr2 )) ;
    4211 
    4212         setBB(  -rmx,  -rmx,  z1,  rmx, rmx, z2 );
    4213     }
    4214     else if( typecode == CSG_TORUS )
    4215     {
    4216         std::cout
    4217             << "sn::setAABB_LeafFrame  "
    4218             << " typecode " << typecode
    4219             << " CSG::Name(typecode) " << CSG::Name(typecode)
    4220             << " DO NOTHING PLACEHOLDER "
    4221             << "\n"
    4222             ;
    4223     }


Replace the placeholder::

    4245     else if( typecode == CSG_TORUS )
    4246     {
    4247         double rmin, rmax, rtor, startPhi_deg, deltaPhi_deg, zero ;
    4248         getParam_(rmin, rmax, rtor, startPhi_deg, deltaPhi_deg, zero) ;
    4249 
    4250         double rext = rtor+rmax ;
    4251         double rint = rtor-rmax ;
    4252         double startPhi = startPhi_deg/180.*M_PI ;
    4253         double deltaPhi = deltaPhi_deg/180.*M_PI ;
    4254         double2 pmin ;
    4255         double2 pmax ;
    4256         sgeomtools::DiskExtent(rint, rext, startPhi, deltaPhi, pmin, pmax );
    4257 
    4258         setBB( pmin.x, pmin.y, -rmax, pmax.x, pmax.y, +rmax );
    4259     
    4260     }



FIXED : Fifth issue : ssst1.sh runs but rem global solid not visible in OpenGL render 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Pressing C to jump to the OptiX render makes the global geom appear 
* based on seeing guide tube the triangulated geom is present in both 

Persisted scene lacks meshmesh entry 10::

    [blyth@localhost scene]$ cd meshmerge/
    [blyth@localhost meshmerge]$ l
    total 4
    0 -rw-rw-r--.  1 blyth blyth   0 Aug 27 21:35 NPFold_names.txt
    4 -rw-rw-r--.  1 blyth blyth  22 Aug 27 21:35 NPFold_index.txt
    0 drwxr-xr-x.  5 blyth blyth 143 Aug 27 10:18 ..
    0 drwxr-xr-x. 12 blyth blyth 144 Aug 27 10:18 .
    0 drwxr-xr-x.  2 blyth blyth  99 Aug 27 10:18 8
    0 drwxr-xr-x.  2 blyth blyth  99 Aug 27 10:18 9
    0 drwxr-xr-x.  2 blyth blyth  99 Aug 27 10:18 5
    0 drwxr-xr-x.  2 blyth blyth  99 Aug 27 10:18 6
    0 drwxr-xr-x.  2 blyth blyth  99 Aug 27 10:18 7
    0 drwxr-xr-x.  2 blyth blyth  99 Aug 27 10:18 3
    0 drwxr-xr-x.  2 blyth blyth  99 Aug 27 10:18 4
    0 drwxr-xr-x.  2 blyth blyth  99 Aug 27 10:18 1
    0 drwxr-xr-x.  2 blyth blyth  99 Aug 27 10:18 2
    0 drwxr-xr-x.  2 blyth blyth  99 Aug 27 10:18 0
    [blyth@localhost meshmerge]$ date
    Wed Aug 28 09:53:14 CST 2024
    [blyth@localhost meshmerge]$ cd ..
    [blyth@localhost scene]$ l meshgroup/
    total 132
      0 -rw-rw-r--.    1 blyth blyth     0 Aug 27 21:35 NPFold_names.txt
      4 -rw-rw-r--.    1 blyth blyth    23 Aug 27 21:35 NPFold_index.txt
     12 drwxr-xr-x.  324 blyth blyth  8192 Aug 27 17:07 10
      0 drwxr-xr-x.   13 blyth blyth   154 Aug 27 17:07 .
      0 drwxr-xr-x.    5 blyth blyth   143 Aug 27 10:18 ..
      4 drwxr-xr-x.  132 blyth blyth  4096 Aug 27 10:18 9
      0 drwxr-xr-x.    3 blyth blyth    63 Aug 27 10:18 7
      0 drwxr-xr-x.    3 blyth blyth    63 Aug 27 10:18 8
      0 drwxr-xr-x.    3 blyth blyth    63 Aug 27 10:18 5
      0 drwxr-xr-x.    3 blyth blyth    63 Aug 27 10:18 6
      0 drwxr-xr-x.    6 blyth blyth    90 Aug 27 10:18 4
      0 drwxr-xr-x.   14 blyth blyth   164 Aug 27 10:18 3
      0 drwxr-xr-x.   11 blyth blyth   135 Aug 27 10:18 2
      0 drwxr-xr-x.    7 blyth blyth    99 Aug 27 10:18 1
    112 drwxr-xr-x. 3220 blyth blyth 53248 Aug 27 10:18 0
    [blyth@localhost scene]$ 



Actually there is a duplicated NPFold key for meshmerge::

    [blyth@localhost meshmerge]$ cat NPFold_index.txt
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    0


Due to duplicated m->name ?::

    498 inline NPFold* SScene::serialize_meshmerge() const
    499 {
    500     NPFold* _meshmerge = new NPFold ;
    501     int num_meshmerge = meshmerge.size();
    502     for(int i=0 ; i < num_meshmerge ; i++)
    503     {
    504         const SMesh* m = meshmerge[i] ;
    505         _meshmerge->add_subfold( m->name, m->serialize() );
    506     }
    507     return _meshmerge ;
    508 }





::

    208 inline void SScene::initFromTree_Global(const stree* st, char ridx_type )
    209 {
    210     assert( ridx_type == 'R' || ridx_type == 'T' );
    211     const std::vector<snode>* _nodes = st->get_node_vector(ridx_type)  ;
    212     assert( _nodes );
    213 
    214     int num_node = _nodes->size() ;
    215     if(dump) std::cout
    216         << "[ SScene::initFromTree_Remainder"
    217         << " num_node " << num_node
    218         << std::endl
    219         ;
    220 
    221     SMeshGroup* mg = new SMeshGroup ;
    222     int ridx = 0 ;
    223     for(int i=0 ; i < num_node ; i++)
    224     {
    225         const snode& node = (*_nodes)[i];
    226         initFromTree_Node(mg, ridx, node, st);
    227         // HUH: CANNOT BE CORRECT : RIDX NOT ZERO FOR TRI
    228     }
    229     const SMesh* _mesh = SMesh::Concatenate( mg->subs, 0 );
    230     meshmerge.push_back(_mesh);
    231     meshgroup.push_back(mg);
    232 
    233     if(dump) std::cout
    234         << "] SScene::initFromTree_Global"
    235         << " num_node " << num_node 
    236         << " ridx_type " << ridx_type
    237         << std::endl
    238         ;
    239 }



FIXED : Fourth issue : ssst1.sh num_inst abort
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [SOPTIX_Options::Desc_pipelineLinkOptions
     pipeline_link_options.maxTraceDepth   2

     pipeline_link_options.debugLevel      0 OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT]SOPTIX_Options::Desc_pipelineLinkOptions
    ]SOPTIX_Options::desc

    ]SOPTIX_Module::desc
    [ 4][   DISKCACHE]: Cache hit for key: ptx-73159-key3961702910e23ce4a85652601da14472-sm_75-rtc1-drv515.43.04
    [ 4][COMPILE FEEDBACK]: 
    [ 4][   DISKCACHE]: Cache hit for key: ptx-30638-keye7bede57aa8f15105c5d28e25df63ca6-sm_75-rtc1-drv515.43.04
    [ 4][COMPILE FEEDBACK]: 
    [ 4][COMPILE FEEDBACK]: Info: Pipeline has 1 module(s), 3 entry function(s), 1 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 29 basic block(s) in entry functions, 853 instruction(s) in entry functions, 7 non-entry function(s), 53 basic block(s) in non-entry functions, 627 instruction(s) in non-entry functions, no debug information

    SGLFW_SOPTIX_Scene_test: ../SOPTIX_Scene.h:155: void SOPTIX_Scene::init_Instances(): Assertion `idx < num_inst' failed.
    /data/blyth/junotop/opticks/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh: line 339: 89286 Aborted                 (core dumped) $bin
    /data/blyth/junotop/opticks/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh : run error
    [blyth@localhost tests]$ 


offset off by one ?::

    SOPTIX_Scene::init_GAS num_mg 11
    SOPTIX_Scene::init_Instances num_gas 11 num_inst 48478
    SOPTIX_Scene::init_Instances i 0 ridx (_inst_info.x) 0 count (_inst_info.y 1 offset (_inst_info.z)  0 num_bi 2896 visibilityMask 1 sbtOffset 0
    SOPTIX_Scene::init_Instances i 1 ridx (_inst_info.x) 1 count (_inst_info.y 25600 offset (_inst_info.z)  1 num_bi 5 visibilityMask 2 sbtOffset 2896
    SOPTIX_Scene::init_Instances i 2 ridx (_inst_info.x) 2 count (_inst_info.y 12615 offset (_inst_info.z)  25601 num_bi 9 visibilityMask 4 sbtOffset 2901
    SOPTIX_Scene::init_Instances i 3 ridx (_inst_info.x) 3 count (_inst_info.y 4997 offset (_inst_info.z)  38216 num_bi 12 visibilityMask 8 sbtOffset 2910
    SOPTIX_Scene::init_Instances i 4 ridx (_inst_info.x) 4 count (_inst_info.y 2400 offset (_inst_info.z)  43213 num_bi 4 visibilityMask 16 sbtOffset 2922
    SOPTIX_Scene::init_Instances i 5 ridx (_inst_info.x) 5 count (_inst_info.y 590 offset (_inst_info.z)  45613 num_bi 1 visibilityMask 32 sbtOffset 2926
    SOPTIX_Scene::init_Instances i 6 ridx (_inst_info.x) 6 count (_inst_info.y 590 offset (_inst_info.z)  46203 num_bi 1 visibilityMask 64 sbtOffset 2927
    SOPTIX_Scene::init_Instances i 7 ridx (_inst_info.x) 7 count (_inst_info.y 590 offset (_inst_info.z)  46793 num_bi 1 visibilityMask 128 sbtOffset 2928
    SOPTIX_Scene::init_Instances i 8 ridx (_inst_info.x) 8 count (_inst_info.y 590 offset (_inst_info.z)  47383 num_bi 1 visibilityMask 128 sbtOffset 2929
    SOPTIX_Scene::init_Instances i 9 ridx (_inst_info.x) 9 count (_inst_info.y 504 offset (_inst_info.z)  47973 num_bi 130 visibilityMask 128 sbtOffset 2930
    SOPTIX_Scene::init_Instances i 10 ridx (_inst_info.x) 10 count (_inst_info.y 1 offset (_inst_info.z)  48478 num_bi 322 visibilityMask 128 sbtOffset 3060
    SOPTIX_Scene::init_Instances j 0 (offset + j)[idx] 48478 num_inst 48478 in_range NO  tot 48477
    SGLFW_SOPTIX_Scene_test: ../SOPTIX_Scene.h:186: void SOPTIX_Scene::init_Instances(): Assertion `in_range' failed.
    /data/blyth/junotop/opticks/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh: line 368: 162656 Aborted                 (core dumped) $bin
    /data/blyth/junotop/opticks/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh : run error
    [blyth@localhost tests]$ echo $(( 47973 + 130 ))
    48103
    [blyth@localhost tests]$ echo $(( 47973 + 590 ))
    48563
    [blyth@localhost tests]$ echo $(( 1 + 25600 ))
    25601
    [blyth@localhost tests]$ echo $(( 1 + 25600 + 4997 ))
    30598
    [blyth@localhost tests]$ echo $(( 1 + 25600 + 12615 ))
    38216
    [blyth@localhost tests]$ echo $(( 1 + 25600 + 12615 + 2400 ))
    40616
    [blyth@localhost tests]$ echo $(( 1 + 25600 + 12615 + 4997  ))
    43213
    [blyth@localhost tests]$ echo $(( 1 + 25600 + 12615 + 4997 + 240  ))
    43453
    [blyth@localhost tests]$ echo $(( 1 + 25600 + 12615 + 4997 + 2400  ))
    45613
    [blyth@localhost tests]$ echo $(( 1 + 25600 + 12615 + 4997 + 2400 + 590 ))
    46203
    [blyth@localhost tests]$ echo $(( 1 + 25600 + 12615 + 4997 + 2400 + 590 + 590 ))
    46793
    [blyth@localhost tests]$ echo $(( 1 + 25600 + 12615 + 4997 + 2400 + 590 + 590 + 590 ))
    47383
    [blyth@localhost tests]$ echo $(( 1 + 25600 + 12615 + 4997 + 2400 + 590 + 590 + 590 + 590 ))
    47973
    [blyth@localhost tests]$ echo $(( 1 + 25600 + 12615 + 4997 + 2400 + 590 + 590 + 590 + 590 + 504 ))
    48477
    [blyth@localhost tests]$ 


After changing stree.h this required a jok-tds-gdb rerun to recreate the persisted SScene. 



FIXED : Third issue : cxr_min.sh runtime the triangulated not rendered 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Looks like IAS issue, missing inst info for the triangulated. 


FIXED : Second issue : missing last meshgroups for GAS creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Looks like relying on stale inst info without the tri entry, plus SScene.h update needed for the tri ?::

   
    stree::get_mmlabel num_ridx 11
    stree::get_mmlabel ridx 0 mmlabel 2896:sWorld
    stree::get_mmlabel ridx 1 mmlabel 5:PMT_3inch_pmt_solid
    stree::get_mmlabel ridx 2 mmlabel 9:NNVTMCPPMTsMask_virtual
    stree::get_mmlabel ridx 3 mmlabel 12:HamamatsuR12860sMask_virtual
    stree::get_mmlabel ridx 4 mmlabel 4:mask_PMT_20inch_vetosMask_virtual
    stree::get_mmlabel ridx 5 mmlabel 1:sStrutBallhead
    stree::get_mmlabel ridx 6 mmlabel 1:uni1
    stree::get_mmlabel ridx 7 mmlabel 1:base_steel
    stree::get_mmlabel ridx 8 mmlabel 1:uni_acrylic1
    stree::get_mmlabel ridx 9 mmlabel 130:sPanel
    stree::get_mmlabel ridx 10 mmlabel 322:solidSJCLSanchor
    ...
    2024-08-27 16:16:57.253 FATAL [48047] [SBT::createGAS@335]  FAILED to SScene::getMeshGroup gas_idx 10
    [ SScene::desc 
     is_empty NO 
    SScene::descSize meshmerge 10 meshgroup 10 inst_info 10 inst_tran 48477
    [SScene::descInstInfo {ridx, inst_count, inst_offset, 0} 
    {  0,      1,      0,  0}
    {  1,  25600,      1,  0}
    {  2,  12615,  25601,  0}
    {  3,   4997,  38216,  0}
    {  4,   2400,  43213,  0}
    {  5,    590,  45613,  0}
    {  6,    590,  46203,  0}
    {  7,    590,  46793,  0}
    {  8,    590,  47383,  0}
    {  9,    504,  47973,  0}
    ]SScene::descInstInfo tot_inst 48477
    [SScene::descFrame num_frame 24

    0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff6b35a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff6b2d1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6b2d252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffc5c1f2d1 in SBT::createGAS (this=0x26ebdb40, gas_idx=10) at /home/blyth/opticks/CSGOptiX/SBT.cc:341
    #5  0x00007fffc5c1ed65 in SBT::createGAS (this=0x26ebdb40) at /home/blyth/opticks/CSGOptiX/SBT.cc:293
    #6  0x00007fffc5c1e72d in SBT::createGeom (this=0x26ebdb40) at /home/blyth/opticks/CSGOptiX/SBT.cc:250
    #7  0x00007fffc5c1e650 in SBT::setFoundry (this=0x26ebdb40, foundry_=0x1a7f63b0) at /home/blyth/opticks/CSGOptiX/SBT.cc:232
    #8  0x00007fffc5b6ed37 in CSGOptiX::initGeometry (this=0x25e0ab50) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:581
    #9  0x00007fffc5b6dc38 in CSGOptiX::init (this=0x25e0ab50) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:480
    #10 0x00007fffc5b6d79f in CSGOptiX::CSGOptiX (this=0x25e0ab50, foundry_=0x1a7f63b0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:454
    #11 0x00007fffc5b6ce8d in CSGOptiX::Create (fd=0x1a7f63b0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:357
    #12 0x00007fffcd2c9f73 in G4CXOpticks::setGeometry_ (this=0xaf31730, fd_=0x1a7f63b0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:316
    #13 0x00007fffcd2c9d81 in G4CXOpticks::setGeometry (this=0xaf31730, fd_=0x1a7f63b0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:283
    #14 0x00007fffcd2c9b21 in G4CXOpticks::setGeometry (this=0xaf31730, world=0x97b0140) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:257
    #15 0x00007fffcd2c81e5 in G4CXOpticks::SetGeometry (world=0x97b0140) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:58
    #16 0x00007fffbe3b6fed in LSExpDetectorConstruction_Opticks::Setup (opticksMode=1, world=0x97b0140, sd=0x999e6b0, ppd=0x5a4230, psd=0x66323b0, pmtscan=0x0)

::

     227 void SBT::setFoundry(const CSGFoundry* foundry_)
     228 {
     229     foundry = foundry_ ;          // analytic
     230     scene = foundry->getScene();  // triangulated
     231 
     232     createGeom();
     233 }


     314 #ifdef WITH_SOPTIX_ACCEL
     315 void SBT::createGAS(unsigned gas_idx)
     316 {
     317     SOPTIX_BuildInput* bi = nullptr ;
     318     SOPTIX_Accel* gas = nullptr ;
     319 
     320     bool trimesh = foundry->isSolidTrimesh(gas_idx); // now based on forced triangulation config 
     321 
     322     const std::string& mmlabel = foundry->getSolidMMLabel(gas_idx);
     323 
     324     LOG(LEVEL)
     325         << " WITH_SOPTIX_ACCEL "
     326         << " gas_idx " << gas_idx
     327         << " trimesh " << ( trimesh ? "YES" : "NO " )
     328         << " mmlabel " << mmlabel
     329         ;
     330 
     331     if(trimesh)
     332     {
     333         // note similarity to SOPTIX_Scene::init_GAS
     334         const SMeshGroup* mg = scene->getMeshGroup(gas_idx) ;
     335         LOG_IF(fatal, mg == nullptr)
     336             << " FAILED to SScene::getMeshGroup"
     337             << " gas_idx " << gas_idx
     338             << "\n"
     339             << scene->desc()
     340             ;
     341         assert(mg);
     ////  FAILING HERE 

     342 
     343         SOPTIX_MeshGroup* xmg = SOPTIX_MeshGroup::Create( mg ) ;
     344         gas = SOPTIX_Accel::Create(Ctx::context, xmg->bis );
     345         xgas[gas_idx] = xmg ;
     346     }
     347     else
     348     {
     349         // analytic geometry 
     350         SCSGPrimSpec ps = foundry->getPrimSpec(gas_idx);
     351         bi = new SOPTIX_BuildInput_CPA(ps) ;
     352         gas = SOPTIX_Accel::Create(Ctx::context, bi );
     353     }
     354     vgas[gas_idx] = gas ;
     355 }





FIXED : First issues from fail to find tri frame and stale solid layout assumption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

After enabling force triangulation for many solids get error::

    ]]stree::postcreate
    SScene::addFrames FAIL to find frame  spec [solidXJfixture:0:-1]
     line [solidXJfixture:0:-1]
    SScene::addFrames FAIL to find frame  spec [solidXJfixture:20:-1]
     line [solidXJfixture:20:-1]
    SScene::addFrames FAIL to find frame  spec [solidXJfixture:40:-1]
     line [solidXJfixture:40:-1]
    SScene::addFrames FAIL to find frame  spec [solidXJfixture:55:-1]
     line [solidXJfixture:55:-1]
    SScene::addFrames FAIL to find frame  spec [solidXJanchor:0:-1]
     line [solidXJanchor:0:-1]
    SScene::addFrames FAIL to find frame  spec [solidXJanchor:20:-1]
     line [solidXJanchor:20:-1]
    SScene::addFrames FAIL to find frame  spec [solidXJanchor:40:-1]
     line [solidXJanchor:40:-1]
    SScene::addFrames FAIL to find frame  spec [solidXJanchor:55:-1]
    ...
    SScene::addFrames FAIL to find frame  spec [sSurftube_38V1_0:0:-1]
     line [sSurftube_38V1_0:0:-1]
    SScene::addFrames FAIL to find frame  spec [sSurftube_38V1_1:0:-1]
     line [sSurftube_38V1_1:0:-1]
    SScene::addFrames FAIL to find frame  spec [solidXJfixture:27:-1]
     line [solidXJfixture:27:-1     ## near bottom of CD]
    [Detaching after fork from child process 320185]
    python: /data/blyth/opticks_Debug/include/SysRap/stree.h:3579: const sfactor& stree::get_factor(unsigned int) const: Assertion `idx < factor.size()' failed.

    Thread 1 "python" received signal SIGABRT, Aborted.
    0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    (gdb) 

    Thread 1 "python" received signal SIGABRT, Aborted.
    0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff6b35a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff6b2d1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6b2d252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffc5a0c80e in stree::get_factor (this=0xaf30780, idx=9) at /data/blyth/opticks_Debug/include/SysRap/stree.h:3579
    #5  0x00007fffc5a0c84e in stree::get_factor_subtree (this=0xaf30780, idx=9) at /data/blyth/opticks_Debug/include/SysRap/stree.h:3585
    #6  0x00007fffc5a0c935 in stree::get_ridx_subtree (this=0xaf30780, ridx=10) at /data/blyth/opticks_Debug/include/SysRap/stree.h:3611
    #7  0x00007fffc5a0a332 in stree::get_mmlabel (this=0xaf30780, names=std::vector of length 10, capacity 16 = {...})
        at /data/blyth/opticks_Debug/include/SysRap/stree.h:2053
    #8  0x00007fffc59f8678 in CSGImport::importNames (this=0x1a7e72e0) at /home/blyth/opticks/CSG/CSGImport.cc:64
    #9  0x00007fffc59f850e in CSGImport::import (this=0x1a7e72e0) at /home/blyth/opticks/CSG/CSGImport.cc:54
    #10 0x00007fffc5979cfb in CSGFoundry::importSim (this=0x1a7f4e50) at /home/blyth/opticks/CSG/CSGFoundry.cc:1660
    #11 0x00007fffc597f312 in CSGFoundry::CreateFromSim () at /home/blyth/opticks/CSG/CSGFoundry.cc:2956
    #12 0x00007fffcd2c9b07 in G4CXOpticks::setGeometry (this=0xaf30460, world=0x97aeec0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:256
    #13 0x00007fffcd2c81e5 in G4CXOpticks::SetGeometry (world=0x97aeec0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:58
    #14 0x00007fffbe3b6fed in LSExpDetectorConstruction_Opticks::Setup (opticksMode=1, world=0x97aeec0, sd=0x999d430, ppd=0x5a3e80, psd=0x66311d0, pmtscan=0x0)
        at /data/blyth/junotop/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction_Opticks.cc:56
    #15 0x00007fffbe38c0cc in LSExpDetectorConstruction::setupOpticks (this=0x95c4670, world=0x97aeec0)


::

     45 void CSGImport::import()
     46 {
     47     LOG(LEVEL) << "[" ;
     48 
     49     st = fd->sim ? fd->sim->tree : nullptr ;
     50     LOG_IF(fatal, st == nullptr) << " fd.sim(SSim) fd.st(stree) required " ;
     51     assert(st);
     52 
     53 
     54     importNames();
     55     importSolid();
     56     importInst();
     57 
     58     LOG(LEVEL) << "]" ;
     59 }

     62 void CSGImport::importNames()
     63 {
     64     st->get_mmlabel( fd->mmlabel);
     65     st->get_meshname(fd->meshname);
     66 }



Review progress
----------------

What configures force triangulation ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    stree__force_triangulate_solid


::

    epsilon:u4 blyth$ opticks-f stree__force_triangulate_solid
    ./sysrap/stree.h:    static constexpr const char* stree__force_triangulate_solid = "stree__force_triangulate_solid" ; 
    ./sysrap/stree.h:    force_triangulate_solid(ssys::getenvvar(stree__force_triangulate_solid,nullptr)), 
    ./sysrap/stree.h:Uses the optional comma delimited stree__force_triangulate_solid envvar list of unique solid names
    ./sysrap/stree.h:depending on the "stree__force_triangulate_solid" envvar list of unique solid names. 
    epsilon:opticks blyth$ 


TODO: test this with a script 



Where are nds/rem/tri collected ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

U4Tree::initNodes_r does initial collection from Geant4 into *nds*, 
subsequently the *rem* and *tri* subsets are populated by stree::collectGlobalNodes
which is invoked at the tail of stree::factorize


stree::get_ridx_type
~~~~~~~~~~~~~~~~~~~~~~~

::

    git diff ed7ced230^-1

     
         int      get_num_ridx() const ;  
    +    int      get_num_remainder() const ; 
    +    int      get_num_triangulated() const ;
    +    char     get_ridx_type(int ridx) const ;
 


where is stree::get_ridx_type used to effect the force triangulation ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First need to import the stree to form the CSGFoundry geom, made changes::

    CSGImport::importSolid
    CSGImport::importSolidGlobal
    CSGImport::importSolidFactor
    
Then need to convert from CSGFoundry geom into GAS/SBT.

* HMM: how to detect triangulated from the solid ? 
* Nope not possible directly, unless use the label eg: r0 f1 f2 f3 t4



how did the old CSGFoundry level trimesh post hoc switch to tri ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


With CSGFoundry::isSolidTrimesh::

     314 #ifdef WITH_SOPTIX_ACCEL
     315 void SBT::createGAS(unsigned gas_idx)
     316 {
     317     SOPTIX_BuildInput* bi = nullptr ;
     318     SOPTIX_Accel* gas = nullptr ;
     319 
     320     bool trimesh = foundry->isSolidTrimesh(gas_idx);  // post-hoc triangulation 
     321     const std::string& label = foundry->getSolidLabel(gas_idx);
     322 


HMM: can/should I co-opt the old CSGFoundry::isSolidTrimesh to adopt force triangulation ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* looks like it 



where are the stree::rem used ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



TODO: generalize old layout assuming code ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


eg::

     82 void CSGImport::importSolid()
     83 {
     84     int num_ridx = st->get_num_ridx() ;
     85     for(int ridx=0 ; ridx < num_ridx ; ridx++)
     86     {
     87         std::string _rlabel = CSGSolid::MakeLabel('r',ridx) ;
     88         const char* rlabel = _rlabel.c_str();
     89 
     90         if( ridx == 0 )
     91         {
     92             importSolidRemainder(ridx, rlabel );
     93         }
     94         else
     95         {
     96             importSolidFactor(ridx, rlabel );
     97         }
     98     }
     99 }





U4Tree.h
----------

U4Tree::initSolids_Mesh 
    All solids have analytic and triangulated forms. The tri/ana fork happens later.  


CSGFoundry::isSolidTrimesh HUH : TOO LATE TO DO THIS HERE ?
------------------------------------------------------------

Yep, its too late to do this within CSG. 
This was for primitive post hoc trimesh control. 

Earlier control used in stree::collectGlobalNodes

* NB simplifying assumption that all configured tri nodes are global (not instanced)


