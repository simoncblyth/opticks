X4Gen-survey
==============

Switch solid by setting LV envvar 
----------------------------------

::

     export LV=211
     x4gen-info   ## run the serialization, see the G4 code
     x4gen-csg    ## visual check 



Survey With Balancing rmx:raw-max-depth bmx:balanced-max-depth
-----------------------------------------------------------------

::

    epsilon:x4gen blyth$ grep bmx:04 solids.txt 
     so:027 lv:232 rmx:12 bmx:04 soName: near_pool_ows_box0xbf8c8a8
     so:029 lv:211 rmx:12 bmx:04 soName: near_pool_iws_box0xc288ce8
     so:067 lv:056 rmx:08 bmx:04 soName: RadialShieldUnit0xc3d7da8
     so:070 lv:057 rmx:08 bmx:04 soName: TopESRCutHols0xbf9de10
     so:111 lv:105 rmx:10 bmx:04 soName: led-source-assy0xc3061d0
     so:118 lv:112 rmx:10 bmx:04 soName: source-assy0xc2d5d78
     so:125 lv:132 rmx:10 bmx:04 soName: amcco60-source-assy0xc0b1df8
     so:153 lv:142 rmx:07 bmx:04 soName: GdsOflTnk0xc3d5160

    epsilon:x4gen blyth$ grep bmx:03 solids.txt 
     so:003 lv:000 rmx:04 bmx:03 soName: near_top_cover_box0xc23f970
     so:025 lv:236 rmx:04 bmx:03 soName: near_pool_dead_box0xbf8a280
     so:026 lv:234 rmx:04 bmx:03 soName: near_pool_liner_box0xc2dcc28
     so:028 lv:213 rmx:04 bmx:03 soName: near_pool_curtain_box0xc2cef48
     so:054 lv:047 rmx:03 bmx:03 soName: pmt-hemi0xc0fed90
     so:055 lv:046 rmx:03 bmx:03 soName: pmt-hemi-vac0xc21e248
     so:068 lv:059 rmx:05 bmx:03 soName: TopRefCutHols0xbf9bd50
     so:069 lv:058 rmx:04 bmx:03 soName: TopRefGapCutHols0xbf9cef8
     so:071 lv:062 rmx:05 bmx:03 soName: BotRefHols0xc3cd380
     so:072 lv:061 rmx:04 bmx:03 soName: BotRefGapCutHols0xc34bb28
     so:073 lv:060 rmx:07 bmx:03 soName: BotESRCutHols0xbfa7368
     so:076 lv:065 rmx:04 bmx:03 soName: SstBotCirRibBase0xc26e2d0
     so:080 lv:069 rmx:06 bmx:03 soName: SstTopCirRibBase0xc264f78
     so:110 lv:098 rmx:03 bmx:03 soName: turntable0xbf784f0
     so:149 lv:145 rmx:05 bmx:03 soName: OflTnkContainer0xc17cf50
     so:151 lv:140 rmx:04 bmx:03 soName: LsoOflTnk0xc17d928
     so:152 lv:141 rmx:03 bmx:03 soName: LsoOfl0xc348ac0
     so:208 lv:200 rmx:05 bmx:03 soName: table_panel_box0xc00f558
     so:248 lv:245 rmx:04 bmx:03 soName: near-radslab-box-90xcd31ea0




Survey ASIS : solids of depth 3 or more : without tree balancing 
------------------------------------------------------------------------

* height greater than 7 is skipped in kernel, so expect no-show for these


The two deepest trees (depth 12) are G4 polygonization skipped because it hangs for them::

    epsilon:x4gen blyth$ egrep so:027\|so:029 solids.txt
     so:027 lv:232 mx:12 soName: near_pool_ows_box0xbf8c8a8
     so:029 lv:211 mx:12 soName: near_pool_iws_box0xc288ce8


::

    epsilon:x4gen blyth$ x4gen-;x4gen-deep

     so:027 lv:232 mx:12 soName: near_pool_ows_box0xbf8c8a8
     so:029 lv:211 mx:12 soName: near_pool_iws_box0xc288ce8
     so:111 lv:105 mx:10 soName: led-source-assy0xc3061d0
     so:118 lv:112 mx:10 soName: source-assy0xc2d5d78
     so:125 lv:132 mx:10 soName: amcco60-source-assy0xc0b1df8
     so:070 lv:057 mx:08 soName: TopESRCutHols0xbf9de10            
          ## all these big ones : poly ok,  raytrace no-show  : as expected without balancing 

     so:067 lv:056 mx:08 soName: RadialShieldUnit0xc3d7da8              
          ## fails to import convexpoly planes  : for the phi segmenting 
          ## after fixing off-by-one planes issue, imports but get expected raytrace no-show
         
    ----------------------------- height greater than 7 skipped in kernel -------

     so:073 lv:060 mx:07 soName: BotESRCutHols0xbfa7368
     so:153 lv:142 mx:07 soName: GdsOflTnk0xc3d5160
          ## these two succeed to raytrace  

     so:080 lv:069 mx:06 soName: SstTopCirRibBase0xc264f78          
          ## plane import issue
          ## after fixing off-by-one plane issue raytrace succeeds, BUT obvious csg-spurions on box cuts 

     so:068 lv:059 mx:05 soName: TopRefCutHols0xbf9bd50
     so:071 lv:062 mx:05 soName: BotRefHols0xc3cd380
     so:149 lv:145 mx:05 soName: OflTnkContainer0xc17cf50
     so:208 lv:200 mx:05 soName: table_panel_box0xc00f558      ## funny shape, rectangle with square hole
     so:003 lv:000 mx:04 soName: near_top_cover_box0xc23f970
     so:025 lv:236 mx:04 soName: near_pool_dead_box0xbf8a280
     so:026 lv:234 mx:04 soName: near_pool_liner_box0xc2dcc28
     so:028 lv:213 mx:04 soName: near_pool_curtain_box0xc2cef48
     so:069 lv:058 mx:04 soName: TopRefGapCutHols0xbf9cef8
     so:072 lv:061 mx:04 soName: BotRefGapCutHols0xc34bb28
     so:151 lv:140 mx:04 soName: LsoOflTnk0xc17d928         ## interesting plate shape
     so:248 lv:245 mx:04 soName: near-radslab-box-90xcd31ea0
          ## all these have raytrace ok

     so:076 lv:065 mx:04 soName: SstBotCirRibBase0xc26e2d0    
          ## planes issue 
          ## after dixing off-by-one issue raytrace succeeds, BUT dont see any cuts in the cylinder

     so:054 lv:047 mx:03 soName: pmt-hemi0xc0fed90
     so:055 lv:046 mx:03 soName: pmt-hemi-vac0xc21e248
     so:110 lv:098 mx:03 soName: turntable0xbf784f0
     so:152 lv:141 mx:03 soName: LsoOfl0xc348ac0    ## interesting saucer shape
          ## raytrace ok

               


lv:065 has planes issue too
----------------------------

Change the tubs phi range to not segment, allows to raytrace : get cylindrical ring with box sliver cut.

::

     23 // start portion generated by nnode::to_g4code 
     24 G4VSolid* make_solid()
     25 {
     26     //G4VSolid* c = new G4Tubs("SstBotCirRibPri0xc26d4e0", 1980, 2000, 215, 0, 0.785398) ; // 2
     27     G4VSolid* c = new G4Tubs("SstBotCirRibPri0xc26d4e0", 1980, 2000, 215, 0, CLHEP::twopi) ; // 2
     28     G4VSolid* e = new G4Box("SstBotRibBase00xc0d1e90", 1010, 12.5, 220) ; // 2


lv:065 lv:056 lv:069 fails to import with plane issue
---------------------------------------------------------

Some issue with planes, which is funny as g4code looks to have no convexpolyhedrons::

    2018-07-29 16:59:56.838 INFO  [5341053] [NCSGData::loadsrc@310]  loadsrc DONE  ht  8 nn  511 snd 511,4,4 nd NULL str 17,4,4 tr NULL gtr NULL pln 5,4
    2018-07-29 16:59:56.838 ERROR [5341053] [*NCSG::import_r@547] import_r node->gtransform_idx 1
    2018-07-29 16:59:56.838 ERROR [5341053] [*NCSG::import_r@547] import_r node->gtransform_idx 1
    Assertion failed: (idx < m_num_planes), function getSrcPlanes, file /Users/blyth/opticks/npy/NCSGData.cpp, line 712.
    /Users/blyth/opticks/bin/op.sh: line 845: 82660 Abort trap: 6           /usr/local/opticks/lib/OTracerTest --rendermode +global,+axis --animtimemax 20 --timemax 20 --geocenter --eye 1,0,0 --dbganalytic --test --testconfig analytic=1_csgpath=/tmp/blyth/opticks/x4gen/x056 --tracer --printenabled
    /Users/blyth/opticks/bin/op.sh RC 134
    epsilon:x4gen blyth$ 

Ahha it does have convexpoly from the phi segmenting::

    epsilon:x4gen blyth$ x4gen-csg -D
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff74570b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff7473b080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff744cc1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff744941ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000104e4abbd libNPY.dylib`NCSGData::getSrcPlanes(this=0x000000010f5e3780, _planes=size=1, idx=4294967295, num_plane=5)0>, std::__1::allocator<glm::tvec4<float, (glm::precision)0> > >&, unsigned int, unsigned int) const at NCSGData.cpp:712
        frame #5: 0x0000000104e3e992 libNPY.dylib`NCSG::import_srcplanes(this=0x000000010f5e36e0, node=0x000000010afa9fb0) at NCSG.cpp:702
        frame #6: 0x0000000104e3dc7c libNPY.dylib`NCSG::import_primitive(this=0x000000010f5e36e0, idx=128, typecode=CSG_CONVEXPOLYHEDRON) at NCSG.cpp:638
        ...
        frame #26: 0x00000001000d3d7b libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe4c8, argc=17, argv=0x00007ffeefbfe5a8, argforced="--tracer") at OKMgr.cc:44
        frame #27: 0x00000001000d41bb libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe4c8, argc=17, argv=0x00007ffeefbfe5a8, argforced="--tracer") at OKMgr.cc:52
        frame #28: 0x000000010000b9b4 OTracerTest`main(argc=17, argv=0x00007ffeefbfe5a8) at OTracerTest.cc:19
        frame #29: 0x00007fff74420015 libdyld.dylib`start + 1
        frame #30: 0x00007fff74420015 libdyld.dylib`start + 1

investigate plane loading issue elsewhere
-------------------------------------------

* :doc:`X4SolidLoadTest_plane_imports_failing`



Hmm lots are no-show : this is without balancing ? YES
----------------------------------------------------------

With *--printenabled* get loadsa::

    evaluative_csg tranOffset 1 numParts 2047 perfect tree height 10 exceeds current limit

This coming from::

     544 static __device__
     545 void evaluative_csg( const Prim& prim, const uint4& identity )
     546 {
     547     unsigned partOffset = prim.partOffset() ;
     548     unsigned numParts   = prim.numParts() ;
     549     unsigned tranOffset = prim.tranOffset() ;
     550 
     551     unsigned height = TREE_HEIGHT(numParts) ; // 1->0, 3->1, 7->2, 15->3, 31->4 
     552 
     553 #ifdef USE_TWIDDLE_POSTORDER
     554     // bit-twiddle postorder limited to height 7, ie maximum of 0xff (255) nodes
     555     // (using 2-bytes with PACK2 would bump that to 0xffff (65535) nodes)
     556     // In any case 0xff nodes are far more than this is expected to be used with
     557     //
     558     if(height > 7)
     559     {
     560         rtPrintf("evaluative_csg tranOffset %u numParts %u perfect tree height %u exceeds current limit\n", tranOffset, numParts, height ) ;
     561         return ;
     562     }
     563 #else




And can see the overlarge part counts in OGeo::makeAnalyticGeometry. 


::

    2018-07-29 16:26:27.072 INFO  [5324916] [GParts::dump@1470] OGeo::makeAnalyticGeometry --dbganalytic lim 10 pbuf 2048,4,4
    2018-07-29 16:26:27.072 INFO  [5324916] [GParts::dumpPrimInfo@1242] OGeo::makeAnalyticGeometry --dbganalytic (part_offset, parts_for_prim, tran_offset, plan_offset)  numPrim: 2 ulim: 2
    2018-07-29 16:26:27.072 INFO  [5324916] [GParts::dumpPrimInfo@1253]  (   0    1    0    0) 
    2018-07-29 16:26:27.072 INFO  [5324916] [GParts::dumpPrimInfo@1253]  (   1 2047    1    0) 
    2018-07-29 16:26:27.072 INFO  [5324916] [GParts::dump@1487] GParts::dump ni 2048 lim 10 ulim 10

::

    2018-07-29 16:32:47.824 INFO  [5327295] [OContext::close@251] OContext::close m_cfg->apply() done.
    // intersect_analytic.cu:bounds buffer sizes pts:2048 pln:   0 trs:  36 
    //csg_bounds_prim CSG_FLAGNODETREE  primIdx   0 partOffset   0  numParts   1 -> height  0 -> numNodes  1  tranBuffer_size  36 
    //csg_bounds_prim CSG_FLAGNODETREE  primIdx   1 partOffset   1  numParts 2047 -> height 10 -> numNodes 2047  tranBuffer_size  36 
    ## csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius  10.035 z1 -14.865 z2  14.865 
    ## csg_bounds_zsphere  zmin  -0.000 zmax  10.035 flags 3 QCAP(zmin) 1 PCAP(zmax) 1  
    ## csg_bounds_zsphere  zmin  -0.000 zmax  10.035 flags 3 QCAP(zmin) 1 PCAP(zmax) 1  
    ## csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius  10.035 z1 -18.475 z2  18.475 
    ## csg_bounds_zsphere  zmin  -0.000 zmax  10.035 flags 3 QCAP(zmin) 1 PCAP(zmax) 1  
    ## csg_bounds_zsphere  zmin  -0.000 zmax  10.035 flags 3 QCAP(zmin) 1 PCAP(zmax) 1  
    ## csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius   0.300 z1 -12.700 z2  12.700 
    ## csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius  10.035 z1 -18.475 z2  18.475 
    ## csg_bounds_zsphere  zmin  -0.000 zmax  10.035 flags 3 QCAP(zmin) 1 PCAP(zmax) 1  
    ## csg_bounds_zsphere  zmin  -0.000 zmax  10.035 flags 3 QCAP(zmin) 1 PCAP(zmax) 1  
    ## csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius   0.300 z1 -12.700 z2  12.700 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 0  bnd0:  0 typ0: 17  min  -162.2202  -162.2202  -162.2202 max   162.2202   162.2202   162.2202 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 1  bnd0:  1 typ0:  1  min   -10.0350   -10.0350   -97.2850 max    10.0350    10.0350   107.3200 
    evaluative_csg tranOffset 1 numParts 2047 perfect tree height 10 exceeds current limit
    evaluative_csg tranOffset 1 numParts 2047 perfect tree height 10 exceeds current limit
    evaluative_csg tranOffset 1 numParts 2047 perfect tree height 10 exceeds current limit
    evaluative_csg tranOffset 1 numParts 2047 perfect tree height 10 exceeds current limit
    evaluative_csg tranOffset 1 numParts 2047 perfect tree height 10 exceeds current limit
    evaluative_csg tranOffset 1 numParts 2047 perfect tree height 10 exceeds current limit
    evaluative_csg tranOffset 1 numParts 2047 perfect tree height 10 exceeds current limit






