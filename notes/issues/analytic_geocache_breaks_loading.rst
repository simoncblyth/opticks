Analytic geocache breaks geometry loading
=============================================

Testing with DYB
------------------

::

    op -G --gltf 3 --debugger

    op --gltf 3 --debugger


FIXED (?)
----------


* may be fixed by avoiding the duplicity



Observations
---------------

* root cause of the difficulties is that GTreeCheck and GScene are being duplicitous, 
  need to factor out the common elements into another class... to avoid the friction

  * ideally rename GTreeCheck to GTree and use it from GScene, but thats
    too much work... (non-adiabatic) easier to pluck off common pieces 
    and put them into other classes 




Issues
--------

Getting different numbers of repeaters in analytic and triangulated branches... 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* rejig of providing NSceneConfig to both GTreeCheck and NScene gets closer...

::

    simon:ggeo blyth$ GGeoLibTest 
    2017-08-29 13:45:09.765 INFO  [1499169] [GGeoLib::loadConstituents@109] GGeoLib::loadConstituents idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-08-29 13:45:09.766 INFO  [1499169] [GGeoLib::loadConstituents@114] GGeoLib::loadConstituents mm.reldir GMergedMesh gp.reldir GParts MAX_MERGED_MESH  10
    2017-08-29 13:45:09.895 INFO  [1499169] [GGeoLib::loadConstituents@152] GGeoLib::loadConstituents loaded 6 ridx (  0,  1,  2,  3,  4,  5,)
    2017-08-29 13:45:09.895 INFO  [1499169] [GGeoLib::dump@238] geolib
    2017-08-29 13:45:09.895 INFO  [1499169] [GGeoLib::dump@239] GGeoLib TRIANGULATED  numMergedMesh 6
    mm i   0 geocode   T                  numSolids      12230 numFaces      403712 numITransforms           1
    mm i   1 geocode   T            EMPTY numSolids          1 numFaces           0 numITransforms        1792
    mm i   2 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   3 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   4 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   5 geocode   T                  numSolids          5 numFaces        2928 numITransforms         672
    2017-08-29 13:45:09.895 INFO  [1499169] [GGeoLib::loadConstituents@109] GGeoLib::loadConstituents idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-08-29 13:45:09.895 INFO  [1499169] [GGeoLib::loadConstituents@114] GGeoLib::loadConstituents mm.reldir GMergedMeshAnalytic gp.reldir GPartsAnalytic MAX_MERGED_MESH  10
    2017-08-29 13:45:10.049 INFO  [1499169] [GGeoLib::loadConstituents@152] GGeoLib::loadConstituents loaded 5 ridx (  0,  2,  3,  4,  5,)
    2017-08-29 13:45:10.049 INFO  [1499169] [GGeoLib::dump@238] geolib_analytic
    2017-08-29 13:45:10.049 INFO  [1499169] [GGeoLib::dump@239] GGeoLib ANALYTIC  numMergedMesh 5
    mm i   0 geocode   T                  numSolids      12230 numFaces      403712 numITransforms           1
    mm i   1 geocode   - NULL             numSolids 4294967295 numFaces  4294967295 numITransforms  4294967295
    mm i   2 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   3 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   4 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    2017-08-29 13:45:10.049 INFO  [1499169] [GMesh::makeFaceRepeatedIdentityBuffer@2061] GMesh::makeFaceRepeatedIdentityBuffer numSolids 12230 numFaces (sum of faces in numSolids)403712
     ----- 403712 
    2017-08-29 13:45:10.077 INFO  [1499169] [GMesh::makeFaceRepeatedIdentityBuffer@2061] GMesh::makeFaceRepeatedIdentityBuffer numSolids 12230 numFaces (sum of faces in numSolids)403712
     ----- 403712 
    simon:ggeo blyth$ 



::

    2017-08-29 13:55:07.693 INFO  [1502772] [GTreeCheck::dumpRepeatCandidates@307] GTreeCheck::dumpRepeatCandidates 
     pdig c4ca4238a0b923820dcc509a6f75849b ndig   1792 nprog      0 placements   1792 n __dd__Geometry__RPC__lvRPCStrip0xc2213c0
     pdig 85d8ce590ad8981ca2c8286f79f59954 ndig    864 nprog      0 placements    864 n __dd__Geometry__PMT__lvMountRib20xc012500
     pdig 0e65972dce68dad4d52d063967f0a705 ndig    864 nprog      0 placements    864 n __dd__Geometry__PMT__lvMountRib30xc00d350
     pdig 0336dcbab05b9d5ad24f4333c7658a0e ndig    864 nprog      0 placements    864 n __dd__Geometry__PMT__lvMountRib10xc3a4cb0
     pdig dcc7055c92a21388d347fc2e03d9ee52 ndig    672 nprog      4 placements    672 n __dd__Geometry__PMT__lvPmtHemi0xc133740
    2017-08-29 13:55:07.702 INFO  [1502772] [GTreeCheck::labelTree@379] GTreeCheck::labelTree count of non-zero setRepeatIndex 7744
    2017-08-29 13:55:07.984 INFO  [1502772] [GTreeCheck::makeAnalyticInstanceIdentityBuffer@523] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 0 numPlacements 1 numSolids 12230
    2017-08-29 13:55:07.987 WARN  [1502772] [GMesh::allocate@627] GMesh::allocate EMPTY numVertices 0 numFaces 0 numSolids 1
    2017-08-29 13:55:07.991 INFO  [1502772] [GTreeCheck::makeAnalyticInstanceIdentityBuffer@523] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 1 numPlacements 1792 numSolids 1
    2017-08-29 13:55:07.996 INFO  [1502772] [GTreeCheck::makeAnalyticInstanceIdentityBuffer@523] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 2 numPlacements 864 numSolids 1
    2017-08-29 13:55:08.000 INFO  [1502772] [GTreeCheck::makeAnalyticInstanceIdentityBuffer@523] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 3 numPlacements 864 numSolids 1
    2017-08-29 13:55:08.005 INFO  [1502772] [GTreeCheck::makeAnalyticInstanceIdentityBuffer@523] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 4 numPlacements 864 numSolids 1
    2017-08-29 13:55:08.009 INFO  [1502772] [GTreeCheck::makeAnalyticInstanceIdentityBuffer@523] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 5 numPlacements 672 numSolids 5
    2017-08-29 13:55:08.010 INFO  [1502772] [TimesTable::dump@103] Timer::dump filter: NONE
              0.000      t_absolute        t_delta
              0.033           0.033          0.033 : deltacheck
              0.568           0.601          0.568 : traverse
              0.004           0.604          0.004 : labelTree
              0.308           0.913          0.308 : makeMergedMeshAndInstancedBuffers
    2017-08-29 13:55:08.010 INFO  [1502772] [GColorizer::traverse@93] GColorizer::traverse START
    2017-08-29 13:55:08.342 INFO  [1502772] [GColorizer::traverse@97] GColorizer::traverse colorized nodes 0
    2017-08-29 13:55:08.342 INFO  [1502772] [GGeo::save@635] GGeo::save
    2017-08-29 13:55:08.342 INFO  [1502772] [GGeoLib::dump@238] GGeo::save.geolib
    2017-08-29 13:55:08.342 INFO  [1502772] [GGeoLib::dump@239] GGeoLib TRIANGULATED  numMergedMesh 6
    mm i   0 geocode   T                  numSolids      12230 numFaces      403712 numITransforms           1
    mm i   1 geocode   T            EMPTY numSolids          1 numFaces           0 numITransforms        1792
    mm i   2 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   3 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   4 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   5 geocode   T                  numSolids          5 numFaces        2928 numITransforms         672



    2017-08-29 13:55:10.108 INFO  [1502772] [NScene::dumpRepeatCount@1430] NScene::dumpRepeatCount m_verbosity 1
     ridx   1 count  1792
     ridx   2 count   864
     ridx   3 count   864
     ridx   4 count   864
     ridx   5 count  3360


    In [2]: 672*5
    Out[2]: 3360


    Lost one ?

    2017-08-29 13:55:17.406 INFO  [1502772] [GGeoLib::dump@239] GGeoLib ANALYTIC  numMergedMesh 5
    mm i   0 geocode   A                  numSolids      12230 numFaces      403712 numITransforms           1
    mm i   1 geocode   - NULL             numSolids 4294967295 numFaces  4294967295 numITransforms  4294967295
    mm i   2 geocode   A                  numSolids          1 numFaces          12 numITransforms         864
    mm i   3 geocode   A                  numSolids          1 numFaces          12 numITransforms         864
    mm i   4 geocode   A                  numSolids          1 numFaces          12 numITransforms         864




Some RPC repeaters have no instances due to selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* suspect the vertex_min criteria was really to kill the RPC repeaters to avoid this



Some identity info is missing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::


    2017-08-29 11:49:34.056 INFO  [1438372] [GGeoLib::dump@235] GGeoLib ANALYTIC  numMergedMesh 10
    mm i   0 geocode   T                  numSolids      12230 numFaces       46988 numITransforms           1
    mm i   1 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   2 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   3 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   4 geocode   T                  numSolids          5 numFaces        2928 numITransforms         672
    mm i   5 geocode   T                  numSolids          1 numFaces         192 numITransforms         384
    mm i   6 geocode   T                  numSolids          1 numFaces          28 numITransforms         330
    mm i   7 geocode   T                  numSolids          1 numFaces          36 numITransforms         288
    mm i   8 geocode   T                  numSolids          1 numFaces         192 numITransforms         288
    mm i   9 geocode   T                  numSolids          1 numFaces         192 numITransforms         288
    2017-08-29 11:49:34.061 FATAL [1438372] [*GMesh::makeFaceRepeatedInstancedIdentityBuffer@1873] GMesh::makeFaceRepeatedInstancedIdentityBuffer iidentity_ok 0 iidentity_buffer_items 256 numFaces (sum of faces in numSolids)46988 numITransforms 1 numSolids*numITransforms 12230 numRepeatedIdentity 46988
    Assertion failed: (iidentity_ok), function makeFaceRepeatedInstancedIdentityBuffer, file /Users/blyth/opticks/ggeo/GMesh.cc, line 1884.



Tri Identity Investigation
----------------------------

::

    simon:optickscore blyth$ op  --gltf 3 --debugger
    ubin /usr/local/opticks/lib/OKTest cfm cmdline --gltf 3 --debugger
    ...
    2017-08-29 13:03:14.150 INFO  [1483950] [GGeo::loadAnalyticFromCache@675] GGeo::loadAnalyticFromCache START
    2017-08-29 13:03:14.150 INFO  [1483950] [GGeoLib::loadConstituents@109] GGeoLib::loadConstituents idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-08-29 13:03:14.150 INFO  [1483950] [GGeoLib::loadConstituents@114] GGeoLib::loadConstituents mm.reldir GMergedMeshAnalytic gp.reldir GPartsAnalytic MAX_MERGED_MESH  10
    2017-08-29 13:03:14.307 INFO  [1483950] [GGeoLib::loadConstituents@152] GGeoLib::loadConstituents loaded 5 ridx (  0,  2,  3,  4,  5,)
    2017-08-29 13:03:14.364 INFO  [1483950] [GGeo::loadAnalyticFromCache@680] GGeo::loadAnalyticFromCache DONE
    2017-08-29 13:03:14.365 INFO  [1483950] [GGeo::loadAnalyticPmt@789] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0
    2017-08-29 13:03:14.375 INFO  [1483950] [GGeo::loadGeometry@593] GGeo::loadGeometry DONE
    2017-08-29 13:03:14.375 INFO  [1483950] [OpticksGeometry::loadGeometryBase@258] OpticksGeometry::loadGeometryBase DONE 
    2017-08-29 13:03:14.375 INFO  [1483950] [OpticksGeometry::loadGeometry@217] OpticksGeometry::loadGeometry DONE 
    2017-08-29 13:03:14.375 INFO  [1483950] [OpticksHub::loadGeometry@257] OpticksHub::loadGeometry DONE
    2017-08-29 13:03:15.532 INFO  [1483950] [OGeo::convert@169] OGeo::convert START  numMergedMesh: 5
    2017-08-29 13:03:15.532 INFO  [1483950] [GGeoLib::dump@238] OGeo::convert GGeoLib
    2017-08-29 13:03:15.532 INFO  [1483950] [GGeoLib::dump@239] GGeoLib ANALYTIC  numMergedMesh 5
    mm i   0 geocode   T                  numSolids      12230 numFaces      403712 numITransforms           1
    mm i   1 geocode   - NULL             numSolids 4294967295 numFaces  4294967295 numITransforms  4294967295
    mm i   2 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   3 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    mm i   4 geocode   T                  numSolids          1 numFaces          12 numITransforms         864
    ...

    2017-08-29 13:03:15.537 WARN  [1483950] [GMesh::makeFaceRepeatedIdentityBuffer@2053] GMesh::makeFaceRepeatedIdentityBuffer only relevant to non-instanced meshes 
    Assertion failed: (id), function makeTriangulatedGeometry, file /Users/blyth/opticks/optixrap/OGeo.cc, line 631.
    Process 9298 stopped
    * thread #1: tid = 0x16a4ae, 0x00007fff95594866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff95594866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff95594866:  jae    0x7fff95594870            ; __pthread_kill + 20
       0x7fff95594868:  movq   %rax, %rdi
       0x7fff9559486b:  jmp    0x7fff95591175            ; cerror_nocancel
       0x7fff95594870:  retq   
    (lldb) bt
    * thread #1: tid = 0x16a4ae, 0x00007fff95594866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff95594866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8cc3135c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff93981b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff9394b9bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x000000010357b8d9 libOptiXRap.dylib`OGeo::makeTriangulatedGeometry(this=0x0000000105ddc900, mm=0x00000001094e10f0) + 2041 at OGeo.cc:631
        frame #5: 0x0000000103579aff libOptiXRap.dylib`OGeo::makeGeometry(this=0x0000000105ddc900, mergedmesh=0x00000001094e10f0) + 127 at OGeo.cc:458
        frame #6: 0x0000000103578c67 libOptiXRap.dylib`OGeo::convertMergedMesh(this=0x0000000105ddc900, i=0) + 679 at OGeo.cc:234
        frame #7: 0x00000001035783f4 libOptiXRap.dylib`OGeo::convert(this=0x0000000105ddc900) + 340 at OGeo.cc:176
        frame #8: 0x000000010357179e libOptiXRap.dylib`OScene::init(this=0x0000000105db7420) + 5790 at OScene.cc:152
        frame #9: 0x0000000103570099 libOptiXRap.dylib`OScene::OScene(this=0x0000000105db7420, hub=0x0000000105e00180) + 313 at OScene.cc:84
        frame #10: 0x0000000103571d5d libOptiXRap.dylib`OScene::OScene(this=0x0000000105db7420, hub=0x0000000105e00180) + 29 at OScene.cc:86
        frame #11: 0x0000000103b03bf6 libOKOP.dylib`OpEngine::OpEngine(this=0x0000000105db73c0, hub=0x0000000105e00180) + 182 at OpEngine.cc:43
        frame #12: 0x0000000103b03f1d libOKOP.dylib`OpEngine::OpEngine(this=0x0000000105db73c0, hub=0x0000000105e00180) + 29 at OpEngine.cc:55
        frame #13: 0x0000000103bf2a44 libOK.dylib`OKPropagator::OKPropagator(this=0x0000000105db7380, hub=0x0000000105e00180, idx=0x000000010af5e320, viz=0x000000010af5fa60) + 196 at OKPropagator.cc:44
        frame #14: 0x0000000103bf2bbd libOK.dylib`OKPropagator::OKPropagator(this=0x0000000105db7380, hub=0x0000000105e00180, idx=0x000000010af5e320, viz=0x000000010af5fa60) + 45 at OKPropagator.cc:52
        frame #15: 0x0000000103bf2377 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfead8, argc=4, argv=0x00007fff5fbfebb0, argforced=0x0000000000000000) + 663 at OKMgr.cc:43
        frame #16: 0x0000000103bf264b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfead8, argc=4, argv=0x00007fff5fbfebb0, argforced=0x0000000000000000) + 43 at OKMgr.cc:49
        frame #17: 0x000000010000adad OKTest`main(argc=4, argv=0x00007fff5fbfebb0) + 1373 at OKTest.cc:58
        frame #18: 0x00007fff90a075fd libdyld.dylib`start + 1
    (lldb) f 7
    frame #7: 0x00000001035783f4 libOptiXRap.dylib`OGeo::convert(this=0x0000000105ddc900) + 340 at OGeo.cc:176
       173  
       174      for(unsigned i=0 ; i < nmm ; i++)
       175      {
    -> 176          convertMergedMesh(i);
       177      }
       178  
       179      // all group and geometry_group need to have distinct acceleration structures
    (lldb) p i
    (unsigned int) $0 = 0
    (lldb) f 6
    frame #6: 0x0000000103578c67 libOptiXRap.dylib`OGeo::convertMergedMesh(this=0x0000000105ddc900, i=0) + 679 at OGeo.cc:234
       231  
       232      if( i == 0 )
       233      {
    -> 234          optix::Geometry gmm = makeGeometry(mm);
       235          optix::Material mat = makeMaterial();
       236          optix::GeometryInstance gi = makeGeometryInstance(gmm,mat);
       237          gi["instance_index"]->setUint( 0u );  // so same code can run Instanced or not 
    (lldb) f 5
    frame #5: 0x0000000103579aff libOptiXRap.dylib`OGeo::makeGeometry(this=0x0000000105ddc900, mergedmesh=0x00000001094e10f0) + 127 at OGeo.cc:458
       455      const char geocode = mergedmesh->getGeoCode();
       456      if(geocode == OpticksConst::GEOCODE_TRIANGULATED)
       457      {
    -> 458          geometry = makeTriangulatedGeometry(mergedmesh);
       459      }
       460      else if(geocode == OpticksConst::GEOCODE_ANALYTIC)
       461      {
    (lldb) f 4
    frame #4: 0x000000010357b8d9 libOptiXRap.dylib`OGeo::makeTriangulatedGeometry(this=0x0000000105ddc900, mm=0x00000001094e10f0) + 2041 at OGeo.cc:631
       628     else
       629     {
       630          id = mm->getFaceRepeatedIdentityBuffer();
    -> 631          assert(id);
       632          LOG(trace) << "OGeo::makeTriangulatedGeometry using FaceRepeatedIdentityBuffer"
       633                    << " frid items " << id->getNumItems() 
       634                    << " numFaces " << numFaces
    (lldb) 







OGeo::


    616     GBuffer* id = NULL ;
    617     if(numITransforms > 1)  //  formerly 0 
    618     {
    619         id = mm->getFaceRepeatedInstancedIdentityBuffer();
    620         assert(id);
    621         LOG(trace) << "OGeo::makeTriangulatedGeometry using FaceRepeatedInstancedIdentityBuffer"
    622                   << " friid items " << id->getNumItems()
    623                   << " numITransforms*numFaces " << numITransforms*numFaces
    624                   ;
    625 
    626         assert( id->getNumItems() == numITransforms*numFaces );
    627    }
    628    else
    629    {
    630         id = mm->getFaceRepeatedIdentityBuffer();
    631         assert(id);
    632         LOG(trace) << "OGeo::makeTriangulatedGeometry using FaceRepeatedIdentityBuffer"
    633                   << " frid items " << id->getNumItems()
    634                   << " numFaces " << numFaces
    635                   ;
    636         assert( id->getNumItems() == numFaces );
    637    }



Probably the GMesh::makeFaceRepeatedInstancedIdentityBuffer depending on something 
only available pre-cache ? 

::

    1834 GBuffer* GMesh::makeFaceRepeatedInstancedIdentityBuffer()


    368   public:
    369       // transient buffers, not persisted : providing node level info in a face level buffer by repetition
    370       GBuffer* getFaceRepeatedInstancedIdentityBuffer();
    371       GBuffer* getFaceRepeatedIdentityBuffer();
    372       GBuffer* getAnalyticGeometryBuffer();
    373   private:
    374       GBuffer* makeFaceRepeatedInstancedIdentityBuffer();
    375       GBuffer* makeFaceRepeatedIdentityBuffer();
    376       GBuffer* loadAnalyticGeometryBuffer(const char* path);
    377 




