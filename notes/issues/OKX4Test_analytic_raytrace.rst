OKX4Test_analytic_raytrace
============================

Try switching in analytic raytrace by changing 
GMesh default m_geocode to 'A' (rather than 'T').




xanalytic switch
-------------------

::


   epsilon:issues blyth$ lldb OKX4Test -- --xanalytic --restrictmesh 0 

   epsilon:issues blyth$ lldb OKX4Test -- --xanalytic  



::

    2018-07-01 15:36:17.102 INFO  [2025080] [Interactor::key_pressed@409] Interactor::key_pressed O nextRenderStyle 
    2018-07-01 15:36:17.231 INFO  [2025080] [OTracer::trace_@128] OTracer::trace  entry_index 0 trace_count 0 resolution_scale 1 size(2880,1704) ZProj.zw (-1.04459,-2229.5) front 0.7071,0.7071,0.0000
    2018-07-01 15:36:17.231 INFO  [2025080] [OContext::close@236] OContext::close numEntryPoint 1
    2018-07-01 15:36:17.242 INFO  [2025080] [OContext::close@240] OContext::close setEntryPointCount done.
    2018-07-01 15:36:17.265 INFO  [2025080] [OContext::close@246] OContext::close m_cfg->apply() done.
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)
    Process 79820 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff734e6b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff734e6b6e <+10>: jae    0x7fff734e6b78            ; <+20>
        0x7fff734e6b70 <+12>: movq   %rax, %rdi
        0x7fff734e6b73 <+15>: jmp    0x7fff734ddb00            ; cerror_nocancel
        0x7fff734e6b78 <+20>: retq   
    Target 0: (OKX4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff734e6b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff736b1080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff734421ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff71346f8f libc++abi.dylib`abort_message + 245
        frame #4: 0x00007fff71347113 libc++abi.dylib`default_terminate_handler() + 241
        frame #5: 0x00007fff7277eeab libobjc.A.dylib`_objc_terminate() + 105
        frame #6: 0x00007fff713627c9 libc++abi.dylib`std::__terminate(void (*)()) + 8
        frame #7: 0x00007fff7136226f libc++abi.dylib`__cxa_throw + 121
        frame #8: 0x00000001004b8ea6 libOptiXRap.dylib`optix::ContextObj::checkError(this=0x0000000120ac0710, code=RT_ERROR_UNKNOWN) const at optixpp_namespace.h:1963
        frame #9: 0x00000001004cd7a0 libOptiXRap.dylib`optix::ContextObj::launch(this=0x0000000120ac0710, entry_point_index=0, image_width=2880, image_height=1704) at optixpp_namespace.h:2536
        frame #10: 0x00000001004cd613 libOptiXRap.dylib`OContext::launch_(this=0x0000000120a5d050, entry=0, width=2880, height=1704) at OContext.cc:330
        frame #11: 0x00000001004cd106 libOptiXRap.dylib`OContext::launch(this=0x0000000120a5d050, lmode=30, entry=0, width=2880, height=1704, times=0x0000000136ea11f0) at OContext.cc:289
        frame #12: 0x00000001004df997 libOptiXRap.dylib`OTracer::trace_(this=0x00000001310a4bf0) at OTracer.cc:142
        frame #13: 0x0000000100131925 libOpticksGL.dylib`OKGLTracer::render(this=0x000000012fbdc100) at OKGLTracer.cc:165
        frame #14: 0x00000001001c7001 libOGLRap.dylib`OpticksViz::render(this=0x00000001204b92d0) at OpticksViz.cc:432
        frame #15: 0x00000001001c5c12 libOGLRap.dylib`OpticksViz::renderLoop(this=0x00000001204b92d0) at OpticksViz.cc:474
        frame #16: 0x00000001001c5352 libOGLRap.dylib`OpticksViz::visualize(this=0x00000001204b92d0) at OpticksViz.cc:135
        frame #17: 0x000000010010a4fd libOK.dylib`OKMgr::visualize(this=0x00007ffeefbfe1f0) at OKMgr.cc:121
        frame #18: 0x0000000100014999 OKX4Test`main(argc=2, argv=0x00007ffeefbfea20) at OKX4Test.cc:86
        frame #19: 0x00007fff73396015 libdyld.dylib`start + 1
    (lldb) 




Restrictmesh succeeds to focus on one mesh : hmm but it has to be mm0 
-------------------------------------------------------------------------

::

    OKX4Test --restrictmesh 0

    lldb OKX4Test -- --restrictmesh 5    



* switching back to GMesh 'T' works and shows the expected raytrace without PMTs 



::

    2018-07-01 14:56:45.310 INFO  [1936122] [OGeo::convert@172] OGeo::convert START  numMergedMesh: 6
    2018-07-01 14:56:45.310 INFO  [1936122] [GGeoLib::dump@321] OGeo::convert GGeoLib
    2018-07-01 14:56:45.310 INFO  [1936122] [GGeoLib::dump@322] GGeoLib TRIANGULATED  numMergedMesh 6 ptr 0x7fb1e6e1ab70
    mm i   0 geocode   A                  numVolumes      12230 numFaces      459328 numITransforms           1 numITransforms*numVolumes       12230
    mm i   1 geocode   K      SKIP  EMPTY numVolumes          1 numFaces           0 numITransforms        1792 numITransforms*numVolumes        1792
    mm i   2 geocode   K      SKIP        numVolumes          1 numFaces          12 numITransforms         864 numITransforms*numVolumes         864
    mm i   3 geocode   K      SKIP        numVolumes          1 numFaces          12 numITransforms         864 numITransforms*numVolumes         864
    mm i   4 geocode   K      SKIP        numVolumes          1 numFaces          12 numITransforms         864 numITransforms*numVolumes         864
    mm i   5 geocode   K      SKIP        numVolumes          5 numFaces        2976 numITransforms         672 numITransforms*numVolumes        3360
     num_total_volumes 12230 num_instanced_volumes 7744 num_global_volumes 4486
    2018-07-01 14:56:45.310 INFO  [1936122] [OGeo::makeGeometry@595] OGeo::makeGeometry geocode A
    2018-07-01 14:56:45.310 INFO  [1936122] [GParts::close@865] GParts::close START  verbosity 0


But gives a launch crash::


    2018-07-01 15:00:57.533 INFO  [1938253] [OContext::close@246] OContext::close m_cfg->apply() done.
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)
    Process 67448 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff734e6b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff734e6b6e <+10>: jae    0x7fff734e6b78            ; <+20>
        0x7fff734e6b70 <+12>: movq   %rax, %rdi
        0x7fff734e6b73 <+15>: jmp    0x7fff734ddb00            ; cerror_nocancel
        0x7fff734e6b78 <+20>: retq   
    Target 0: (OKX4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff734e6b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff736b1080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff734421ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff71346f8f libc++abi.dylib`abort_message + 245
        frame #4: 0x00007fff71347113 libc++abi.dylib`default_terminate_handler() + 241
        frame #5: 0x00007fff7277eeab libobjc.A.dylib`_objc_terminate() + 105
        frame #6: 0x00007fff713627c9 libc++abi.dylib`std::__terminate(void (*)()) + 8
        frame #7: 0x00007fff7136226f libc++abi.dylib`__cxa_throw + 121
        frame #8: 0x00000001004b8f76 libOptiXRap.dylib`optix::ContextObj::checkError(this=0x000000011c568540, code=RT_ERROR_UNKNOWN) const at optixpp_namespace.h:1963
        frame #9: 0x00000001004cd870 libOptiXRap.dylib`optix::ContextObj::launch(this=0x000000011c568540, entry_point_index=0, image_width=2880, image_height=1704) at optixpp_namespace.h:2536
        frame #10: 0x00000001004cd6e3 libOptiXRap.dylib`OContext::launch_(this=0x000000011c5615b0, entry=0, width=2880, height=1704) at OContext.cc:330
        frame #11: 0x00000001004cd1d6 libOptiXRap.dylib`OContext::launch(this=0x000000011c5615b0, lmode=30, entry=0, width=2880, height=1704, times=0x000000013042e920) at OContext.cc:289
        frame #12: 0x00000001004dfa67 libOptiXRap.dylib`OTracer::trace_(this=0x000000013042da60) at OTracer.cc:142
        frame #13: 0x0000000100131925 libOpticksGL.dylib`OKGLTracer::render(this=0x0000000130082590) at OKGLTracer.cc:165
        frame #14: 0x00000001001c7001 libOGLRap.dylib`OpticksViz::render(this=0x000000011f7189a0) at OpticksViz.cc:432
        frame #15: 0x00000001001c5c12 libOGLRap.dylib`OpticksViz::renderLoop(this=0x000000011f7189a0) at OpticksViz.cc:474
        frame #16: 0x00000001001c5352 libOGLRap.dylib`OpticksViz::visualize(this=0x000000011f7189a0) at OpticksViz.cc:135
        frame #17: 0x000000010010a4fd libOK.dylib`OKMgr::visualize(this=0x00007ffeefbfe240) at OKMgr.cc:121
        frame #18: 0x0000000100014999 OKX4Test`main(argc=3, argv=0x00007ffeefbfea78) at OKX4Test.cc:86
        frame #19: 0x00007fff73396015 libdyld.dylib`start + 1
        frame #20: 0x00007fff73396015 libdyld.dylib`start + 1
    (lldb) 



Get crash in OGeo geometry conversion
-----------------------------------------

* perhaps from inconsistency with analytic toggle ?


::

    2018-07-01 14:41:35.396 INFO  [1929481] [OScene::init@130] OScene::init ggeobase identifier : GGeo
    2018-07-01 14:41:35.396 WARN  [1929481] [OColors::convert@30] OColors::convert SKIP no composite color buffer 
    2018-07-01 14:41:35.426 INFO  [1929481] [OGeo::convert@172] OGeo::convert START  numMergedMesh: 6
    2018-07-01 14:41:35.426 INFO  [1929481] [GGeoLib::dump@321] OGeo::convert GGeoLib
    2018-07-01 14:41:35.426 INFO  [1929481] [GGeoLib::dump@322] GGeoLib TRIANGULATED  numMergedMesh 6 ptr 0x1144644a0
    mm i   0 geocode   A                  numVolumes      12230 numFaces      459328 numITransforms           1 numITransforms*numVolumes       12230
    mm i   1 geocode   A            EMPTY numVolumes          1 numFaces           0 numITransforms        1792 numITransforms*numVolumes        1792
    mm i   2 geocode   A                  numVolumes          1 numFaces          12 numITransforms         864 numITransforms*numVolumes         864
    mm i   3 geocode   A                  numVolumes          1 numFaces          12 numITransforms         864 numITransforms*numVolumes         864
    mm i   4 geocode   A                  numVolumes          1 numFaces          12 numITransforms         864 numITransforms*numVolumes         864
    mm i   5 geocode   A                  numVolumes          5 numFaces        2976 numITransforms         672 numITransforms*numVolumes        3360
     num_total_volumes 12230 num_instanced_volumes 7744 num_global_volumes 4486
    2018-07-01 14:41:35.427 INFO  [1929481] [OGeo::makeGeometry@595] OGeo::makeGeometry geocode A
    2018-07-01 14:41:35.427 INFO  [1929481] [GParts::close@865] GParts::close START  verbosity 0
    2018-07-01 14:41:35.487 INFO  [1929481] [GParts::close@881] GParts::close DONE  verbosity 0
    2018-07-01 14:41:35.487 INFO  [1929481] [OGeo::makeAnalyticGeometry@646] OGeo::makeAnalyticGeometry pts:  GParts  primflag         flagnodetree numParts 12496 numPrim 3116
    2018-07-01 14:41:35.487 FATAL [1929481] [OGeo::makeAnalyticGeometry@672]  NodeTree : MISMATCH (numPrim != numVolumes)  numVolumes 12230 numVolumesSelected 3116 numPrim 3116 numPart 12496 numTran 5344 numPlan 672
    2018-07-01 14:41:35.830 WARN  [1929481] [OGeo::convertMergedMesh@230] OGeo::convertMesh skipping mesh 1
    2018-07-01 14:41:35.843 INFO  [1929481] [OGeo::makeTriangulatedGeometry@815] OGeo::makeTriangulatedGeometry  lod 0 mmIndex 2 numFaces (PrimitiveCount) 12 numFaces0 (Outermost) 12 numVolumes 1 numITransforms 864
    2018-07-01 14:41:35.843 FATAL [1929481] [*GMesh::makeFaceRepeatedInstancedIdentityBuffer@2043] GMesh::makeFaceRepeatedInstancedIdentityBuffer nodeinfo_ok 1 nodeinfo_buffer_items 1 numVolumes 1
    2018-07-01 14:41:35.843 FATAL [1929481] [*GMesh::makeFaceRepeatedInstancedIdentityBuffer@2051] GMesh::makeFaceRepeatedInstancedIdentityBuffer iidentity_ok 1 iidentity_buffer_items 864 numFaces (sum of faces in numVolumes)12 numITransforms 864 numVolumes*numITransforms 864 numRepeatedIdentity 10368
    2018-07-01 14:41:35.844 INFO  [1929481] [OGeo::makeTriangulatedGeometry@815] OGeo::makeTriangulatedGeometry  lod 1 mmIndex 2 numFaces (PrimitiveCount) 12 numFaces0 (Outermost) 12 numVolumes 1 numITransforms 864
    Process 67368 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x10)
        frame #0: 0x00000001004f975c libOptiXRap.dylib`optix::GeometryObj::get(this=0x0000000000000000) at optixpp_namespace.h:3533
       3530	
       3531	  inline RTgeometry GeometryObj::get()
       3532	  {
    -> 3533	    return m_geometry;
       3534	  }
       3535	
       3536	  inline void MaterialObj::destroy()
    Target 0: (OKX4Test) stopped.
    (lldb) bt

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x10)
      * frame #0: 0x00000001004f975c libOptiXRap.dylib`optix::GeometryObj::get(this=0x0000000000000000) at optixpp_namespace.h:3533
        frame #1: 0x00000001004f94da libOptiXRap.dylib`optix::GeometryInstanceObj::setGeometry(this=0x0000000131170820, geometry=<unavailable>) at optixpp_namespace.h:3305
        frame #2: 0x00000001004f25cb libOptiXRap.dylib`optix::Handle<optix::GeometryInstanceObj> optix::ContextObj::createGeometryInstance<std::__1::__wrap_iter<optix::Handle<optix::MaterialObj>*> >(this=0x000000011b7ebef0, geometry=optix::Geometry @ 0x00007ffeefbfa7c8, matlbegin=__wrap_iter<optix::Handle<optix::MaterialObj> *> @ 0x00007ffeefbfa6f0, matlend=__wrap_iter<optix::Handle<optix::MaterialObj> *> @ 0x00007ffeefbfa6e8) at optixpp_namespace.h:2227
        frame #3: 0x00000001004ebe55 libOptiXRap.dylib`OGeo::makeGeometryInstance(this=0x000000012f9b51c0, geometry=optix::Geometry @ 0x00007ffeefbfae40, material=<unavailable>) at OGeo.cc:576
        frame #4: 0x00000001004ece70 libOptiXRap.dylib`OGeo::makeRepeatedGroup(this=0x000000012f9b51c0, mm=0x00000001204561c0, raylod=false) at OGeo.cc:335
        frame #5: 0x00000001004ea1a3 libOptiXRap.dylib`OGeo::convertMergedMesh(this=0x000000012f9b51c0, i=2) at OGeo.cc:251
        frame #6: 0x00000001004e9505 libOptiXRap.dylib`OGeo::convert(this=0x000000012f9b51c0) at OGeo.cc:179
        frame #7: 0x00000001004e1d29 libOptiXRap.dylib`OScene::init(this=0x00000001204f3170) at OScene.cc:156
        frame #8: 0x00000001004e0854 libOptiXRap.dylib`OScene::OScene(this=0x00000001204f3170, hub=0x000000011e5cff20) at OScene.cc:78
        frame #9: 0x00000001004e22bd libOptiXRap.dylib`OScene::OScene(this=0x00000001204f3170, hub=0x000000011e5cff20) at OScene.cc:77
        frame #10: 0x0000000100406d7e libOKOP.dylib`OpEngine::OpEngine(this=0x000000012be00380, hub=0x000000011e5cff20) at OpEngine.cc:44
        frame #11: 0x000000010040726d libOKOP.dylib`OpEngine::OpEngine(this=0x000000012be00380, hub=0x000000011e5cff20) at OpEngine.cc:52
        frame #12: 0x000000010010a5f6 libOK.dylib`OKPropagator::OKPropagator(this=0x000000012be00320, hub=0x000000011e5cff20, idx=0x000000011e5d3ea0, viz=0x000000011e5d42e0) at OKPropagator.cc:50
        frame #13: 0x000000010010a75d libOK.dylib`OKPropagator::OKPropagator(this=0x000000012be00320, hub=0x000000011e5cff20, idx=0x000000011e5d3ea0, viz=0x000000011e5d42e0) at OKPropagator.cc:54
        frame #14: 0x0000000100109f10 libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe260, argc=1, argv=0x00007ffeefbfea98, argforced=0x0000000000000000) at OKMgr.cc:50
        frame #15: 0x000000010010a1cb libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe260, argc=1, argv=0x00007ffeefbfea98, argforced=0x0000000000000000) at OKMgr.cc:52
        frame #16: 0x0000000100014988 OKX4Test`main(argc=1, argv=0x00007ffeefbfea98) at OKX4Test.cc:84
        frame #17: 0x00007fff73396015 libdyld.dylib`start + 1
        frame #18: 0x00007fff73396015 libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 6
    frame #6: 0x00000001004e9505 libOptiXRap.dylib`OGeo::convert(this=0x000000012f9b51c0) at OGeo.cc:179
       176 	
       177 	    for(unsigned i=0 ; i < nmm ; i++)
       178 	    {
    -> 179 	        convertMergedMesh(i);
       180 	    }
       181 	
       182 	    // all group and geometry_group need to have distinct acceleration structures
    (lldb) p nmm
    (unsigned int) $0 = 6
    (lldb) p i
    (unsigned int) $1 = 2
    (lldb) 
    (lldb) p mm->m_parts->m_idx_buffer->data()
    (std::__1::vector<unsigned int, std::__1::allocator<unsigned int> >) $7 = size=4 {
      [0] = 0
      [1] = 205
      [2] = 197
      [3] = 0
    }
    (lldb) 


::

    epsilon:0 blyth$ mesh.py 197
    INFO:__main__:Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
    MOInMOFT0xc047100




