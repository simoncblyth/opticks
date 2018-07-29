OKX4Test_analytic_raytrace
============================

Summary
---------

* even after getting the analytic buffers near identical 
  the new direct route ray trace remains fragile, with 
  some lv (notably lvIdx:56 RadialShieldUnit0xc3d7da8 )
  particularly prone to causing crashes 

  * much more fragile than the old geocache gltf route 
  * initially thought this may be GPU resource problem, 
    but the success of the old route suggests that that 
    is not the full story 


July 29 
----------

::

    OKX4Test


    2018-07-29 08:59:42.249 INFO  [5044504] [Interactor::key_pressed@408] Interactor::key_pressed O nextRenderStyle 
    2018-07-29 08:59:42.388 INFO  [5044504] [OTracer::trace_@128] OTracer::trace  entry_index 0 trace_count 0 resolution_scale 1 size(2880,1704) ZProj.zw (-1.04459,-2229.5) front 0.8756,0.0629,-0.4789
    2018-07-29 08:59:42.388 INFO  [5044504] [OContext::close@241] OContext::close numEntryPoint 1
    2018-07-29 08:59:42.399 INFO  [5044504] [OContext::close@245] OContext::close setEntryPointCount done.
    2018-07-29 08:59:42.423 INFO  [5044504] [OContext::close@251] OContext::close m_cfg->apply() done.
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)
    Abort trap: 6
    epsilon:okg4 blyth$ 



Ideas to try
----------------

1. launch with reduced resolution 
2. get opticks-nnt working again, extend it by generating 
   some G4 C++ that creates each lvIdx, and use that to
   implement 248 single lv visualizations   

   * are pursuing this approach over in X4 


3. non-graphical ray tracing (snaps) of geometry 
4. keep plogging away at matching the new direct workflow buffers with the old ones

   * surfaces next 

5. have neglected the triangles in the new route, suspect the lack of deduping so far
   may mean are using more GPU resources than necessary, causing fragility 

   * idea : keep count of all GPU buffer allocations (OpenGL tri, OptiX tri, OptiX ana)
     in an NGPU class recording (numBytes, name, owner) : compare totals between branches
     investigate outliers 

6. simpler testing environment

   * remember the very messy multi Opticks/OptickGeo and Hub environment
     to fixup problems with the GDML : need to find way to test in a cleaner environment

7. i have not checked the planeBuffer between old and new, that could easily 
    cause crashes for geometry with convexpolyhedrons

   * nothing smoking at a glance :doc:`OKX4Test_planBuffer_mm0`




Added OpticksQuery lvr selecting on lvIdx 
-----------------------------------------------

::

    export OPTICKS_QUERY_LIVE="lvr:47:48" ; lldb OKX4Test 
           ## soft crashes for lack of any global geometry

    export OPTICKS_QUERY_LIVE="lvr:0:1,lvr:47:48" ; lldb OKX4Test 
           ## pool cover and PMTs 

    export OPTICKS_QUERY_LIVE="lvr:0:1,lvr:47:48,lvr:248:249" ; lldb OKX4Test 
           ## including world box 248 : not very useful as far too big 

    export OPTICKS_QUERY_LIVE="lvr:0:1,lvr:47:48,lvr:56:57" ; lldb OKX4Test 
           ##  BINGO : including radial shield unit lv:56 works in OGLRender
           ##          but hard crashes in raytrace, causing system panic, reboot   


    export OPTICKS_QUERY_LIVE="lvr:0:56" ; lldb OKX4Test 
           ## raytrace works : see the building and the 2 ADs  

    export OPTICKS_QUERY_LIVE="lvr:0:56,lvr:57:67" ; lldb OKX4Test 
           ## crashes 

    export OPTICKS_QUERY_LIVE="lvr:236:248" ; lldb OKX4Test 
           ## ok 

    export OPTICKS_QUERY_LIVE="lvr:200:248" ; lldb OKX4Test 
           ## ok 

    export OPTICKS_QUERY_LIVE="lvr:150:248" ; lldb OKX4Test 
           ## ok 

    export OPTICKS_QUERY_LIVE="lvr:100:248" ; lldb OKX4Test 
           ## ok 

    export OPTICKS_QUERY_LIVE="lvr:0:56,lvr:100:248" ; lldb OKX4Test 
           ## ok 

    export OPTICKS_QUERY_LIVE="lvr:0:56,lvr:80:248" ; lldb OKX4Test 
           ## ok 
           
    export OPTICKS_QUERY_LIVE="lvr:0:56,lvr:70:248" ; lldb OKX4Test 
           ## ok 

    export OPTICKS_QUERY_LIVE="lvr:0:56,lvr:60:248" ; lldb OKX4Test 
           ## ok 
            
    export OPTICKS_QUERY_LIVE="lvr:0:56,lvr:58:248" ; lldb OKX4Test 
           ## ok 

    export OPTICKS_QUERY_LIVE="lvr:0:56,lvr:57:248" ; lldb OKX4Test  
           ## ok 

    export OPTICKS_QUERY_LIVE="lvr:0:56,lvr:56:248" ; lldb OKX4Test  
           ## ok : huh, it works this time : twas not a closup, perhaps depends on position 

    export OPTICKS_QUERY_LIVE="lvr:0:56,lvr:56:248" ; lldb OKX4Test 
           ## this time navigate into closer position (bookmark 2), then switch on raytrace : get the crash  

    export OPTICKS_QUERY_LIVE="lvr:0:56,lvr:58:248" ; OKX4Test --stack 4360
           ## again crash from bookmark 2 

    export OPTICKS_QUERY_LIVE="range:3153:12221" ; lldb OKX4Test 
           ## raytrace crash 

    export OPTICKS_QUERY_LIVE="range:3153:12221" ; lldb OKX4Test -- --stack 3180
           ## raytrace crash

    export OPTICKS_QUERY_LIVE="lvr:0:1,lvr:57:58" ; OKX4Test --stack 4360 
           ## 

    export OPTICKS_QUERY_LIVE="lvr:0:1,lvr:57:58" ; OKX4Test --stack 4360 
           ## works : pool cover and two top? reflector

    export OPTICKS_QUERY_LIVE="lvr:0:1,lvr:56:57" ; OKX4Test --stack 4360 
           ## crash : pool cover and two radial shield units

    export OPTICKS_QUERY="lvr:0:1,lvr:56:57" ; OTracerTest --gltf 3 
    OPTICKS_RESOURCE_LAYOUT=103 OTracerTest --gltf 3 
            black renders


     OPTICKS_RESOURCE_LAYOUT=103 OTracerTest --gltf 3 
           actually the starting point and near/far are way out, need 
           to use bookmarks to see something 
           raytrace works 




::

    In [18]: for k,v in ma.idx2name.items(): print "%3d : %s " % (k,v )
      0 : near_top_cover_box0xc23f970 
      1 : RPCStrip0xc04bcb0 
     ..
     54 : headon-pmt-assy0xbf55198 
     55 : headon-pmt-mount0xc2a7670 

     56 : RadialShieldUnit0xc3d7da8 

     57 : TopESRCutHols0xbf9de10 
     58 : TopRefGapCutHols0xbf9cef8 
     59 : TopRefCutHols0xbf9bd50 
     60 : BotESRCutHols0xbfa7368 
     61 : BotRefGapCutHols0xc34bb28 
     62 : BotRefHols0xc3cd380 
     63 : SstBotRib0xc26c4c0 




hmm : select on CSG tree height ?
------------------------------------

Hmm attempt gives black render.  Need to test per lv.  H

::

    export OPTICKS_QUERY_LIVE="lvr:0:3" ; lldb OKX4Test 



lvr:0:56,lvr:57:67 crashes too
---------------------------------

::

    export OPTICKS_QUERY_LIVE="lvr:0:56,lvr:57:67" ; lldb OKX4Test 

    2018-07-03 16:45:17.364 INFO  [619762] [OTracer::trace_@128] OTracer::trace  entry_index 0 trace_count 0 resolution_scale 1 size(2880,1704) ZProj.zw (-1.13622,-6811.12) front 0.8437,0.5368,0.0000
    2018-07-03 16:45:17.365 INFO  [619762] [OContext::close@236] OContext::close numEntryPoint 1
    2018-07-03 16:45:17.370 INFO  [619762] [OContext::close@240] OContext::close setEntryPointCount done.
    2018-07-03 16:45:17.394 INFO  [619762] [OContext::close@246] OContext::close m_cfg->apply() done.
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (719): Launch failed)
    Process 70365 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff7aacbb6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff7aacbb6e <+10>: jae    0x7fff7aacbb78            ; <+20>
        0x7fff7aacbb70 <+12>: movq   %rax, %rdi
        0x7fff7aacbb73 <+15>: jmp    0x7fff7aac2b00            ; cerror_nocancel
        0x7fff7aacbb78 <+20>: retq   
    Target 0: (OKX4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff7aacbb6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff7ac96080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff7aa271ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff7892bf8f libc++abi.dylib`abort_message + 245
        frame #4: 0x00007fff7892c113 libc++abi.dylib`default_terminate_handler() + 241
        frame #5: 0x00007fff79d63eab libobjc.A.dylib`_objc_terminate() + 105
        frame #6: 0x00007fff789477c9 libc++abi.dylib`std::__terminate(void (*)()) + 8
        frame #7: 0x00007fff7894726f libc++abi.dylib`__cxa_throw + 121
        frame #8: 0x00000001004b9ce6 libOptiXRap.dylib`optix::ContextObj::checkError(this=0x000000011b46dad0, code=RT_ERROR_UNKNOWN) const at optixpp_namespace.h:1963
        frame #9: 0x00000001004ce5e0 libOptiXRap.dylib`optix::ContextObj::launch(this=0x000000011b46dad0, entry_point_index=0, image_width=2880, image_height=1704) at optixpp_namespace.h:2536
        frame #10: 0x00000001004ce453 libOptiXRap.dylib`OContext::launch_(this=0x000000012c46c6c0, entry=0, width=2880, height=1704) at OContext.cc:330
        frame #11: 0x00000001004cdf46 libOptiXRap.dylib`OContext::launch(this=0x000000012c46c6c0, lmode=30, entry=0, width=2880, height=1704, times=0x000000011e1ac370) at OContext.cc:289
        frame #12: 0x00000001004e07d7 libOptiXRap.dylib`OTracer::trace_(this=0x000000012d4ec460) at OTracer.cc:142
        frame #13: 0x0000000100131925 libOpticksGL.dylib`OKGLTracer::render(this=0x000000012d4e7380) at OKGLTracer.cc:165
        frame #14: 0x00000001001c7001 libOGLRap.dylib`OpticksViz::render(this=0x000000011cb862c0) at OpticksViz.cc:432
        frame #15: 0x00000001001c5c12 libOGLRap.dylib`OpticksViz::renderLoop(this=0x000000011cb862c0) at OpticksViz.cc:474
        frame #16: 0x00000001001c5352 libOGLRap.dylib`OpticksViz::visualize(this=0x000000011cb862c0) at OpticksViz.cc:135
        frame #17: 0x000000010010a4ed libOK.dylib`OKMgr::visualize(this=0x00007ffeefbfe438) at OKMgr.cc:121
        frame #18: 0x0000000100014c1b OKX4Test`main(argc=1, argv=0x00007ffeefbfea68) at OKX4Test.cc:99
        frame #19: 0x00007fff7a97b015 libdyld.dylib`start + 1
        frame #20: 0x00007fff7a97b015 libdyld.dylib`start + 1
    (lldb) 




Still get launch crash : even now that prim/part/tran are very close to perfect matches ?
---------------------------------------------------------------------------------------------

::


    2018-07-03 16:37:31.132 INFO  [614164] [Interactor::key_pressed@409] Interactor::key_pressed O nextRenderStyle 
    2018-07-03 16:37:31.249 INFO  [614164] [OTracer::trace_@128] OTracer::trace  entry_index 0 trace_count 0 resolution_scale 1 size(2880,1704) ZProj.zw (-1.04459,-2229.5) front 0.9371,0.3491,0.0000
    2018-07-03 16:37:31.250 INFO  [614164] [OContext::close@236] OContext::close numEntryPoint 1
    2018-07-03 16:37:31.260 INFO  [614164] [OContext::close@240] OContext::close setEntryPointCount done.
    2018-07-03 16:37:31.285 INFO  [614164] [OContext::close@246] OContext::close m_cfg->apply() done.
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)
    Abort trap: 6
    epsilon:analytic blyth$ 




lvIdx 56 
----------

::

    56 : RadialShieldUnit0xc3d7da8 


This one caused problems before, slab-segment intersects : tree balancing 
putting two slabs together.

* :doc:`vidx56_RadialShieldUnit0xc3d7da8`



NTreeProcess stats
--------------------

::

    NTreeProcess

    60     if(ProcBuffer) ProcBuffer->add(soIdx, lvIdx, height0, height1);

    In [2]: prb = np.load(os.path.expandvars("$TMP/ProcBuffer.npy"))


       [ 64,  50,   0,   0],
       [ 65,  53,   0,   0],
       [ 66,  55,   2,   2],
       [ 67,  56,   8,   4],    <--- radial shield unit, height of 4 not too terrible ?
       [ 68,  59,   5,   3],
       [ 69,  58,   5,   3],
       [ 70,  57,   9,   4],



    In [3]: prb
    Out[3]: 
    array([[  0, 248,   0,   0],
           [  1, 247,   1,   1],
           [  2,  21,   1,   1],
           [  3,   0,   4,   4],
           [  4,   7,   0,   0],
           [  5,   6,   0,   0],
           [  6,   3,   0,   0],
           [  7,   2,   0,   0],
           [  8,   1,   0,   0],
           [  9,   5,   0,   0],
           [ 10,   4,   0,   0],
           [ 11,   8,   0,   0],
           [ 12,  20,   0,   0],
           [ 13,  16,   0,   0],
           [ 14,   9,   2,   2],
           [ 15,  10,   2,   2],
           [ 16,  11,   1,   1],
           [ 17,  12,   1,   1],
           [ 18,  13,   1,   1],
           [ 19,  14,   0,   0],
           [ 20,  15,   0,   0],





Meaning of the indices corresponding to the source IDPATH, not the created one ?::

    epsilon:extg4 blyth$ mesh.py 0 47 248 
    INFO:__main__:Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 
      0 : near_top_cover_box0xc23f970 
     47 : pmt-hemi0xc0fed90 
    248 : WorldBox0xc15cf40 
    epsilon:extg4 blyth$ 



Try full with some selection
------------------------------

Direct raytrace working for restricted selections.


::

    export OPTICKS_QUERY_LIVE="range:3153:12221"  # this is the default from OpticksResource::DEFAULT_QUERY_LIVE

    export OPTICKS_QUERY_LIVE="range:3153:3154" ; lldb OKX4Test   ## surprised to get a cylinder 

    export OPTICKS_QUERY_LIVE="range:3201:3202,range:3153:3154" ; lldb OKX4Test 




        ## shows 

::

    392 op-geometry-query-dyb()
    393 {
    394     case $1 in
    395    DYB|DLIN)  echo "range:3153:12221"  ;;
    396        DFAR)  echo "range:4686:18894"   ;;  #  
    397        IDYB)  echo "range:3158:3160" ;;  # 2 volumes : pvIAV and pvGDS
    398        JDYB)  echo "range:3158:3159" ;;  # 1 volume : pvIAV
    399        KDYB)  echo "range:3159:3160" ;;  # 1 volume : pvGDS
    400        LDYB)  echo "range:3156:3157" ;;  # 1 volume : pvOAV
    401        MDYB)  echo "range:3201:3202,range:3153:3154"  ;;  # 2 volumes : all the pmt-hemi-cathode instances and ADE  
    402        DSST2)  echo "range:3155:3156,range:4440:4448" ;;    # large BBox discrep
    403        DRV3153) echo "index:3153,depth:13" ;;
    404        DRV3155) echo "index:3155,depth:20" ;;
    405        DLV17)  echo "range:3155:3156,range:2436:2437" ;;    # huh just see the cylinder
    406        DLV30)  echo "range:3155:3156,range:3167:3168" ;;    #
    407        DLV46)  echo "range:3155:3156,range:3200:3201" ;;    #
    408        DLV55)  echo "range:3155:3156,range:4357:4358" ;;    #
    409        DLV56)  echo "range:3155:3156,range:4393:4394" ;;    #
    410        DLV65)  echo "range:3155:3156,range:4440:4441" ;;
    411        DLV66)  echo "range:3155:3156,range:4448:4449" ;;
    412        DLV67)  echo "range:3155:3156,range:4456:4457" ;;
    413        DLV68)  echo "range:3155:3156,range:4464:4465" ;;    # 
    414       DLV103)  echo "range:3155:3156,range:4543:4544" ;;    #
    415       DLV140)  echo "range:3155:3156,range:4606:4607" ;;    #
    416       DLV185)  echo "range:3155:3156,range:4799:4800" ;;    #
    417     esac




Succeed to get a simple sphere thru the machinery
-----------------------------------------------------

Required to set the query envvar and change
code to skip OScintillatorLib when no scintillators.

::

   OPTICKS_QUERY_LIVE="range:0:1" OKX4Test 

   lldb OKX4Test  
   (lldb) env OPTICKS_QUERY_LIVE="range:0:1"
   (lldb) r 

   export OPTICKS_QUERY_LIVE="range:0:1"    ## simpler to just set in invoking environment
   lldb OKX4Test  
 


Hmm how to debug
------------------

There is some issue with the directly converted analytic geometry. 
How to find what ?

1. Some GGeoTest equivalent ?

   * GGeoTest is based on python CSG which becomes a nnode tree ... which is working, 
     unclear how to make an equivalent

2. Create some simple Geant4 geometry instead of the GDML one, and 
   see if can analytic ray trace it 

3. Play around with full geometry but changing the query to pull out bits of 
   geometry   

xanalytic switch
-------------------

Actually because of the two Opticks instances, its
cleaner just to change the argforced of the 2nd Opticks
inside the test, rather than using cmdline.

1. to assist with getting the G4VPhysicalVolume with GDML fixups
2. to check the the conversion to GGeo 



So use no args::

   epsilon:issues blyth$ lldb OKX4Test 
    

Rather than providing args that go to both Opticks::

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




First try changing GMesh default : subsequently added --xanalytic
----------------------------------------------------------------------

Try switching in analytic raytrace by changing 
GMesh default m_geocode to 'A' (rather than 'T').



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




