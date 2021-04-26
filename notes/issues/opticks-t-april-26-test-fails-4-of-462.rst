opticks-t-april-26-test-fails-4-of-462
=========================================





FIXED : Notice the FAIL reported by Zike is from lack of the NumPy input file 
--------------------------------------------------------------------------------

Reproduce that::

    O[blyth@localhost opticks]$ rm $HOME/.opticks/flight/RoundaboutXY.npy
    O[blyth@localhost opticks]$ gdb OpFlightPathTest 

::

    2021-04-26 22:21:30.674 INFO  [4100] [OGeo::convert@302] [ nmm 10
    2021-04-26 22:21:32.008 INFO  [4100] [OGeo::convert@321] ] nmm 10
    2021-04-26 22:21:32.155 INFO  [4100] [FlightPath::setPathFormat@200]  m_outdir $TMP/flight name frame%0.5d.jpg fmt /tmp/blyth/opticks/flight/frame%0.5d.jpg
    2021-04-26 22:21:32.155 WARN  [4100] [Composition::applyViewType@998] Composition::applyViewType(KEY_U) switching FLIGHTPATH
    2021-04-26 22:21:32.155 INFO  [4100] [FlightPath::load@181]  path $HOME/.opticks/flight/RoundaboutXY.npy
    2021-04-26 22:21:32.155 ERROR [4100] [NPY<T>::load@1011] NPY<T>::load failed for path [/home/blyth/.opticks/flight/RoundaboutXY.npy] use debugload with NPYLoadTest to investigate (problems are usually from dtype mismatches) 
    2021-04-26 22:21:32.156 FATAL [4100] [FlightPath::load@187]  MISSING expected path $HOME/.opticks/flight/RoundaboutXY.npy for flight RoundaboutXY (bad name OR need to run ana/makeflight.sh)
    OpFlightPathTest: /home/blyth/opticks/optickscore/FlightPath.cc:192: void FlightPath::load(): Assertion `m_eluc' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffeddf4387 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-39.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-39.el7.x86_64 openssl-libs-1.0.2k-19.el7.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffeddf4387 in raise () from /lib64/libc.so.6
    #1  0x00007fffeddf5a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffedded1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffedded252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff5273282 in FlightPath::load (this=0x23671110) at /home/blyth/opticks/optickscore/FlightPath.cc:192
    #5  0x00007ffff52737bb in FlightPath::makeInterpolatedView (this=0x23671110) at /home/blyth/opticks/optickscore/FlightPath.cc:239
    #6  0x00007ffff52738b7 in FlightPath::getInterpolatedView (this=0x23671110) at /home/blyth/opticks/optickscore/FlightPath.cc:253
    #7  0x00007ffff53054b2 in Composition::applyViewType (this=0x661a20) at /home/blyth/opticks/optickscore/Composition.cc:1009
    #8  0x00007ffff5303c9e in Composition::setViewType (this=0x661a20, type=View::FLIGHTPATH) at /home/blyth/opticks/optickscore/Composition.cc:636
    #9  0x00007ffff52739aa in FlightPath::render (this=0x23671110, renderer=0x2366f6a0) at /home/blyth/opticks/optickscore/FlightPath.cc:264
    #10 0x00007ffff7b598b7 in OpTracer::render_flightpath (this=0x2366f6a0) at /home/blyth/opticks/okop/OpTracer.cc:184
    #11 0x00007ffff7b589b5 in OpPropagator::render_flightpath (this=0x8a2fd90) at /home/blyth/opticks/okop/OpPropagator.cc:137
    #12 0x00007ffff7b57ad1 in OpMgr::render_flightpath (this=0x7fffffffae00) at /home/blyth/opticks/okop/OpMgr.cc:199
    #13 0x0000000000402b35 in main (argc=1, argv=0x7fffffffb148) at /home/blyth/opticks/okop/tests/OpFlightPathTest.cc:30
    (gdb) 

    (gdb) f 13
    #13 0x0000000000402b35 in main (argc=1, argv=0x7fffffffb148) at /home/blyth/opticks/okop/tests/OpFlightPathTest.cc:30
    30	    op.render_flightpath();
    (gdb) f 12
    #12 0x00007ffff7b57ad1 in OpMgr::render_flightpath (this=0x7fffffffae00) at /home/blyth/opticks/okop/OpMgr.cc:199
    199	    m_propagator->render_flightpath(); 
    (gdb) f 11
    #11 0x00007ffff7b589b5 in OpPropagator::render_flightpath (this=0x8a2fd90) at /home/blyth/opticks/okop/OpPropagator.cc:137
    137	    m_tracer->render_flightpath();
    (gdb) f 10
    #10 0x00007ffff7b598b7 in OpTracer::render_flightpath (this=0x2366f6a0) at /home/blyth/opticks/okop/OpTracer.cc:184
    184	    fp->render( (SRenderer*)this  );  
    (gdb) f 9
    #9  0x00007ffff52739aa in FlightPath::render (this=0x23671110, renderer=0x2366f6a0) at /home/blyth/opticks/optickscore/FlightPath.cc:264
    264	    m_composition->setViewType(View::FLIGHTPATH);
    (gdb) f 8
    #8  0x00007ffff5303c9e in Composition::setViewType (this=0x661a20, type=View::FLIGHTPATH) at /home/blyth/opticks/optickscore/Composition.cc:636
    636	    applyViewType();
    (gdb) f 7
    #7  0x00007ffff53054b2 in Composition::applyViewType (this=0x661a20) at /home/blyth/opticks/optickscore/Composition.cc:1009
    1009	        InterpolatedView* iv = m_flightpath->getInterpolatedView();     
    (gdb) f 6
    #6  0x00007ffff52738b7 in FlightPath::getInterpolatedView (this=0x23671110) at /home/blyth/opticks/optickscore/FlightPath.cc:253
    253	    if(!m_view) m_view = makeInterpolatedView();
    (gdb) f 5
    #5  0x00007ffff52737bb in FlightPath::makeInterpolatedView (this=0x23671110) at /home/blyth/opticks/optickscore/FlightPath.cc:239
    239	    load(); 
    (gdb) f 4
    #4  0x00007ffff5273282 in FlightPath::load (this=0x23671110) at /home/blyth/opticks/optickscore/FlightPath.cc:192
    192	    assert( m_eluc ) ; 
    (gdb) 






AFTER FIXES : down to 1 FAIL, same 3 SLOW : the 1 FAIL from an CUDA/OptiX launch fail : Illegal Address
------------------------------------------------------------------------------------------------------------


All the SLOW ones are using Geant4 : hence as are using the full JUNO tds geometry 
the slowness is not surprising, coming from Geant4 voxelisation.

::

    SLOW: tests taking longer that 15 seconds
      8  /39  Test #8  : CFG4Test.CG4Test                              Passed                         124.48 
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         145.47 
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      15.62  


    FAILS:  1   / 462   :  Mon Apr 26 21:52:13 2021   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      15.62  
    O[blyth@localhost opticks]$ 





Issue 4/462 FAILS on Precision, CentOS7 Linux
-------------------------------------------------

With tds (lastest JUNO) geocache selected via OPTICKS_KEY::

    SLOW: tests taking longer that 15 seconds
      8  /39  Test #8  : CFG4Test.CG4Test                              Passed                         125.92 
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         146.25 
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      15.77  


    FAILS:  4   / 462   :  Mon Apr 26 20:03:26 2021   
      7  /117 Test #7  : NPYTest.NFlightConfigTest                     Child aborted***Exception:     0.09   
      44 /44  Test #44 : OpticksCoreTest.FlightPathTest                Child aborted***Exception:     0.08   
      6  /6   Test #6  : OKOPTest.OpFlightPathTest                     Child aborted***Exception:     9.97   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      15.77  
    O[blyth@localhost opticks]$ 




FIXED : NFlightConfigTest stale config
-----------------------------------------

::

    (lldb) r
    Process 8271 launched: '/usr/local/opticks/lib/NFlightConfigTest' (x86_64)
    2021-04-26 14:28:26.307 FATAL [32325992] [BConfig::parse@110]  UNKNOWN/DUPLICATE KEY prefix : frame_ found 0 in config prefix=frame_,ext=.ppm,scale0=1,scale1=10
    Assertion failed: (found == 1), function parse, file /Users/blyth/opticks/boostrap/BConfig.cc, line 116.
    (lldb) bt
        frame #3: 0x00007fff535681ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001007d96e1 libBoostRap.dylib`BConfig::parse(this=0x0000000101800ac0) at BConfig.cc:116
        frame #5: 0x0000000100428fbd libNPY.dylib`NFlightConfig::NFlightConfig(this=0x00007ffeefbfcf50, cfg="prefix=frame_,ext=.ppm,scale0=1,scale1=10") at NFlightConfig.cpp:56
        frame #6: 0x000000010042900d libNPY.dylib`NFlightConfig::NFlightConfig(this=0x00007ffeefbfcf50, cfg="prefix=frame_,ext=.ppm,scale0=1,scale1=10") at NFlightConfig.cpp:44
        frame #7: 0x0000000100004b81 NFlightConfigTest`main(argc=1, argv=0x00007ffeefbfd0b8) at NFlightConfigTest.cc:39
        frame #8: 0x00007fff534f4015 libdyld.dylib`start + 1
        frame #9: 0x00007fff534f4015 libdyld.dylib`start + 1
    (lldb) 


FIXED : OpFlightPathTest unchecked --targetpvn option for annotation
-----------------------------------------------------------------------

::

    Target 0: (OpFlightPathTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x0)
      * frame #0: 0x00007fff53544232 libsystem_c.dylib`strlen + 18
        frame #1: 0x000000010305bab5 libOpticksGeo.dylib`std::__1::char_traits<char>::length(__s=0x0000000000000000) at __string:215
        frame #2: 0x0000000103064e7c libOpticksGeo.dylib`std::__1::basic_ostream<char, std::__1::char_traits<char> >& std::__1::operator<<<std::__1::char_traits<char> >(__os=0x00007ffeefbfb6a0, __str=0x0000000000000000) at ostream:866
        frame #3: 0x000000010370a3b8 libOpticksCore.dylib`Opticks::getFrameAnnotation(this=0x00007ffeefbfcee0, frame=0, num_frame=3, dt=12.387667000002693) const at Opticks.cc:2517
        frame #4: 0x000000010366c48e libOpticksCore.dylib`FlightPath::render(this=0x0000000130cb4360, renderer=0x000000012fefc030) at FlightPath.cc:300
        frame #5: 0x00000001000ddf02 libOKOP.dylib`OpTracer::render_flightpath(this=0x000000012fefc030) at OpTracer.cc:184
        frame #6: 0x00000001000dcb45 libOKOP.dylib`OpPropagator::render_flightpath(this=0x0000000116d8fcf0) at OpPropagator.cc:137
        frame #7: 0x00000001000dbaa5 libOKOP.dylib`OpMgr::render_flightpath(this=0x00007ffeefbfce80) at OpMgr.cc:199
        frame #8: 0x0000000100006ad3 OpFlightPathTest`main(argc=1, argv=0x00007ffeefbfd110) at OpFlightPathTest.cc:30
        frame #9: 0x00007fff534f4015 libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 4
    frame #4: 0x000000010366c48e libOpticksCore.dylib`FlightPath::render(this=0x0000000130cb4360, renderer=0x000000012fefc030) at FlightPath.cc:300
       297 	
       298 	        double dt = renderer->render();   // calling OTracer::trace_
       299 	        
    -> 300 	        std::string bottom_annotation = m_ok->getFrameAnnotation(i, imax, dt ); 
       301 	
       302 	        fp->fillPathFormat(path, 128, i ); 
       303 	
    (lldb) p dt
    (double) $0 = 12.387667000002693
    (lldb) p imax
    (int) $1 = 3
    (lldb) p i 
    (int) $2 = 0
    (lldb) f 3
    frame #3: 0x000000010370a3b8 libOpticksCore.dylib`Opticks::getFrameAnnotation(this=0x00007ffeefbfcee0, frame=0, num_frame=3, dt=12.387667000002693) const at Opticks.cc:2517
       2514	        << std::setw(5) << frame << "/" << num_frame
       2515	        << " dt " << std::setw(10) << std::fixed << std::setprecision(4) << dt  
       2516	        << " | "
    -> 2517	        << " --targetpvn " << targetpvn 
       2518	        << " -e " <<  emm
       2519	        ;   
       2520	    std::string s = ss.str(); 
    (lldb) p targetpvn
    (const char *) $3 = 0x0000000000000000
    (lldb) 


FIXED : FlightPathTest handling NULL cfg 
--------------------------------------------

::

    (lldb) f 14
    frame #14: 0x0000000100118da1 libOpticksCore.dylib`FlightPath::save(this=0x0000000101501230) const at FlightPath.cc:112
       109 	    nlohmann::json& js = m_meta->js ; 
       110 	
       111 	    js["argline"] = m_ok->getArgLine(); 
    -> 112 	    js["cfg"] = m_cfg->getCfg(); 
       113 	    js["nameprefix"] = m_nameprefix ;  
       114 	    js["scale"] = m_scale ;  
       115 	    js["emm"] = m_ok->getEnabledMergedMesh() ;  
    (lldb) p m_cfg->bconfig
    (BConfig *) $0 = 0x00000001015012d0
    (lldb) p *(m_cfg->bconfig)
    (BConfig) $1 = {
      cfg = 0x0000000000000000 <no value available>
      edelim = ','
      kvdelim = 0x0000000100bb1d02 "="
      ekv = size=0 {}
      eki = size=3 {




tboolean_box.sh : mysterious failed OPropagator::launch : Illegal address 
----------------------------------------------------------------------------------------

* tboolean are special in that that change geometry on top of a basis geometry
* potentially the current geometry is missing something needed for that 


::


    O[blyth@localhost tests]$ pwd
    /home/blyth/opticks/integration/tests
    O[blyth@localhost tests]$ ./tboolean_box.sh 

    ...

    2021-04-26 21:34:24.280 INFO  [377892] [OGeo::convert@302] [ nmm 10
    2021-04-26 21:34:25.612 INFO  [377892] [OGeo::convert@321] ] nmm 10
    2021-04-26 21:34:25.688 ERROR [377892] [cuRANDWrapper::setItems@154] CAUTION : are resizing the launch sequence 
    2021-04-26 21:34:26.562 FATAL [377892] [ORng::setSkipAhead@160]  skip as as WITH_SKIPAHEAD not enabled 
    2021-04-26 21:34:26.638 INFO  [377892] [OpticksRun::createEvent@115]  tagoffset 0 skipaheadstep 0 skipahead 0
    2021-04-26 21:34:26.664 INFO  [377892] [OpEngine::close@168]  sensorlib NULL : defaulting it with zero sensors 
    2021-04-26 21:34:26.664 ERROR [377892] [SensorLib::close@374]  SKIP as m_sensor_num zero 
    2021-04-26 21:34:26.664 FATAL [377892] [OCtx::create_buffer@300] skip upload_buffer as num_bytes zero key:OSensorLib_sensor_data
    2021-04-26 21:34:26.664 FATAL [377892] [OCtx::create_buffer@300] skip upload_buffer as num_bytes zero key:OSensorLib_texid
    2021-04-26 21:34:26.665 INFO  [377892] [OEvent::markDirty@300] OEvent::markDirty(source) PROCEED
    2021-04-26 21:34:29.365 INFO  [377892] [OPropagator::prelaunch@202] 0 : (0;0,0) 
    OPropagator::prelaunch
                  validate000                 0.055106
                   compile000                    7e-06
                 prelaunch000                  2.59227

    2021-04-26 21:34:29.365 FATAL [377892] [OPropagator::launch@272]  skipahead 0
    2021-04-26 21:34:29.365 FATAL [377892] [ORng::setSkipAhead@160]  skip as as WITH_SKIPAHEAD not enabled 
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)

    Program received signal SIGABRT, Aborted.
    0x00007fffe5772387 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-57.el7.x86_64 libgcc-4.8.5-39.el7.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libstdc++-4.8.5-39.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-21.el7_6.x86_64 openssl-libs-1.0.2k-19.el7.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe5772387 in raise () from /lib64/libc.so.6
    #1  0x00007fffe5773a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffe60827d5 in __gnu_cxx::__verbose_terminate_handler() () from /lib64/libstdc++.so.6
    #3  0x00007fffe6080746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00007fffe6080773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00007fffe6080993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x00007ffff637189b in optix::ContextObj::checkError (this=0xa0f17d0, code=RT_ERROR_UNKNOWN) at /home/blyth/local/opticks/externals/OptiX_650/include/optixu/optixpp_namespace.h:2219
    #7  0x00007ffff63b048e in optix::ContextObj::launch (this=0xa0f17d0, entry_point_index=0, image_width=10000, image_height=1)
        at /home/blyth/local/opticks/externals/OptiX_650/include/optixu/optixpp_namespace.h:3006
    #8  0x00007ffff63ae208 in OContext::launch_ (this=0xa1deb60, entry=0, width=10000, height=1) at /home/blyth/opticks/optixrap/OContext.cc:893
    #9  0x00007ffff63adef9 in OContext::launch (this=0xa1deb60, lmode=16, entry=0, width=10000, height=1, times=0x24104570) at /home/blyth/opticks/optixrap/OContext.cc:853
    #10 0x00007ffff63c4831 in OPropagator::launch (this=0x8a92f40) at /home/blyth/opticks/optixrap/OPropagator.cc:279
    #11 0x00007ffff673310d in OpEngine::propagate (this=0x9e51e80) at /home/blyth/opticks/okop/OpEngine.cc:213
    #12 0x00007ffff79738b8 in OKPropagator::propagate (this=0x9e51cc0) at /home/blyth/opticks/ok/OKPropagator.cc:111
    #13 0x00007ffff7bafcd4 in OKG4Mgr::propagate_ (this=0x7fffffff4580) at /home/blyth/opticks/okg4/OKG4Mgr.cc:217
    #14 0x00007ffff7bafb8d in OKG4Mgr::propagate (this=0x7fffffff4580) at /home/blyth/opticks/okg4/OKG4Mgr.cc:157
    #15 0x00000000004038c9 in main (argc=33, argv=0x7fffffff48c8) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:28
    (gdb) 


