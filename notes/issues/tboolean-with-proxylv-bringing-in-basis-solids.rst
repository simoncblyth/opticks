tboolean-with-proxylv-bringing-in-basis-solids
=================================================

Context
----------

Following :doc:`tboolean-resurrection` added capability 
for python opticks.analytic.csg:CSG to use a *proxylv=lvIdx* argument 
causing the corresponding standard solid to be included from the 
basis GMeshLib (whica also houses the analytic NCSG).

From tboolean-box--::

 753 box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2", proxylv=${PROXYLV:--1} )
 754 
 755 CSG.Serialize([container, box], args )


Doing this required adding GMesh+NCSG to GMeshLib persisting as described 
in :doc:`review-test-geometry` and handling the proxying in GGeoTest and GMaker.


Observations
---------------

* having to double run the compute and viz is a pain when proxying 
* the black time-zero moving into position glitch is distracting 


::

   PROXYLV=0 tboolean.sh proxy --cvd 1 

      # large cylinder poking out the box,  
      # container auto-resizing not working ? NOW FIXED

   PROXYLV=1 tboolean.sh proxy
   PROXYLV=1 tbooleanviz.sh proxy

      # cylinder with a hole
      # photons only in ring ? but black time zero viz glitch makes uncertain 
      # setting start time to zero rather than 0.2 avoids the glitch

   PROXYLV=4 tboolean.sh proxy --cvd 1 
   PROXYLV=5 tboolean.sh proxy --cvd 1 
      # thin beams, the black before time issue particularly clear 

   PROXYLV=6 tboolean.sh proxy --cvd 1 
      # thin plate 

   PROXYLV=10 tboolean.sh proxy --cvd 1 
      # thick plate with cyclindrical hole part way thru, CSG coincidence speckles apparent
     
   PROXYLV=11 tboolean.sh proxy --cvd 1 
   PROXYLV=12 tboolean.sh proxy --cvd 1
      # squat box   

   PROXYLV=13 tboolean.sh proxy --cvd 1 
      # sphere with small cylinder protrusion on top  

   PROXYLV=14 tboolean.sh proxy --cvd 1 
      # sphere 

   PROXYLV=15 tboolean.sh proxy --cvd 1 
     # vertical cylinder with hole
     # more normal photon behaviour with this smaller piece  
     # perhaps issue is get absorption before long with large geometry ? but surely vacuum ?


   PROXYLV=16 tboolean.sh proxy --cvd 1 

    2019-06-09 23:16:02.989 FATAL [310360] [CTestDetector::makeChildVolume@135]  csg.spec Rock///Rock boundary 2 mother - lv UNIVERSE_LV pv UNIVERSE_PV mat Rock
    2019-06-09 23:16:02.989 INFO  [310360] [CTestDetector::makeDetector_NCSG@199]    0 spec Rock//perfectAbsorbSurface/Vacuum
    2019-06-09 23:16:02.989 FATAL [310360] [CTestDetector::makeChildVolume@135]  csg.spec Rock//perfectAbsorbSurface/Vacuum boundary 0 mother UNIVERSE_LV lv box_lv0_ pv box_pv0_ mat Vacuum
    2019-06-09 23:16:02.989 INFO  [310360] [CTestDetector::makeDetector_NCSG@199]    1 spec Vacuum///GlassSchottF2
    2019-06-09 23:16:02.990 FATAL [310360] [CMaker::MakeSolid_r@142]  unexpected non-identity left transform  depth 3 name un label un
    1.0000,0.0000,0.0000,0.0000 0.0000,1.0000,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 0.0000,125.0000,-70.0000,1.0000
    OKG4Test: /home/blyth/opticks/cfg4/CMaker.cc:150: static G4VSolid* CMaker::MakeSolid_r(const nnode*, unsigned int): Assertion `0' failed.


   PROXYLV=17 tboolean.sh proxy --cvd 1 
      # observatory dome, nice propagation

   PROXYLV=18 tboolean.sh proxy --cvd 1 
      # cathode cap, nice propagation

   PROXYLV=19 tboolean.sh proxy --cvd 1 
      # remainder with cap cut, nice propagation, but non-physical pre-emption evident


   PROXYLV=20 tboolean.sh proxy --cvd 1 

      # 20-inch PMT shape
      # changing sheetmask from 0x1 to 0x2 to make +Z emissive rather that -Z not working 
      #
      # similar pre-emption issue to 18 


    PROXYLV=22 tboolean.sh proxy --cvd 1 

    2019-06-09 23:29:06.619 INFO  [333016] [CTestDetector::makeDetector_NCSG@199]    1 spec Vacuum///GlassSchottF2
    OKG4Test: /home/blyth/opticks/cfg4/CMaker.cc:417: static G4VSolid* CMaker::ConvertPrimitive(const nnode*): Assertion `z2 > z1 && z2 == -z1' failed.
    /home/blyth/opticks/bin/o.sh: line 179: 333016 Aborted                 (core dumped) /home/blyth/local/opticks/lib/OKG4Test --okg4 --cvd 1 --envkey --rendermode +global,+axis --animtimemax 20 --timemax 20 --geocenter --stack 2180 --eye 1,0,0 --test --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-proxy-22_outerfirst=1_analytic=1_csgpath=tboolean-proxy-22_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0_autocontainer=Rock//perfectAbsorbSurface/Vacuum --torch --torchconfig type=disc_




try to viz and propagate together fails : the old linux chestnut
-----------------------------------------------------------------------

Hmm, maybe not the same old problem : commenting out downloading hits makes it work.

::

     PROXYLV=3 tboolean.sh proxy --cvd 1 --dbgdownload 



::

    ## temporaily remove --compute in tboolean.sh to tickle this problem

    PROXYLV=2 tboolean.sh proxy --cvd 1 --dbgdownload -D


    2019-06-09 22:28:19.770 INFO  [225075] [OpEngine::propagate@129] ) propagator.launch 
    2019-06-09 22:28:19.770 INFO  [225075] [OpIndexer::indexSequenceInterop@254] OpIndexer::indexSequenceInterop slicing (OBufBase*)m_seq 
    2019-06-09 22:28:19.780 INFO  [225075] [OpEngine::propagate@132] ]
    2019-06-09 22:28:19.780 INFO  [225075] [OpticksViz::indexPresentationPrep@394] OpticksViz::indexPresentationPrep
    2019-06-09 22:28:19.781 INFO  [225075] [OpticksViz::downloadEvent@384] OpticksViz::downloadEvent (1)
    2019-06-09 22:28:19.787 INFO  [225075] [Rdr::download@74] Rdr::download SKIP for sequence as OPTIX_NON_INTEROP
    2019-06-09 22:28:19.787 INFO  [225075] [OpticksViz::downloadEvent@386] OpticksViz::downloadEvent (1) DONE 
    2019-06-09 22:28:19.787 INFO  [225075] [OpEngine::downloadEvent@149] .
    2019-06-09 22:28:19.787 INFO  [225075] [OContext::download@693] OContext::download PROCEED for sequence as OPTIX_NON_INTEROP
    terminate called after throwing an instance of 'optix::Exception'
      what():  Invalid value (Details: Function "RTresult _rtBufferGetDevicePointer(RTbuffer, int, void**)" caught exception: Cannot get device pointers from non-CUDA interop buffers.)
    
    Program received signal SIGABRT, Aborted.
    0x00007fffe2023207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libX11-devel-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libdrm-2.4.91-3.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe2023207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe20248f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe29327d5 in __gnu_cxx::__verbose_terminate_handler() () from /lib64/libstdc++.so.6
    #3  0x00007fffe2930746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00007fffe2930773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00007fffe2930993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x00007ffff652e4f0 in optix::APIObj::checkError (this=0x7e7f510, code=RT_ERROR_INVALID_VALUE) at /home/blyth/local/opticks/externals/OptiX/include/optixu/optixpp_namespace.h:2151
    #7  0x00007ffff6570529 in OBufBase::getDevicePtr() () from /home/blyth/local/opticks/lib/../lib64/libOptiXRap.so
    #8  0x00007ffff65706fe in OBufBase::bufspec() () from /home/blyth/local/opticks/lib/../lib64/libOptiXRap.so
    #9  0x00007ffff6552716 in OEvent::downloadHits (this=0x7e54c20, evt=0x7c14680) at /home/blyth/opticks/optixrap/OEvent.cc:412
    #10 0x00007ffff65522ae in OEvent::download (this=0x7e54c20) at /home/blyth/opticks/optixrap/OEvent.cc:354
    #11 0x00007ffff68a517e in OpEngine::downloadEvent (this=0x7049cf0) at /home/blyth/opticks/okop/OpEngine.cc:151
    #12 0x00007ffff79ccc5c in OKPropagator::downloadEvent (this=0x7049a10) at /home/blyth/opticks/ok/OKPropagator.cc:99
    #13 0x00007ffff79cca64 in OKPropagator::propagate (this=0x7049a10) at /home/blyth/opticks/ok/OKPropagator.cc:73
    #14 0x00007ffff7bd5829 in OKG4Mgr::propagate_ (this=0x7fffffffcc70) at /home/blyth/opticks/okg4/OKG4Mgr.cc:190
    #15 0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffcc70) at /home/blyth/opticks/okg4/OKG4Mgr.cc:117
    #16 0x00000000004039a7 in main (argc=32, argv=0x7fffffffcfa8) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) 



Rerun with "--dbgdownload" shows the other buffers downloaded ok, some problem with hits buffer::

    2019-06-09 22:42:31.342 INFO  [249806] [BTimes::dump@146] OPropagator::launch
                    launch001                 0.003147
    2019-06-09 22:42:31.342 INFO  [249806] [OpEngine::propagate@129] ) propagator.launch 
    2019-06-09 22:42:31.342 INFO  [249806] [OpIndexer::indexSequenceInterop@254] OpIndexer::indexSequenceInterop slicing (OBufBase*)m_seq 
    2019-06-09 22:42:31.352 INFO  [249806] [OpEngine::propagate@132] ]
    2019-06-09 22:42:31.352 INFO  [249806] [OpticksViz::indexPresentationPrep@394] OpticksViz::indexPresentationPrep
    2019-06-09 22:42:31.353 INFO  [249806] [OpticksViz::downloadEvent@384] OpticksViz::downloadEvent (1)
    2019-06-09 22:42:31.362 INFO  [249806] [Rdr::download@74] Rdr::download SKIP for sequence as OPTIX_NON_INTEROP
    2019-06-09 22:42:31.362 INFO  [249806] [OpticksViz::downloadEvent@386] OpticksViz::downloadEvent (1) DONE 
    2019-06-09 22:42:31.362 INFO  [249806] [OpEngine::downloadEvent@149] .
    2019-06-09 22:42:31.362 INFO  [249806] [OEvent::download@389] ox 10000,4,4
    2019-06-09 22:42:31.362 INFO  [249806] [OEvent::download@396] rx 10000,10,2,4
    2019-06-09 22:42:31.362 INFO  [249806] [OContext::download@693] OContext::download PROCEED for sequence as OPTIX_NON_INTEROP
    2019-06-09 22:42:31.362 INFO  [249806] [OEvent::download@402] sq 10000,1,2
    terminate called after throwing an instance of 'optix::Exception'
      what():  Invalid value (Details: Function "RTresult _rtBufferGetDevicePointer(RTbuffer, int, void**)" caught exception: Cannot get device pointers from non-CUDA interop buffers.)

    Program received signal SIGABRT, Aborted.
    0x00007fffe2023207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libX11-devel-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libdrm-2.4.91-3.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) 




::

    147 unsigned OpEngine::downloadEvent()
    148 {
    149     LOG(info) << "." ;
    150     LOG(debug) << "[" ;
    151     unsigned n = m_oevt->download();
    152     LOG(debug) << "]" ;
    153     return n ;
    154 }

    351 unsigned OEvent::download()
    352 {
    353     if(!m_ok->isProduction()) download(m_evt, DOWNLOAD_DEFAULT);
    354     return downloadHits(m_evt);
    355 }





container auto sizing not working with proxies : FIXED by a refactor
-------------------------------------------------------------------------


* done in NCSGList::load so not proxy aware

* fixed by refactor of NCSGList GGeoTest 
  and additions to GMaker and GMeshMaker


event and animation timings need auto adjustment as change size of geometry
---------------------------------------------------------------------------------

* when *proxylv* pulls in a big piece of geometry the animation goes real slow 
  as the time ranges are setup for smaller geometry









