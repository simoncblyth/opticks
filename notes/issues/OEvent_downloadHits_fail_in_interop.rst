OEvent_downloadHits_fail_in_interop : FIXED 
================================================

context
---------

* :doc:`tboolean-with-proxylv-bringing-in-basis-solids`


maybe FIXED : just missing imp in --interop
----------------------------------------------

::

    456 unsigned OEvent::downloadHitsInterop(OpticksEvent* evt)
    457 {
    458     OK_PROFILE("_OEvent::downloadHitsInterop");
    459 
    460     NPY<float>* hit = evt->getHitData();
    461     NPY<float>* npho = evt->getPhotonData();
    462 
    463     unsigned photon_id = npho->getBufferId();
    464 
    465     LOG(fatal) << "[ cpho" ;
    466     CResource rphoton( photon_id, CResource::R );
    467     //CBufSpec cpho = rphoton.mapGLToCUDA<cfloat4x4>();
    468     CBufSpec cpho = rphoton.mapGLToCUDA<float>();
    469     LOG(fatal) << "] cpho DONE " ;
    470 
    471     assert( cpho.size % 16 == 0 );
    472     cpho.size /= 16 ;    //  decrease size by factor of 16, increases cpho "item" from 1*float to 4*4*float 
    473 
    474 
    475     bool verbose = true ;
    476     TBuf tpho("tpho", cpho );
    477     unsigned nhit = tpho.downloadSelection4x4("OEvent::downloadHits", hit, verbose);
    478     // hit buffer (0,4,4) resized to fit downloaded hits (nhit,4,4)
    479     assert(hit->hasShape(nhit,4,4));
    480 
    481     OK_PROFILE("OEvent::downloadHitsInterop");
    482 
    483     return nhit ;
    484 }



issue : downloadHits aborts in --interop
------------------------------------------------------

try to viz (--interop) and propagate together fails : the old linux chestnut ?

* maybe not the same old problem : commenting out downloading hits makes it work.

::

     PROXYLV=2 tboolean.sh --interop --dbgdownload -D 


* now "--interop" trumps "--compute" within the same commandline  
* presence of PROXYLV auto-chooses tboolean-proxy      

::

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



Hits are special
------------------

Hits are special, because they are a selection of the photons buffer downloaded via stream 
compaction with Thrust.

In interop that means have to talk to the buffer from : OptiX/Thrust/OpenGL 
whereas in compute just OptiX/Thrust 

::

    415 unsigned OEvent::downloadHits(OpticksEvent* evt)
    416 {
    417     OK_PROFILE("_OEvent::downloadHits");
    418 
    419     NPY<float>* hit = evt->getHitData();
    420 
    421     
    422     LOG(info) << "[ cpho" ;
    423     CBufSpec cpho = m_photon_buf->bufspec();
    424     LOG(info) << "] cpho DONE " ;
    425     assert( cpho.size % 4 == 0 );
    426     cpho.size /= 4 ;    //  decrease size by factor of 4, increases cpho "item" from 1*float4 to 4*float4 
    427 
    428     bool verbose = false ;
    429     TBuf tpho("tpho", cpho );
    430     unsigned nhit = tpho.downloadSelection4x4("OEvent::downloadHits", hit, verbose);
    431     // hit buffer (0,4,4) resized to fit downloaded hits (nhit,4,4)
    432     assert(hit->hasShape(nhit,4,4));
    433 
    434     OK_PROFILE("OEvent::downloadHits");
    435 
    436     return nhit ;
    437 }




::

    2019-06-12 20:16:48.777 INFO  [75619] [OContext::download@693] OContext::download PROCEED for sequence as OPTIX_NON_INTEROP
    2019-06-12 20:16:48.777 INFO  [75619] [OEvent::downloadHits@422] [ cpho
    terminate called after throwing an instance of 'optix::Exception'
      what():  Invalid value (Details: Function "RTresult _rtBufferGetDevicePointer(RTbuffer, int, void**)" caught exception: Cannot get device pointers from non-CUDA interop buffers.)
    
    Program received signal SIGABRT, Aborted.
    ...
    #3  0x00007fffe2928746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00007fffe2928773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00007fffe2928993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x00007ffff652e550 in optix::APIObj::checkError (this=0x7c02100, code=RT_ERROR_INVALID_VALUE) at /home/blyth/local/opticks/externals/OptiX/include/optixu/optixpp_namespace.h:2151
    #7  0x00007ffff6570b49 in OBufBase::getDevicePtr() () from /home/blyth/local/opticks/lib/../lib64/libOptiXRap.so
    #8  0x00007ffff6570d1e in OBufBase::bufspec() () from /home/blyth/local/opticks/lib/../lib64/libOptiXRap.so
    #9  0x00007ffff6552d32 in OEvent::downloadHits (this=0x7cbccc0, evt=0x7d698c0) at /home/blyth/opticks/optixrap/OEvent.cc:423
    #10 0x00007ffff655232c in OEvent::download (this=0x7cbccc0) at /home/blyth/opticks/optixrap/OEvent.cc:356
    #11 0x00007ffff68a517e in OpEngine::downloadEvent (this=0x70295b0) at /home/blyth/opticks/okop/OpEngine.cc:151
    #12 0x00007ffff79ccc5c in OKPropagator::downloadEvent (this=0x7028390) at /home/blyth/opticks/ok/OKPropagator.cc:99
    #13 0x00007ffff79cca64 in OKPropagator::propagate (this=0x7028390) at /home/blyth/opticks/ok/OKPropagator.cc:73
    #14 0x00007ffff7bd5829 in OKG4Mgr::propagate_ (this=0x7fffffffcc10) at /home/blyth/opticks/okg4/OKG4Mgr.cc:190
    #15 0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffcc10) at /home/blyth/opticks/okg4/OKG4Mgr.cc:117
    #16 0x00000000004039a7 in main (argc=34, argv=0x7fffffffcf48) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) f 9
    #9  0x00007ffff6552d32 in OEvent::downloadHits (this=0x7cbccc0, evt=0x7d698c0) at /home/blyth/opticks/optixrap/OEvent.cc:423
    423     CBufSpec cpho = m_photon_buf->bufspec();  
    (gdb) f 8
    #8  0x00007ffff6570d1e in OBufBase::bufspec() () from /home/blyth/local/opticks/lib/../lib64/libOptiXRap.so
    (gdb) 



opticks-f bufspec::

     ./okop/tests/compactionTest.cc:    CBufSpec cpho = pbuf->bufspec();   // getDevicePointer happens here with OBufBase::bufspec

Need OpenGL+OptiX to test thus, so has to be OKGL, OK or above::

    [blyth@localhost opticks]$ opticks-deps
     10          OKCONF :               okconf :               OKConf : OpticksCUDA OptiX G4  
     20          SYSRAP :               sysrap :               SysRap : OKConf PLog  
     30            BRAP :             boostrap :             BoostRap : Boost PLog SysRap  
     40             NPY :                  npy :                  NPY : PLog GLM OpenMesh BoostRap YoctoGL ImplicitMesher DualContouringSample  
     45             YOG :           yoctoglrap :           YoctoGLRap : NPY  
     50          OKCORE :          optickscore :          OpticksCore : NPY  
     60            GGEO :                 ggeo :                 GGeo : OpticksCore YoctoGLRap  
     70          ASIRAP :            assimprap :            AssimpRap : OpticksAssimp GGeo  
     80         MESHRAP :          openmeshrap :          OpenMeshRap : GGeo OpticksCore  
     90           OKGEO :           opticksgeo :           OpticksGeo : OpticksCore AssimpRap OpenMeshRap  
    100         CUDARAP :              cudarap :              CUDARap : SysRap OpticksCUDA  
    110           THRAP :            thrustrap :            ThrustRap : OpticksCore CUDARap  
    120           OXRAP :             optixrap :             OptiXRap : OKConf OptiX OpticksGeo ThrustRap  
    130            OKOP :                 okop :                 OKOP : OptiXRap  
    140          OGLRAP :               oglrap :               OGLRap : ImGui OpticksGLEW OpticksGLFW OpticksGeo  
    150            OKGL :            opticksgl :            OpticksGL : OGLRap OKOP  
    160              OK :                   ok :                   OK : OpticksGL  
    165              X4 :                extg4 :                ExtG4 : G4 GGeo OpticksXercesC  
    170            CFG4 :                 cfg4 :                 CFG4 : G4 ExtG4 OpticksXercesC OpticksGeo  
    180            OKG4 :                 okg4 :                 OKG4 : OK CFG4  
    190            G4OK :                 g4ok :                 G4OK : CFG4 ExtG4 OKOP 


::

     08 template <typename T>
     09 CBufSpec make_bufspec(const thrust::device_vector<T>& d_vec )
     10 {
     11     const T* raw_ptr = thrust::raw_pointer_cast(d_vec.data());
     12 
     13     unsigned int size = d_vec.size() ;
     14     unsigned int nbytes =  size*sizeof(T) ;
     15 
     16     return CBufSpec( (void*)raw_ptr, size, nbytes );
     17 }
     18 
     19 

::

    007 OBufBase::OBufBase(const char* name, optix::Buffer& buffer)
     ..
     30 CBufSpec OBufBase::bufspec()
     31 {
     32    return CBufSpec( getDevicePtr(), getSize(), getNumBytes()) ;
     33 }
     34 

    201 void* OBufBase::getDevicePtr()
    202 {
    203     //printf("OBufBase::getDevicePtr %s \n", ( m_name ? m_name : "-") ) ;
    204     //return (void*) m_buffer->getDevicePointer(m_device); 
    205 
    206     CUdeviceptr cu_ptr = (CUdeviceptr)m_buffer->getDevicePointer(m_device) ;
    207     return (void*)cu_ptr ;
    208 }




Smoking gun, this is assuming m_device ordinal zero (ie a single GPU)::

     07 OBufBase::OBufBase(const char* name, optix::Buffer& buffer)
      8    :
      9    m_buffer(buffer),
     10    m_name(strdup(name)),
     11    m_multiplicity(0u),
     12    m_sizeofatom(0u),
     13    m_device(0u),
     14    m_hexdump(false)
     15 {
     16     init();
     17 }


::

    [blyth@localhost include]$ optix-ifind getDevicePointer
    /home/blyth/local/opticks/externals/OptiX/include/optixu/optixpp_namespace.h:    void getDevicePointer( int optix_device_ordinal, void** device_pointer );
    /home/blyth/local/opticks/externals/OptiX/include/optixu/optixpp_namespace.h:    void* getDevicePointer( int optix_device_ordinal );
    /home/blyth/local/opticks/externals/OptiX/include/optixu/optixpp_namespace.h:  inline void BufferObj::getDevicePointer(int optix_device_ordinal, void** device_pointer)
    /home/blyth/local/opticks/externals/OptiX/include/optixu/optixpp_namespace.h:  inline void* BufferObj::getDevicePointer(int optix_device_ordinal)
    /home/blyth/local/opticks/externals/OptiX/include/optixu/optixpp_namespace.h:    getDevicePointer( optix_device_ordinal, &dptr );

::

    1808     /// Get the pointer to buffer memory on a specific device. See @ref rtBufferGetDevicePointer
    1809     void getDevicePointer( int optix_device_ordinal, void** device_pointer );
    1810     void* getDevicePointer( int optix_device_ordinal );
    ...
    4604   inline void BufferObj::getDevicePointer(int optix_device_ordinal, void** device_pointer)
    4605   {
    4606     checkError( rtBufferGetDevicePointer( m_buffer, optix_device_ordinal, device_pointer ) );
    4607   }
    4608 
    4609   inline void* BufferObj::getDevicePointer(int optix_device_ordinal)
    4610   {
    4611     void* dptr;
    4612     getDevicePointer( optix_device_ordinal, &dptr );
    4613     return dptr;
    4614   }


/home/blyth/local/opticks/externals/OptiX/include/optix_cuda_interop.h
/home/blyth/local/opticks/externals/OptiX/include/optix_gl_interop.h


FromGLBO::

    blyth@localhost issues]$ opticks-f FromGLBO
    ./bin/oks.bash:    352         buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT_OUTPUT, buffer_id);
    ./bin/oks.bash:the OpenGL buffer is referred by id to createBufferFromGLBO and
    ./bin/oks.bash:In OptiX 4 this is not working for FromGLBO buffers::
    ./bin/oks.bash:* OpenGL/OptiX/CUDA interop has changed : can no longer get a CUDA pointer in from a FromGLBO OptiX buffer 
    ./bin/oks.bash:2016-07-21 16:35:24.083 INFO  [9524] [OContext::createIOBuffer@324] OContext::createIOBuffer (INTEROP) createBufferFromGLBO  name             gensteps buffer_id 16
    ./bin/oks.bash:2016-07-21 16:35:24.083 INFO  [9524] [OContext::createIOBuffer@324] OContext::createIOBuffer (INTEROP) createBufferFromGLBO  name               photon buffer_id 18
    ./bin/oks.bash:2016-07-21 16:35:24.083 INFO  [9524] [OContext::createIOBuffer@324] OContext::createIOBuffer (INTEROP) createBufferFromGLBO  name               record buffer_id 19
    ./bin/oks.bash:2016-07-21 16:35:24.083 INFO  [9524] [OContext::createIOBuffer@324] OContext::createIOBuffer (INTEROP) createBufferFromGLBO  name             sequence buffer_id 20
    ./ok/ok.bash:        m_genstep_buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, genstep_buffer_id);
    ./optixrap/OptiXPBO.cc:optix::Buffer buffer = optixContext->createBufferFromGLBO(RT_BUFFER_INPUT, buffer->setFormat(RT_FORMAT_USER);
    ./optixrap/OGeo.cc:        buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, buffer_id);
    ./optixrap/OGeo.cc:        buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, buffer_id);
    ./optixrap/OContext.cc:             LOG(fatal) << "OContext::createBuffer CANNOT createBufferFromGLBO as not uploaded  "
    ./optixrap/OContext.cc:         buffer = m_context->createBufferFromGLBO(type, buffer_id);
    ./optixrap/OEvent.cc:    // with createBufferFromGLBO by Scene::uploadEvt Scene::uploadSelection
    ./opticksgl/OFrame.cc:    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, id);
    ./opticksgl/OFrame.cc:    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, id);
    ./opticksgl/OFrame.hh:        // create GL buffer VBO/PBO first then address it as OptiX buffer with optix::Context::createBufferFromGLBO  
    ./externals/optixnote.bash:    OBuffer::mapGLToOptiX (createBufferFromGLBO) 1  size 30
    ./externals/optixnote.bash:with rtBufferCreateFromGLBO. The resulting buffer is a reference only to the
    ./thrustrap/thrap.bash:    m_genstep_buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, genstep_buffer_id);
    ./thrustrap/thrap.bash:with rtBufferCreateFromGLBO. The resulting buffer is a reference only to the
    [blyth@localhost opticks]$ 


Look into photon buffer creation
--------------------------------------

::


    145         optix::Buffer   m_photon_buffer ;
    ...
    156         OBuf*           m_photon_buf ;


    080 void OEvent::createBuffers(OpticksEvent* evt)
    081 {
    ...

    108     NPY<float>* photon = evt->getPhotonData() ;
    109     assert(photon);
    110 
    111     OpticksBufferControl* photonCtrl = evt->getPhotonCtrl();
    112     m_photonMarkDirty = photonCtrl->isSet("BUFFER_COPY_ON_DIRTY") ;
    113 
    114     m_photon_buffer = m_ocontext->createBuffer<float>( photon, "photon");
    115 
    116     m_context["photon_buffer"]->set( m_photon_buffer );
    117     m_photon_buf = new OBuf("photon", m_photon_buffer);
    118 
    119 


compute 
    ordinary OptiX buffer

interop 
    createBufferFromGLBO 
 
    * OpenGL needs access for the visualization of photons
    * actually the photon viz is less used that the record points as its just 
      final positions     
 

::

     725 template <typename T>
     726 optix::Buffer OContext::createBuffer(NPY<T>* npy, const char* name)
     727 {
     728     assert(npy);
     729     OpticksBufferControl ctrl(npy->getBufferControlPtr());
     730     bool verbose = ctrl("VERBOSE_MODE") || SSys::IsVERBOSE() ;
     731 
     732     bool compute = isCompute()  ;
     ...
     758     optix::Buffer buffer ;
     759 
     760     if( compute )
     761     {
     762         buffer = m_context->createBuffer(type);
     763     }   
     764     else if( ctrl("OPTIX_NON_INTEROP") )
     765     {
     766         buffer = m_context->createBuffer(type);
     767     }   
     768     else
     769     {
     770         int buffer_id = npy ? npy->getBufferId() : -1 ;
     771         if(!(buffer_id > -1))
     772             LOG(fatal)  
     773                 << "CANNOT createBufferFromGLBO as not uploaded  "
     774                 << " name " << std::setw(20) << name
     775                 << " buffer_id " << buffer_id  
     776                 ;        
     777         assert(buffer_id > -1 );
     778         buffer = m_context->createBufferFromGLBO(type, buffer_id);
     779     }
     780 
     781     configureBuffer<T>(buffer, npy, name );
     782     return buffer ;
     783 }
     784 






::

    2019-06-12 23:20:39.423 INFO  [411280] [OpEngine::uploadEvent@108] .
    2019-06-12 23:20:39.423 ERROR [411280] [OContext::createBuffer@779] createBufferFromGLBO name             gensteps buffer_id 19
    2019-06-12 23:20:39.424 ERROR [411280] [OContext::createBuffer@779] createBufferFromGLBO name               photon buffer_id 21
    2019-06-12 23:20:39.424 ERROR [411280] [OContext::createBuffer@779] createBufferFromGLBO name               source buffer_id 22
    2019-06-12 23:20:39.424 ERROR [411280] [OContext::createBuffer@779] createBufferFromGLBO name               record buffer_id 23
    2019-06-12 23:20:39.424 INFO  [411280] [OEvent::uploadGensteps@312] OEvent::uploadGensteps (INTEROP) SKIP OpenGL BufferId 19
    2019-06-12 23:20:39.424 INFO  [411280] [OEvent::uploadSource@332] OEvent::uploadSource (INTEROP) SKIP OpenGL BufferId 22


* in interop mode, unless the buffer is marked as OPTIX_NON_INTEROP it gets treated as an interop buffer.
* buffers without visualizations are marked OPTIX_NON_INTEROP in OpticksBufferSpec

::

    [blyth@localhost optickscore]$ OpticksBufferSpecTest 
    2019-06-12 20:29:58.498 INFO  [96832] [main@15] OKCONF_OPTIX_VERSION_INTEGER : 60000
    2019-06-12 20:29:58.498 INFO  [96832] [main@16] OKCONF_OPTIX_VERSION_MAJOR   : 6
    2019-06-12 20:29:58.499 INFO  [96832] [main@17] OKCONF_OPTIX_VERSION_MINOR   : 0
    2019-06-12 20:29:58.499 INFO  [96832] [main@18] OKCONF_OPTIX_VERSION_MICRO   : 0
    2019-06-12 20:29:58.499 INFO  [96832] [main@22] OKCONF_GEANT4_VERSION_INTEGER : 1042
    2019-06-12 20:29:58.499 INFO  [96832] [main@26] WITH_SEED_BUFFER

    COMPUTE
                 genstep : OPTIX_INPUT_ONLY,UPLOAD_WITH_CUDA,BUFFER_COPY_ON_DIRTY
                 nopstep : 
                  photon : OPTIX_OUTPUT_ONLY
                  source : OPTIX_INPUT_ONLY,UPLOAD_WITH_CUDA,BUFFER_COPY_ON_DIRTY,VERBOSE_MODE
                  record : OPTIX_OUTPUT_ONLY
                  phosel : 
                  recsel : 
                sequence : OPTIX_NON_INTEROP,OPTIX_OUTPUT_ONLY
                    seed : OPTIX_NON_INTEROP,OPTIX_INPUT_ONLY
                     hit : 

    INTEROP
                 genstep : OPTIX_INPUT_ONLY
                 nopstep : 
                  photon : OPTIX_OUTPUT_ONLY,**INTEROP_PTR_FROM_OPENGL**
                  source : OPTIX_INPUT_ONLY
                  record : OPTIX_OUTPUT_ONLY
                  phosel : 
                  recsel : 
                sequence : OPTIX_NON_INTEROP,OPTIX_OUTPUT_ONLY
                    seed : OPTIX_NON_INTEROP,OPTIX_INPUT_ONLY
                     hit : 



INTEROP_PTR_FROM_OPENGL
--------------------------

The setting **INTEROP_PTR_FROM_OPENGL** is currently only honoured for indexing 

::

    [blyth@localhost opticks]$ opticks-f INTEROP_PTR_FROM_OPENGL
    ./examples/UseOptiXRap/UseOptiXRap.cc:   //const char* photon_ctrl  = "OPTIX_INPUT_OUTPUT,INTEROP_PTR_FROM_OPENGL" ;
    ./optickscore/OpticksBufferControl.cc:const char* OpticksBufferControl::INTEROP_PTR_FROM_OPENGL_ = "INTEROP_PTR_FROM_OPENGL" ; 
    ./optickscore/OpticksBufferControl.cc:    tags.push_back(INTEROP_PTR_FROM_OPENGL_);
    ./optickscore/OpticksBufferControl.cc:   if( ctrl & INTEROP_PTR_FROM_OPENGL     ) ss << INTEROP_PTR_FROM_OPENGL_ << " "; 
    ./optickscore/OpticksBufferControl.cc:    else if(strcmp(k,INTEROP_PTR_FROM_OPENGL_)==0)    tag = INTEROP_PTR_FROM_OPENGL ;
    ./optickscore/OpticksBufferControl.hh:                INTEROP_PTR_FROM_OPENGL = 0x1 << 7,
    ./optickscore/OpticksBufferControl.hh:        static const char* INTEROP_PTR_FROM_OPENGL_ ; 
    ./optickscore/OpticksBufferSpec.cc: INTEROP_PTR_FROM_OPENGL  
    ./optickscore/OpticksBufferSpec.cc:const char* OpticksBufferSpec::photon_interop_ = "OPTIX_OUTPUT_ONLY,INTEROP_PTR_FROM_OPENGL"  ;
    ./optickscore/OpticksBufferSpec.cc:const char* OpticksBufferSpec::photon_interop_ = "OPTIX_INPUT_OUTPUT,BUFFER_COPY_ON_DIRTY,INTEROP_PTR_FROM_OPENGL"  ;
    ./optickscore/OpticksBufferSpec.cc:const char* OpticksBufferSpec::photon_interop_ = "OPTIX_OUTPUT_ONLY,INTEROP_PTR_FROM_OPENGL"  ;
    ./optickscore/OpticksBufferSpec.cc:const char* OpticksBufferSpec::photon_interop_ = "OPTIX_INPUT_OUTPUT,INTEROP_PTR_FROM_OPENGL,BUFFER_COPY_ON_DIRTY"  ;
    ./optixrap/tests/bufferTest.cc:   const char* photon_ctrl  = "OPTIX_INPUT_OUTPUT,INTEROP_PTR_FROM_OPENGL" ;
    ./okop/OpIndexer.cc:    else if(ctrl & OpticksBufferControl::INTEROP_PTR_FROM_OPENGL)


::

    120 void OpIndexer::indexBoundaries()
    121 {
    122     OK_PROFILE("_OpIndexer::indexBoundaries");
    123 
    124     update();
    125 
    126     if(!m_pho)
    127     {
    128         LOG(warning) << "OpIndexer::indexBoundaries OBuf m_pho is NULL : SKIPPING " ;
    129         return ;
    130     }
    131 
    132 
    133     bool compute = m_ocontext->isCompute() ;
    134     //NPYBase* npho = m_pho->getNPY();
    135     NPYBase* npho = m_evt->getData(OpticksEvent::photon_);
    136     unsigned int buffer_id = npho->getBufferId();
    137     unsigned long long ctrl = npho->getBufferControl();
    138 
    139     unsigned int stride = 4*4 ;
    140     unsigned int begin  = 4*3+0 ;
    141 
    142     if(compute)
    143     {
    144          indexBoundariesFromOptiX(m_pho, stride, begin);
    145     }
    146     else if(ctrl & OpticksBufferControl::INTEROP_PTR_FROM_OPTIX )
    147     {
    148          indexBoundariesFromOptiX(m_pho, stride, begin);
    149     }
    150     else if(ctrl & OpticksBufferControl::INTEROP_PTR_FROM_OPENGL)
    151     {
    152          assert(buffer_id > 0);
    153          indexBoundariesFromOpenGL(buffer_id, stride, begin);
    154     }
    155     else
    156     {
    157          assert(0 && "NO BUFFER CONTROL");
    158     }
    159 
    160 
    161     OK_PROFILE("OpIndexer::indexBoundaries");
    162 }
    163 



OpIndexer::indexBoundariesFromOptiX OR OpenGL 
---------------------------------------------------------------------------------------

* they differ by how to get access to the buffer pointer

* huh kinda funny to be in okop ? as that is beneath OGLRAP ?
* but its OK as OpenGL comes in only with the index of the buffer, no headers or enums or anything 

::


    148 void OpIndexer::indexBoundariesFromOptiX(OBuf* pho, unsigned int stride, unsigned int begin)
    149 {
    150      CBufSlice cbnd = pho->slice(stride,begin) ;    // gets CUDA devPtr from OptiX
    151 
    152      TSparse<int> boundaries(OpticksConst::BNDIDX_NAME_, cbnd, false); // hexkey effects Index and dumping only 
    153     
    154      m_evt->setBoundaryIndex(boundaries.getIndex());
    155      
    156      boundaries.make_lookup();
    157      
    158      if(m_verbose)
    159         boundaries.dump("OpIndexer::indexBoundariesFromOptiX INTEROP_PTR_FROM_OPTIX TSparse<int>::dump");
    160 }    


    162 void OpIndexer::indexBoundariesFromOpenGL(unsigned int photon_id, unsigned int stride, unsigned int begin)
    163 {
    164     // NB this is not using the OptiX buffer, 
    165     //    OpenGL buffer is interop to CUDA accessed directly 
    166 
    167     CResource rphoton( photon_id, CResource::R );
    168 
    169     CBufSpec rph = rphoton.mapGLToCUDA<int>();    // gets CUDA devPtr from OpenGL
    170     {
    171         CBufSlice cbnd = rph.slice(stride,begin) ; // stride, begin  
    172 
    173         TSparse<int> boundaries(OpticksConst::BNDIDX_NAME_, cbnd, false);
    174 
    175         m_evt->setBoundaryIndex(boundaries.getIndex());
    176 
    177         boundaries.make_lookup();
    178 
    179         if(m_verbose)
    180            boundaries.dump("OpIndexer::indexBoundariesFromOpenGL INTEROP_PTR_FROM_OPTIX TSparse<int>::dump");
    181 
    182         rphoton.unmapGLToCUDA();
    183     }
    184 }

cudarap/CResource_.cu
     provides the CResource::mapGLToCUDA






