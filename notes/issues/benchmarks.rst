benchmarks
==============


Principals


* need running from cache 

* "--compute" mode is what matters 
* time for raytrace snapshots is an obvious metric, eg start from okop/tests/OpSnapTest.cc 

  * need to automate viewpoint and camera params in a non-fragile way (bookmarks are fragile)
    commandline arguments less so

* possibly running without GUI (runlevel 3) can avoid any OpenGL involvement

* use CUDA_VISIBLE_DEVICES 

  1. unset
  2. 0,1   # expect same as unset
  3. 1,0   # expect same as unset
  4. 0
  5. 1

* implement sensitivity to OPTICKS_RTX=0,1 for switching the attribute 
* currently there is an OpenGL way of detecting the GPU for the context, 
  instead need a compute version of that (see UseOptiX) that OContext 
  perhaps holds onto and reports into metadata



First in interop for dev
----------------------------

No obvious change in interop::

    [blyth@localhost optixrap]$ CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1 OKTest --envkey --xanalytic --target 10000
    [blyth@localhost optixrap]$ OPTICKS_RTX=0 OKTest --envkey --xanalytic --target 10000


Found a good viewpoint, looking up at chimney::

    CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1 OKTest --envkey --xanalytic --target 352851 --eye -1,-1,-1        ## analytic
    CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1 OKTest --envkey --target 352851 --eye -1,-1,-1                    ## tri 

* target is 0-based 
* numbers listed in PVNames.txt from *vi* in the below are 1-based 
* 352851 is pLowerChimneyLS0x5b317e0 

GNodeLib/PVNames.txt::

    .1 lWorld0x4bc2710_PV
     2 pTopRock0x4bcd120
     3 pExpHall0x4bcd520
     4 lUpperChimney_phys0x5b308a0
     5 pUpperChimneyLS0x5b2f160
    ...

    352847 PMT_3inch_inner1_phys0x510beb0
    352848 PMT_3inch_inner2_phys0x510bf60
    352849 PMT_3inch_cntr_phys0x510c010
    352850 lLowerChimney_phys0x5b32c20
    352851 pLowerChimneyAcrylic0x5b31720
    352852 pLowerChimneyLS0x5b317e0
    352853 pLowerChimneySteel0x5b318b0
    352854 lSurftube_phys0x5b3c810
    352855 pvacSurftube0x5b3c120
    352856 lMaskVirtual_phys0x5cc1ac0






OpSnapTest : compute only snaps : based on OpMgr which is used only by G4Opticks and OpSnapTest 
---------------------------------------------------------------------------------------------------------

::

    int main(int argc, char** argv)
    {
        OPTICKS_LOG(argc, argv);
        Opticks ok(argc, argv, "--tracer"); 
        OpMgr op(&ok);
        op.snap();
        return 0 ; 
    }

    155 void OpMgr::snap()
    156 {
    157     LOG(info) << "OpMgr::snap" ;
    158     m_propagator->snap();
    159 }

    106 void OpPropagator::snap()
    107 {
    108     LOG(info) << "OpPropagator::snap" ;
    109     m_tracer->snap();
    110 }


     92 /**
     93 OpTracer::snap
     94 ----------------
     95  
     96 Takes one or more GPU raytrace snapshots of geometry
     97 at various positions configured via m_snap_config.  
     98  
     99 **/
    100  
    101 void OpTracer::snap()
    102 {
    103     LOG(info) << "OpTracer::snap START" ;
    104     m_snap_config->dump();
    105  
    106     int num_steps = m_snap_config->steps ;
    107     float eyestartz = m_snap_config->eyestartz ;
    108     float eyestopz = m_snap_config->eyestopz ;
    109  
    110     for(int i=0 ; i < num_steps ; i++)
    111     {
    112         std::string path = m_snap_config->getSnapPath(i) ;
    113  
    114         float frac = num_steps > 1 ? float(i)/float(num_steps-1) : 0.f ;
    115         float eyez = eyestartz + (eyestopz-eyestartz)*frac ;
    116  
    117         std::cout << " i " << std::setw(5) << i
    118                   << " eyez " << std::setw(10) << eyez
    119                   << " path " << path
    120                   << std::endl ;
    121  
    122         m_composition->setEyeZ( eyez );
    123  
    124         render();
    125  
    126         m_ocontext->snap(path.c_str());
    127     }
    128  
    129     LOG(info) << "OpTracer::snap DONE " ;
    130 }

    079 void OpTracer::render()
     80 {
     81     if(m_count == 0 )
     82     {
     83         m_hub->setupCompositionTargetting();
     84         m_otracer->setResolutionScale(1) ;
     85     }
     86  
     87     m_otracer->trace_();
     88     m_count++ ;
     89 }


Launch times are collected into m_trace_times STimes instance held in OTracer 
with sums compile/prelaunch/launch times and counts calls (so effectively average timings over all snaps).::

    284 void OContext::launch(unsigned int lmode, unsigned int entry, unsigned int width, unsigned int height, STimes* times )
    285 {
    286     if(!m_closed) close();
    287 
    288     LOG(LEVEL)
    289               << " entry " << entry
    290               << " width " << width
    291               << " height " << height
    292               ;
    293 
    294     if(times) times->count     += 1 ;
    295 
    296     if(lmode & VALIDATE)
    297     {
    298         double dt = validate_();
    299         LOG(LEVEL) << "VALIDATE time: " << dt ;
    300         if(times) times->validate  += dt  ;
    301     }
    302 
    303     if(lmode & COMPILE)
    304     {
    305         double dt = compile_();
    306         LOG(LEVEL) << "COMPILE time: " << dt ;
    307         if(times) times->compile  += dt ;
    308     }
    309 
    310     if(lmode & PRELAUNCH)
    311     {
    312         double dt = launch_(entry, width, height );
    313         LOG(LEVEL) << "PRELAUNCH time: " << dt ;
    314         if(times) times->prelaunch  += dt ;
    315     }
    316 
    317     if(lmode & LAUNCH)
    318     {
    319         double dt = m_llogpath ? launch_redirected_(entry, width, height ) : launch_(entry, width, height );
    320         LOG(LEVEL) << "LAUNCH time: " << dt  ;
    321         if(times) times->launch  += dt  ;
    322     }
    323 }
    324 





::

    OpSnapTest --envkey --xanalytic --target 10000

    ...

    2019-04-19 22:58:50.772 INFO  [441886] [OTracer::trace_@128] OTracer::trace  entry_index 0 trace_count 0 resolution_scale 1 size(1920,1080) ZProj.zw (-1.04082,-1155) front 0.6061,0.6061,-0.5152
    2019-04-19 22:58:52.304 INFO  [441886] [OContext::snap@681]  path /tmp/snap00000.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00000.ppm
     i     1 eyez   0.838889 path /tmp/snap00001.ppm
    2019-04-19 22:58:52.379 INFO  [441886] [OContext::snap@681]  path /tmp/snap00001.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00001.ppm
     i     2 eyez   0.827778 path /tmp/snap00002.ppm
    2019-04-19 22:58:52.451 INFO  [441886] [OContext::snap@681]  path /tmp/snap00002.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00002.ppm
     i     3 eyez   0.816667 path /tmp/snap00003.ppm
    2019-04-19 22:58:52.520 INFO  [441886] [OContext::snap@681]  path /tmp/snap00003.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00003.ppm
     i     4 eyez   0.805556 path /tmp/snap00004.ppm
    2019-04-19 22:58:52.588 INFO  [441886] [OContext::snap@681]  path /tmp/snap00004.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00004.ppm
     i     5 eyez   0.794444 path /tmp/snap00005.ppm
    2019-04-19 22:58:52.656 INFO  [441886] [OContext::snap@681]  path /tmp/snap00005.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00005.ppm
     i     6 eyez   0.783333 path /tmp/snap00006.ppm
    2019-04-19 22:58:52.724 INFO  [441886] [OContext::snap@681]  path /tmp/snap00006.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00006.ppm
     i     7 eyez   0.772222 path /tmp/snap00007.ppm
    2019-04-19 22:58:52.791 INFO  [441886] [OContext::snap@681]  path /tmp/snap00007.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00007.ppm
     i     8 eyez   0.761111 path /tmp/snap00008.ppm
    2019-04-19 22:58:52.859 INFO  [441886] [OContext::snap@681]  path /tmp/snap00008.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00008.ppm
     i     9 eyez       0.75 path /tmp/snap00009.ppm
    2019-04-19 22:58:52.927 INFO  [441886] [OContext::snap@681]  path /tmp/snap00009.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00009.ppm
    2019-04-19 22:58:52.948 INFO  [441886] [OpTracer::snap@129] OpTracer::snap DONE 
    2019-04-19 22:58:52.948 ERROR [441886] [OpticksHub::cleanup@991] OpticksHub::cleanup







Next

* revive the profiling/metadata machinery 
* make the snaps more demanding : maybe screen resolution 2560x1440 or twice that 
* output directory control in snap config


::


    CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=-1 OpSnapTest --envkey --xanalytic --target 10000
    CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=0  OpSnapTest --envkey --xanalytic --target 10000
    CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1  OpSnapTest --envkey --xanalytic --target 10000


No difference, bumping against overheads::

    2019-04-19 23:31:37.973 INFO  [35651] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count    10 
     validate      0.0557     0.0056 
     compile       0.0000     0.0000 
     prelaunch     1.1470     0.1147 
     launch        0.4722     0.0472 

    2019-04-19 23:33:49.576 INFO  [39125] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count    10 
     validate      0.0541     0.0054 
     compile       0.0000     0.0000 
     prelaunch     1.2097     0.1210 
     launch        0.4565     0.0457 


::

    OPTICKS_RTX_REQUEST=0 OpSnapTest --envkey --xanalytic --target 10000 --size 2560,1440,1
    OPTICKS_RTX_REQUEST=1 OpSnapTest --envkey --xanalytic --target 10000 --size 2560,1440,1

    OPTICKS_RTX_REQUEST=0 OpSnapTest --envkey --xanalytic --target 10000 --size 5120,2880,1
    OPTICKS_RTX_REQUEST=1 OpSnapTest --envkey --xanalytic --target 10000 --size 5120,2880,1


::

    2019-04-19 23:51:10.019 INFO  [66551] [OContext::snap@681]  path /tmp/snap00008.ppm width 5120 width 5120 height 2880 height 2880 depth 1
    Wrote file /tmp/snap00008.ppm
     i     9 eyez       0.75 path /tmp/snap00009.ppm
    2019-04-19 23:51:10.450 INFO  [66551] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count    10 
     validate      0.1000     0.0100 
     compile       0.0000     0.0000 
     prelaunch     1.4062     0.1406 
     launch        2.7674     0.2767 






TITAN V looks clear winner over TITAN RTX and RTX mode aint changing anything::

    [blyth@localhost okop]$ CUDA_VISIBLE_DEVICES=0 OPTICKS_RTX_REQUEST=1 OpSnapTest --envkey --xanalytic --target 10000 --size 5120,2880,1

    rote file /tmp/snap00008.ppm
     i     9 eyez       0.75 path /tmp/snap00009.ppm
    2019-04-19 23:56:14.063 INFO  [74700] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count    10 
     validate      0.0554     0.0055 
     compile       0.0000     0.0000 
     prelaunch     3.4029     0.3403 
     launch        1.9315     0.1932 


    [blyth@localhost okop]$ CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX_REQUEST=1 OpSnapTest --envkey --xanalytic --target 10000 --size 5120,2880,1

    2019-04-20 00:00:25.528 INFO  [81734] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count    10 
     validate      0.0543     0.0054 
     compile       0.0000     0.0000 
     prelaunch     1.3829     0.1383 
     launch        2.7688     0.2769 

    2019-04-20 00:00:25.528 INFO  [81734] [OContext::snap@681]  path /tmp/snap00009.ppm width 5120 width 5120 height 2880 height 2880 depth 1



Unless I am missing something. 

* perhaps compiling with CC 75 rather than current 70 ?
* also need to check with snap paths across more demanding geometry 

Take a look at a more demanding render over in env- rtow-



Perhaps JIT compilation killing perfermanance for TITAN RTX ?

cmake/Modules/OpticksCUDAFlags.cmake needs to handle a comma delimited COMPUTE_CAPABILITY ?::

     09 if(NOT (COMPUTE_CAPABILITY LESS 30))
     10 
     11    #list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${COMPUTE_CAPABILITY}")
     12    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
     13    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
     14 
     15    #list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
     16    # https://github.com/facebookresearch/Detectron/issues/185
     17 
     18    list(APPEND CUDA_NVCC_FLAGS "-O2")
     19    #list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
     20    list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
     21 
     22    #list(APPEND CUDA_NVCC_FLAGS "-m64")
     23    #list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")
     24 
     25    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
     26    set(CUDA_VERBOSE_BUILD OFF)
     27 
     28 endif()




