interop-seedComputeSeedsFromInteropGensteps-fails-with-cudaErrorInvalidDevice
=================================================================================



FIXED : this was cvd confusion again
---------------------------------------

Is this is the same old familiar interop Linux issue again ? Nope its cvd confusion.

* see :doc:`nvidia-smi-slot-index-doesnt-match-UseOptiX-index-and-may-change-on-reboot`

::

    geocache-gui () 
    { 
        local dbg;
        [ -n "$DBG" ] && dbg="gdb --args" || dbg="";
        local cvd=1;
        UseOptiX --cvd $cvd;
        $dbg OKTest --cvd $cvd --envkey --xanalytic --timemax 400 --animtimemax 400 --target 352851 --eye -1,-1,-1
    }


TODO : in interop mode do automatic assert matching between OpenGL renderer and OptiX visible GPU 
----------------------------------------------------------------------------------------------------

::

   2019-05-10 20:48:52.353 ERROR [12774] [Frame::initContext@356] Frame::gl_init_window Renderer: TITAN RTX/PCIe/SSE2





Possible workaround 
--------------------------

If so the workaround is:

1. create and save propagation  "--compute --save --cvd 0,1 --rtx 0"

   Because are using compute are at liberty to use both GPUs "--cvd 0,1"  or more on the cluster
   for the propagation so long as use "--rtx 0" when they are not all Turing or later.


2. visualize it with  "--load --cvd 0" (check the cvd slot needs to match the GPU driving the display, 
   have observed this slot to change after a reboot).
 


issue : interop fails with cudaErrorInvalidDevice
----------------------------------------------------- 


::

    [blyth@localhost optickscore]$ t geocache-gui
    geocache-gui is a function
    geocache-gui () 
    { 
        local dbg;
        [ -n "$DBG" ] && dbg="gdb --args" || dbg="";
        $dbg OKTest --cvd 0 --envkey --xanalytic --timemax 400 --animtimemax 400 --target 352851 --eye -1,-1,-1
    }



::

    ...
    2019-05-10 17:30:17.629 INFO  [159840] [OGeo::convertMergedMesh@232] ( 5
    2019-05-10 17:30:17.629 INFO  [159840] [OGeo::makeOGeometry@495] ugeocode [A]
    2019-05-10 17:30:17.629 INFO  [159840] [OGeo::makeAnalyticGeometry@544] pts:  GParts  primflag         flagnodetree numParts   31 numPrim    1
    2019-05-10 17:30:17.642 INFO  [159840] [OGeo::convertMergedMesh@264] ) 5 numInstances 480
    2019-05-10 17:30:17.642 INFO  [159840] [OGeo::convert@227] ] nmm 6
    2019-05-10 17:30:17.823 INFO  [159840] [OScene::init@165] ]
    2019-05-10 17:30:18.739 INFO  [159840] [SLog::SLog@12]  ( OKGLTracer::OKGLTracer 
    2019-05-10 17:30:18.767 ERROR [159840] [OTracer::init@79]  isTimeTracer NO timetracerscale 1e-06
    2019-05-10 17:30:18.823 INFO  [159840] [SLog::operator@28]  ) OKGLTracer::OKGLTracer  DONE
    2019-05-10 17:30:18.823 INFO  [159840] [SLog::operator@28]  ) OKPropagator::OKPropagator  DONE
    OKMgr::init
        OPTICKS_KEY : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.699463ea0065185a7ffaf10d4935fc61
            CMDLINE :  /home/blyth/local/opticks/lib/OKTest --cvd 0 --envkey --xanalytic --timemax 400 --animtimemax 400 --target 352851 --eye -1,-1,-1
       OptiXVersion :           60000
    2019-05-10 17:30:18.825 INFO  [159840] [OpticksRun::setGensteps@137] genstep 1,6,4
    2019-05-10 17:30:18.825 FATAL [159840] [OKPropagator::propagate@70] OKPropagator::propagate(1) OK INTEROP DEVELOPMENT
    2019-05-10 17:30:18.825 INFO  [159840] [OpticksAim::target@146] OpticksAim::target SKIP as geometry target already set  352851
    2019-05-10 17:30:18.826 INFO  [159840] [OpticksViz::uploadEvent@351] OpticksViz::uploadEvent (1)
    2019-05-10 17:30:18.828 INFO  [159840] [OpticksViz::uploadEvent@358] OpticksViz::uploadEvent (1) DONE 
    2019-05-10 17:30:18.828 INFO  [159840] [OpEngine::uploadEvent@108] .
    2019-05-10 17:30:18.828 INFO  [159840] [OEvent::uploadGensteps@311] OEvent::uploadGensteps (INTEROP) SKIP OpenGL BufferId 69
    2019-05-10 17:30:18.828 INFO  [159840] [OpEngine::propagate@117] [
    2019-05-10 17:30:18.829 INFO  [159840] [OpSeeder::seedComputeSeedsFromInteropGensteps@63] OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER 
    CUDA error at /home/blyth/opticks/cudarap/CResource_.cu:43 code=101(cudaErrorInvalidDevice) "cudaGraphicsGLRegisterBuffer(&resource, buffer_id, flags)" 











