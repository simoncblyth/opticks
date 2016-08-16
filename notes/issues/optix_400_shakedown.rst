OptiX 400 Shakedown
===========================

Following the migration of buffer and textures to work with OptiX 400 
notice several peculiarities in photon propagation.


.. toctree::

    optix_400_seedPhotonsFromGenstepsImp_MISMATCH


GGeoViewTest : Slow Photons
-----------------------------

Notice some rafts of parallel slowly propagating photons.
Looking at photons in different history sequences suggests 
those ending in AB (bulk absorb) are primary mis-behavers.

tpmt-- : origin attraction and swarming
------------------------------------------

Small numbers of slower photons seem attracted to origin
and clusters form !!


op --cerenkov
------------------

10 percent of material sequence selection with NULL label, 
and slow backwards photons.


tpmt


compute mode dirty buffer
---------------------------

::

    GGeoViewTest --compute

    2016-08-16 13:55:17.440 INFO  [3257228] [OConfig::configureSampler@377] OPropertyLib::configureSampler
    2016-08-16 13:55:17.455 INFO  [3257228] [OEngineImp::prepareOptiX@156] OEngineImp::prepareOptiX DONE
    2016-08-16 13:55:17.455 INFO  [3257228] [OpEngine::prepareOptiX@87] OpEngine::prepareOptiX DONE
    2016-08-16 13:55:17.455 INFO  [3257228] [OpEngine::preparePropagator@92] OpEngine::preparePropagator START 
    2016-08-16 13:55:17.455 INFO  [3257228] [OEngineImp::preparePropagator@175] OEngineImp::preparePropagatorNORMAL override_ -1
    2016-08-16 13:55:17.468 INFO  [3257228] [OPropagator::initRng@163] OPropagator::initRng rng_max 3000000 num_photons 100000 rngCacheDir /usr/local/opticks/installcache/RNG
    cuRANDWrapper::instanciate with cache enabled : cachedir /usr/local/opticks/installcache/RNG
    cuRANDWrapper::fillHostBuffer
    cuRANDWrapper::LoadIntoHostBuffer
    cuRANDWrapper::LoadIntoHostBuffer : loading from cache /usr/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin 
    cuRANDWrapper::LoadIntoHostBuffer 3000000 items from /usr/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin load_digest 82f3f46e78f078d98848d3e07ee1a654 
    2016-08-16 13:55:18.785 INFO  [3257228] [OPropagator::initEvent@209] OPropagator::initEvent
    2016-08-16 13:55:18.785 INFO  [3257228] [OContext::createBuffer@325] OContext::createBuffer             gensteps               1,6,4 mode : COMPUTE  BufferControl : OPTIX_SETSIZE OPTIX_INPUT_ONLY 
    2016-08-16 13:55:18.785 INFO  [3257228] [OContext::configureBuffer@396]   gensteps               1,6,4 QUAD size (gnq)          6
    2016-08-16 13:55:18.788 INFO  [3257228] [OPropagator::initEvent@220] OPropagator::initGenerate (COMPUTE) uploading gensteps 
    2016-08-16 13:55:18.788 INFO  [3257228] [OContext::upload@287] OContext::upload numBytes 96upload (1,6,4)  NumBytes(0) 96 NumBytes(1) 96 NumValues(0) 24 NumValues(1) 24{}
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error 
        (Details: Function "RTresult _rtBufferMarkDirty(RTbuffer)" caught exception:
          Mark dirty only allowed on buffers created with RT_BUFFER_COPY_ON_DIRTY, 
         file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/Objects/Buffer.cpp, line: 867)
    Process 91414 stopped
    ...
        frame #8: 0x0000000103b29d78 libOptiXRap.dylib`optix::APIObj::checkError(this=0x00000001115f3550, code=RT_ERROR_UNKNOWN) const + 184 at optixpp_namespace.h:1804
        frame #9: 0x0000000103b3d8c7 libOptiXRap.dylib`optix::BufferObj::markDirty(this=0x00000001115f3550) + 55 at optixpp_namespace.h:3620
        frame #10: 0x0000000103b3d836 libOptiXRap.dylib`void OContext::upload<float>(buffer=0x00000001115f2998, npy=0x0000000109021230) + 550 at OContext.cc:297
        frame #11: 0x0000000103b4e100 libOptiXRap.dylib`OPropagator::initEvent(this=0x00000001115f2960, evt=0x0000000106805bb0) + 1072 at OPropagator.cc:221
        frame #12: 0x0000000103b4dcc7 libOptiXRap.dylib`OPropagator::initEvent(this=0x00000001115f2960) + 55 at OPropagator.cc:199
        frame #13: 0x0000000103b4c055 libOptiXRap.dylib`OEngineImp::preparePropagator(this=0x00000001090222e0) + 917 at OEngineImp.cc:206
        frame #14: 0x000000010402c377 libOpticksOp.dylib`OpEngine::preparePropagator(this=0x0000000106806140) + 247 at OpEngine.cc:93
        frame #15: 0x000000010412033a libGGeoView.dylib`App::preparePropagator(this=0x00007fff5fbfed18) + 58 at App.cc:1019
        frame #16: 0x000000010000ae52 GGeoViewTest`main(argc=2, argv=0x00007fff5fbfeec0) + 1794 at GGeoViewTest.cc:116
        frame #17: 0x00007fff8652f5fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 11
    frame #11: 0x0000000103b4e100 libOptiXRap.dylib`OPropagator::initEvent(this=0x00000001115f2960, evt=0x0000000106805bb0) + 1072 at OPropagator.cc:221
       218      if(m_ocontext->isCompute()) 
       219      {
       220          LOG(info) << "OPropagator::initGenerate (COMPUTE)" << " uploading gensteps " ;
    -> 221          OContext::upload<float>(m_genstep_buffer, gensteps);
       222      }
       223      else if(m_ocontext->isInterop())

    (lldb) f 10
    frame #10: 0x0000000103b3d836 libOptiXRap.dylib`void OContext::upload<float>(buffer=0x00000001115f2998, npy=0x0000000109021230) + 550 at OContext.cc:297
       294      void* d_ptr = NULL;
       295      rtBufferGetDevicePointer(buffer->get(), 0, &d_ptr);
       296      cudaMemcpy(d_ptr, npy->getBytes(), numBytes, cudaMemcpyHostToDevice);
    -> 297      buffer->markDirty();
       298  }



