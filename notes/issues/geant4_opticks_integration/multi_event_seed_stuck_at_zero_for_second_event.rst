Multi Event : Seed Stuck at Zero for 2nd Event
================================================

* `ggeoview-/tests/TrivialTest.cc`


Interop Mode
--------------

::

    OKTest 

    2016-09-13 12:55:19.887 INFO  [495555] [OEvent::upload@176] OEvent::upload (INTEROP) gensteps handed to OptiX by referencing OpenGL buffer id  
    2016-09-13 12:55:19.887 INFO  [495555] [OEvent::upload@180] OEvent::upload DONE
    2016-09-13 12:55:19.887 INFO  [495555] [OpSeeder::seedPhotonsFromGensteps@52] OpSeeder::seedPhotonsFromGensteps
    2016-09-13 12:55:19.887 INFO  [495555] [OpSeeder::seedPhotonsFromGenstepsViaOpenGL@67] OpSeeder::seedPhotonsFromGenstepsViaOpenGL
    iexpand  counts_size 1 output_size 100000
    2016-09-13 12:55:19.941 INFO  [495555] [OEvent::markDirtyPhotonBuffer@98] OEvent::markDirtyPhotonBuffer
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error 
            (Details: Function "RTresult _rtBufferMarkDirty(RTbuffer)" caught exception: 
             Must set or get buffer device pointer before calling rtBufferMarkDirty()., 
             file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/Objects/Buffer.cpp, line: 861)
    Abort trap: 6


Cannot markDirty in INTEROP mode, with photon buffer control

*  "OPTIX_INPUT_OUTPUT,INTEROP_PTR_FROM_OPENGL,BUFFER_COPY_ON_DIRTY"

OptiX is not aware that the ptr was grabbed from OpenGL (INTEROP_PTR_FROM_OPENGL) so cannot mark it dirty 
and get OptiX to sync.


WITH_SEED_BUFFER
~~~~~~~~~~~~~~~~~

Using seed buffer has some advantages

* photon buffer becomes "OPTIX_OUTPUT_ONLY,INTEROP_PTR_FROM_OPENGL"
* seed buffer "OPTIX_NON_INTEROP,OPTIX_INPUT_ONLY" (ie always pure compute buffer)


::

    simon:optickscore blyth$ opticks-find WITH_SEED_BUFFER
    ./optixrap/cu/generate.cu:#ifdef WITH_SEED_BUFFER
    ./optixrap/cu/generate.cu:#ifdef WITH_SEED_BUFFER
    ./optixrap/cu/generate.cu:    rtPrintf("(dumpseed WITH_SEED_BUFFER) genstep_id %u \n", genstep_id );
    ./optickscore/OpticksEvent.cc:#ifdef WITH_SEED_BUFFER
    ./opticksop/OpEngine.cc:#ifdef WITH_SEED_BUFFER
    ./opticksop/OpSeeder.cc:#ifdef WITH_SEED_BUFFER
    ./opticksop/OpSeeder.cc:#ifdef WITH_SEED_BUFFER
    ./optickscore/OpticksSwitches.h:#define WITH_SEED_BUFFER 1 
    simon:opticks blyth$ 
    simon:opticks blyth$ 
    simon:opticks blyth$ opticks-find OpticksSwitches.h
    ./optixrap/cu/generate.cu:#include "OpticksSwitches.h"
    ./optickscore/OpticksEvent.cc:#include "OpticksSwitches.h"
    ./opticksop/OpEngine.cc:#include "OpticksSwitches.h" 
    ./opticksop/OpSeeder.cc:#include "OpticksSwitches.h"  
    simon:opticks blyth$ 





Compute Mode
--------------

Up to 2::

   OKTest --cerenkov --trivial --save --compute --multievent 2
   TrivialTest --cerenkov --multievent 2   


* 1st evt passes
* 2nd evt fails with seed stuck at zero


Seed location in photon_buffer being zeroed in trivial, 
but that should be rewritten by the seeding of next event ?  

OpticksEvent buffer control::

* genstep : "OPTIX_INPUT_ONLY,UPLOAD_WITH_CUDA,BUFFER_COPY_ON_DIRTY"
* photon :  "OPTIX_INPUT_OUTPUT,PTR_FROM_OPENGL"




The seeds (genstep_id) stuck at zero for the 2nd evt::

    In [9]: ox.view(np.int32)
    Out[9]: 
    array([[[         0,          0,          0,          0],
            [1065353216, 1065353216, 1065353216, 1065353216],
            [         1,          1,         67,         80],  <<< copied ghead always that of genstep 0
            [         0,          0,          0,          0]],

           [[         0,          0,          0,          0],
            [1065353216, 1065353216, 1065353216, 1065353216],
            [         1,          1,         67,         80],
            [         1,          4,          0,          0]],   // photon_id photon_offset genstep_id genstep_offset

           [[         0,          0,          0,          0],
            [1065353216, 1065353216, 1065353216, 1065353216],
            [         1,          1,         67,         80],
            [         2,          8,          0,          0]],

           ..., 
           [[         0,          0,          0,          0],
            [1065353216, 1065353216, 1065353216, 1065353216],
            [         1,          1,         67,         80],
            [    612838,    2451352,          0,          0]],





DONE: Getting OptiX to notice CUDA/Thrust change to Photon buffer
----------------------------------------------------------------------

Seeding is done by CUDA/Thrust, seems OptiX is not noticing.

* try manual dirtying of the photon buffer to force resync   


Dirty CUDA Interop Buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
* https://devtalk.nvidia.com/default/topic/925622/optix/should-i-free-buffer-data-/
  
Detlef:
  
Another case would be CUDA interop buffers which use device side pointers where
the update happens through CUDA device code. Then you'd need to make the buffer
dirty manually to let OptiX know its contents have changed, to be able to
rebuild accelerations structures etc.


FIX: Using BUFFER_COPY_ON_DIRTY for photon buffer and manually marking dirty
------------------------------------------------------------------------------

*  https://bitbucket.org/simoncblyth/opticks/commits/e3d49ccd4dd6


Maybe Use Separate INPUT_ONLY SEED BUFFER ?
-----------------------------------------------

* THIS IS IMPLEMENTED BUT PRIOR TO TESTING FOUND THAT COULD
  GET MULTI EVENT TO WORK AS SHOWN ABOVE

::

    simon:opticksop blyth$ opticks-find WITH_SEED_BUF
    ./opticksop/OpSeeder.cc://#define WITH_SEED_BUF 1
    ./opticksop/OpSeeder.cc:#ifdef WITH_SEED_BUF
    ./opticksop/OpSeeder.cc:#ifdef WITH_SEED_BUF


TODO: Measure multievent compute speed using INPUT_OUTPUT seeded photon buffer vs INPUT only seed buffer
----------------------------------------------------------------------------------------------------------


