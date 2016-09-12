Multi Event : Seed Stuck at Zero for 2nd Event
================================================

* `ggeoview-/tests/TrivialTest.cc`


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
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
* https://devtalk.nvidia.com/default/topic/925622/optix/should-i-free-buffer-data-/
  
Detlef:
  
Another case would be CUDA interop buffers which use device side pointers where
the update happens through CUDA device code. Then you'd need to make the buffer
dirty manually to let OptiX know its contents have changed, to be able to
rebuild accelerations structures etc.


FIX: Using BUFFER_COPY_ON_DIRTY for photon buffer and manually marking dirty
------------------------------------------------------------------------------






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


