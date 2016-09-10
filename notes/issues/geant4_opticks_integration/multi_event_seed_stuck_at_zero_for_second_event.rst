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


* maybe use separate seed_buffer ??


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


