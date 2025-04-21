changing_COMPUTE_CAPABILITY in PTX .target
===============================================


Full rebuild needed to change target
-------------------------------------

Have observed that to change the compute capability requires nuclear rebuild::

   o
   om-clean
   om-conf
   oo

OR::
   
   o
   om-
   om-cleaninstall



Check it worked with::

    opticks-ptx-head



See :doc:`../../notes/issues/OPTIX_ERROR_INVALID_INPUT_with_optixModuleCreate_and_distributed_ptx`


 
What Compute Capability to target for the distributed PTX ? 
---------------------------------------------------------------

* https://forums.developer.nvidia.com/t/understanding-compute-capability/313577


dhart::

    th OptiX, if you’re compiling to PTX or OptiX-IR, you can use the compute
    capability for whatever the minimum GPU version you need to support is, and
    newer GPUs will work. For example, use 50 if you need Maxwell support, or 60
    for Pascal and beyond. This is detailed in the “Program Input” section of the
    “Pipeline” chapter in the OptiX Programming Guide: 



Note the following requirements for nvcc and nvrtc compilation:

The streaming multiprocessor (SM) target of the input OptiX program must be
less than or equal to the SM version of the GPU for which the module is
compiled.  To generate code for the minimum supported GPU (Maxwell), use
architecture targets for SM 5.0, for example, --gpu-architecture=compute_50.
Because OptiX rewrites the code internally, those targets will work on any
newer GPU as well.  CUDA Toolkits 10.2 and newer throw deprecation warnings for
SM 5.0 targets. These can be suppressed with the compiler option
-Wno-deprecated-gpu-targets.

If support for Maxwell GPUs is not required, you can use the next higher
GPU architecture target SM 6.0 (Pascal) to suppress these warnings.  Use
--machine=64 (-m64). Only 64-bit code is supported in OptiX.  Define the output
type with --optix-ir or --ptx. Do not compile to obj or cubin.

* https://raytracing-docs.nvidia.com/optix8/guide/index.html#program_pipeline_creation#program-input





