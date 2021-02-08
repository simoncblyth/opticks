UseOptiX7GeometryInstancedGASComp 
===================================

.. contents:: Table Of Contents


Overview
----------

This example is used as a playground for learning/investigating OptiX 7 techniques needed by Opticks.
Hopefully also some of the OptiX 7 interface classes will subsequenly become part of Opticks. 

Standalone Building
---------------------

Does not require Opticks, just depends on NVIDIA OptiX 7.0::

    export OPTIX_PREFIX=/usr/local/OptiX_700    # (might need to put this in .bashrc/.bash_profile)
    git clone https://bitbucket.org/simoncblyth/opticks
    cd opticks/examples/UseOptiX7GeometryInstancedGASComp
    ./go.sh # gets glm, builds, runs -> ppm image file    
     
Links
--------

* :doc:`../README`

* https://simoncblyth.bitbucket.io/env/presentation/lz_opticks_optix7.html 
* https://www.nvidia.com/en-us/gtc/on-demand/?search=OptiX
* https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21888-rtx-accelerated-raytracing-with-optix-7.pdf


Top Level
------------

UseOptiX7GeometryInstancedGASComp.cc
   main implemented with Geo and Engine

Geo
   High level holder of GAS and IAS 

Binding
   Types needed both on CPU and GPU 

Engine
   Holder of OptixDeviceContext context and PIP pipeline


Interface to OptiX 7 
----------------------

UseOptiX7GeometryInstancedGASComp.cu
   OptiX Programs

   * RG : RayGen
   * MS : Miss
   * IS : Intersection 
   * CH : ClosestHit 


PIP
   Pipeline and SBT (Shader Binding Table) construction 
AS
   Acceleration Structure, base struct for GAS and IAS
GAS
   Geometry Acceleration Structure 
IAS
   Instance Acceleration Structure 
   
GAS_Builder
   Builds GAS from bbox 

IAS_Builder
   Builds IAS from transforms and GAS indices


Utilities
------------

SPPM
   Writing PPM image files

sutil_Exception
   OPTIX_CHECK CUDA_CHECK macros 

sutil_vec_math
   simple vec math 

sutil_Preprocessor
   Defines used by sutil_vec_math.h 



diff ../UseOptiX7GeometryInstancedGASComp with UseOptiX7GeometryInstancedGASCompDyn
-----------------------------------------------------------------------------------------

Changing SBT record data from from static single radius float to dynamic values array.::

    [blyth@localhost UseOptiX7GeometryInstancedGASCompDyn]$ ./diff.sh 
    21c21
    < UseOptiX7GeometryInstancedGASCompDyn
    ---
    > UseOptiX7GeometryInstancedGASComp
    113c113
    <     const char* name = "UseOptiX7GeometryInstancedGASCompDyn" ; 
    ---
    >     const char* name = "UseOptiX7GeometryInstancedGASComp" ; 
    diff ../UseOptiX7GeometryInstancedGASComp/AS.h AS.h
    diff ../UseOptiX7GeometryInstancedGASComp/Binding.h Binding.h
    32c32
    <     float radius;
    ---
    >     float* values ;
    diff ../UseOptiX7GeometryInstancedGASComp/Engine.cc Engine.cc
    diff ../UseOptiX7GeometryInstancedGASComp/Engine.h Engine.h
    diff ../UseOptiX7GeometryInstancedGASComp/GAS_Builder.cc GAS_Builder.cc
    diff ../UseOptiX7GeometryInstancedGASComp/GAS_Builder.h GAS_Builder.h
    diff ../UseOptiX7GeometryInstancedGASComp/GAS.cc GAS.cc
    diff ../UseOptiX7GeometryInstancedGASComp/GAS.h GAS.h
    diff ../UseOptiX7GeometryInstancedGASComp/Geo.cc Geo.cc
    diff ../UseOptiX7GeometryInstancedGASComp/Geo.h Geo.h
    diff ../UseOptiX7GeometryInstancedGASComp/IAS_Builder.cc IAS_Builder.cc
    diff ../UseOptiX7GeometryInstancedGASComp/IAS_Builder.h IAS_Builder.h
    diff ../UseOptiX7GeometryInstancedGASComp/IAS.cc IAS.cc
    diff ../UseOptiX7GeometryInstancedGASComp/IAS.h IAS.h
    diff ../UseOptiX7GeometryInstancedGASComp/PIP.cc PIP.cc
    261a262,263
    >     // allocate CPU side records 
    > 
    269a272,273
    >     
    >     unsigned num_values = 1 ; 
    274c278,301
    <         (hg_sbt_ptr + i)->data.radius = gas.extent ;
    ---
    > 
    >         // -------- dynamic SBT values ---------------------------   
    >         float* values = new float[num_values]; 
    >         for(unsigned i=0 ; i < num_values ; i++) values[i] = 0.f ; 
    >         values[0] = gas.extent ;  
    > 
    >         float* d_values = nullptr ; 
    >         CUDA_CHECK( cudaMalloc(
    >                     reinterpret_cast<void**>( &d_values ),
    >                     num_values*sizeof(float)
    >                     ) );
    > 
    >         CUDA_CHECK( cudaMemcpy(
    >                     reinterpret_cast<void*>( d_values ),
    >                     values,
    >                     sizeof(float)*num_values,
    >                     cudaMemcpyHostToDevice
    >                     ) );
    > 
    >         delete [] values ; 
    >         // --------------------------------------------------------
    > 
    >         (hg_sbt_ptr + i)->data.values = d_values ;  
    >         // sets device pointer into CPU struct about to be copied to device
    277a305
    >     // copy CPU side records to GPU 
    285a314
    > 
    diff ../UseOptiX7GeometryInstancedGASComp/PIP.h PIP.h
    diff ../UseOptiX7GeometryInstancedGASComp/SPPM.h SPPM.h
    diff ../UseOptiX7GeometryInstancedGASComp/sutil_Exception.h sutil_Exception.h
    diff ../UseOptiX7GeometryInstancedGASComp/sutil_Preprocessor.h sutil_Preprocessor.h
    diff ../UseOptiX7GeometryInstancedGASComp/sutil_vec_math.h sutil_vec_math.h
    [blyth@localhost UseOptiX7GeometryInstancedGASCompDyn]$ 

