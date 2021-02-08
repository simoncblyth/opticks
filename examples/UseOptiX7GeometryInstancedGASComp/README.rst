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


