##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

optix7-source(){   echo ${BASH_SOURCE} ; }
optix7-vi(){       vi $(optix7-source) ; }
optix7-env(){      olocal- ; }
optix7-usage(){ cat << \EOU

OptiX 7 : Brand New Lower Level API
======================================= 

* no multi-GPU, no mem mamagement : all that burden is shifted to application
  (Vulkanization of OptiX 6) 
* but its thread safe 

* Introduced Aug 2019 : OptiX 7 API is not backwards compatible


* https://www.nvidia.com/en-us/gtc/session-catalog/?search=OptiX
* https://developer.nvidia.com/gtc/2020/video/s21904

  GTC 2020: New Features in OptiX 7


* https://news.developer.nvidia.com/optix-7-delivers-new-levels-of-flexibility-to-application-developers/
* https://devtalk.nvidia.com/default/topic/1058310/optix/optix-7-0-release/
* https://devtalk.nvidia.com/default/topic/1061831/optix/optix-debugging-and-profiling-tools/
* https://devtalk.nvidia.com/default/topic/1058577/optix/porting-to-optix-7/

  Detlef : The OptiX 7 API is completely different and the host code effectively requires a rewrite


* https://raytracing-docs.nvidia.com/optix7/index.html
* https://raytracing-docs.nvidia.com/optix7/guide/index.html#introduction#


* https://gitlab.com/ingowald/optix7course
* ~/opticks_refs/SIG19_OptiX7_Main_Talk.pdf


* http://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.5.pdf

* ~/opticks_refs/unofficial_RayTracingGems_v1.5.pdf



NVIDIA Collective Communications Library (NCCL)
--------------------------------------------------

* https://developer.nvidia.com/nccl
* TODO: try this out using CuPy 

DASK
-----

* https://dask.org/


SIGGRAPH 2019 Videos
------------------------

* https://developer.nvidia.com/events/recordings/siggraph-2019

* https://www.nvidia.com/en-us/events/siggraph/schedule/

  * links to more talk videos

* http://www.realtimerendering.com/raytracing/roundup.html


Resources
-----------

* https://gitlab.com/ingowald/optix7course


Gems
--------

* The Iray Light Transport Simulation and Rendering System, A Keller et.al (on arxiv)


Steve Parker SIGGRAPH 2019 Video
----------------------------------

* https://devtalk.nvidia.com/default/topic/1062216/optix/optix-talks-from-siggraph-2019/

16:05
     Sounds like OptiX7 not "fork", its the future

     * explicit memory management (standard CUDA)
     * explicit build AS (accel structures)
     * explicit multi-GPU (up to application)
     * no state on host corresponding to scene

       * zero bytes on host
       * improves startup
       * reduce host resources

19:49
     RAY Tracing APIs : 
 
     * Microsoft DXR (DirectX 12) https://devblogs.microsoft.com/directx/announcing-microsoft-directx-raytracing/
     * NVIDIA VKRay (Vulkan) https://developer.nvidia.com/rtx/raytracing/vkray
     * fat OptiX 1-6
     * thin OptiX 7 


     * sustainable APIs 


20:34
     Buffers -> CUDA Pointers 

     Variables -> shader binding table

     Global scoped variables -> Launch parameters

         * enables overlapping of launches asynchronously 

     Semantic variables -> query functions to access internal state (current ray etc..)

     Amorphous programs -> Tagged program types 

     Geometry Group -> Geometry AS (primitives: triangles or programmable) 

     Group -> Instance AS

         * culling and masking availble on Instance basis

     Transform -> Just Input to Instance AS build


24:21
     OptiX 6 : still continuing to support 

26:55
    3D DAZ STUDIO : IRAY now built on OptiX7     
27:10
     People from NVIDIA worked to integrate OptiX7 into Blender



David Hart, "OptiX Performance Tools and Tricks" SIGGRAPH 2019
-----------------------------------------------------------------

* https://developer.nvidia.com/siggraph/2019/video/sig915-vid

12:25 BVH flatten scene

   Example : modelling two cars with wheels, 

   * should you instance the wheels and the cars or just copy the wheels into car instances 
   * its faster to keep it flatter, ie copy the wheels into two car instances 

   * Hardware Instancing : your first level is free!

     * ie it can handle one level of descent into a smaller BVH 

   * flatten if you can, ie if you have the memory  
   * better to avoid overlapped BVHs

   BVH : one level of instancing comes for free (ie RT cores handle it)


16:24 Reduce memory access

    Quadro RTX 6000 memory bandwidth : 672 GB/s
    *IF* 10 GRays/s is goal, then :

    672 GB/s /  10 GRays/s = 67.2 bytes/ray  (includes traversal)

    => Care must be taken to stay in this budget
    => way to calculate the upper bound of the ray tracer
  
    ::

       BUT : for me, I dont need to pull all the photons back to 
       the GPU, only interested in those that hit PMTs : so I think
       this memory bandwidth should not be a problem

18:01 8 32-bit registers for payload and attribs



Multi-GPU Scheduling 
-----------------------

* https://github.com/owensgroup/mgpuscheduler




GTC Silicon Valley 2019 ID:S9768 : New Features in OptiX 6.0 
----------------------------------------------------------------

* https://developer.nvidia.com/gtc/2019/video/S9768/video

20:17
    Performance with RT cores

                               1
    Amdahl's Law :   S =  -------------    
                           (1-p) + p/s

    For RT core speedup to be large, need to be traversal bound
    50:50 shade/traverse  (shade on SM, traverse on RT cores) 
    -> maximum factor 2 speedup

36:39
    rtContextSetUsageReportCallback

39:34
    BVH compaction : can save 1.5x-2x on memory for +10% build time
    on by default, can be switched off    



Build SDK
--------------

Observations

* glfw, glad and imgui are incorporated

::

    [blyth@localhost OptiX_700]$ mkdir SDK.build
    [blyth@localhost OptiX_700]$ cd SDK.build
    [blyth@localhost SDK.build]$ cmake ../SDK
    ... 
    -- Could NOT find OpenEXR (missing: OpenEXR_IlmImf_RELEASE OpenEXR_Half_RELEASE OpenEXR_Iex_RELEASE OpenEXR_Imath_RELEASE OpenEXR_IlmThread_RELEASE OpenEXR_INCLUDE_DIR) (found version "")
    CMake Warning at optixDemandTexture/CMakeLists.txt:62 (message):
      OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in
      optixDemandTexture.

    -- Found ZLIB: /usr/lib64/libz.so (found version "1.2.7") 
    -- Found ZlibStatic: /usr/lib64/libz.so (found version "1.2.7") 
    -- Could NOT find OpenEXR (missing: OpenEXR_IlmImf_RELEASE OpenEXR_Half_RELEASE OpenEXR_Iex_RELEASE OpenEXR_Imath_RELEASE OpenEXR_IlmThread_RELEASE OpenEXR_INCLUDE_DIR) (found version "")
    CMake Warning at optixDemandTextureAdvanced/CMakeLists.txt:62 (message):
      OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in
      optixDemandTextureAdvanced.

    -- Found OpenGL: /usr/lib64/libOpenGL.so   
    -- Could NOT find Vulkan (missing: VULKAN_LIBRARY VULKAN_INCLUDE_DIR) 


    [blyth@localhost SDK.build]$ make

    [ 18%] Building C object support/GLFW/src/CMakeFiles/glfw.dir/egl_context.c.o
    [ 20%] Linking C shared library ../../../lib/libglfw.so
    [ 20%] Built target glfw
    Scanning dependencies of target glad
    [ 21%] Building C object support/CMakeFiles/glad.dir/glad/glad.c.o
    [ 22%] Linking C shared library ../lib/libglad.so
    [ 22%] Built target glad
    Scanning dependencies of target imgui
    [ 23%] Building CXX object support/imgui/CMakeFiles/imgui.dir/imgui.cpp.o
    [ 25%] Building CXX object support/imgui/CMakeFiles/imgui.dir/imgui_demo.cpp.o
    ...
    [ 31%] Linking CXX static library ../../lib/libimgui.a
    [ 31%] Built target imgui
    ...


SDK 7 examples
--------------------

Some examples require to pick the display device only::

   Caught exception: GL interop is only available on display device, please use
   display device for optimal performance.  Alternatively you can disable GL
   interop with --no-gl-interop and run with degraded performance.

::

    CUDA_VISIBLE_DEVICES=1 ./optixMeshViewer    ## water bottle
    CUDA_VISIBLE_DEVICES=1 ./optixWhitted       ## bubble and checker board
    CUDA_VISIBLE_DEVICES=1 ./optixCutouts       ## Cornell box with cutouts in sphere and cube 
    CUDA_VISIBLE_DEVICES=1 ./optixSimpleMotionBlur  ## blue and red blurs
    CUDA_VISIBLE_DEVICES=1 ./optixPathTracer     ## Cornell box


    ./optixHello   ## green frame
    ./optixSphere  ## purple sphere
    ./optixSphere  ## blue/cyan/magenta triangle
    ./optixRaycasting      ## writes ppm of duck and translated duck
    ./optixDemandTexture           ## red black checkered sphere
    ./optixDemandTextureAdvanced   ## red black checkered sphere

    CUDA_VISIBLE_DEVICES=0 ./optixMultiGPU    ## Cornell box
    CUDA_VISIBLE_DEVICES=1 ./optixMultiGPU
    CUDA_VISIBLE_DEVICES=0,1 ./optixMultiGPU
    ## all work, but 0,1 has funny checker pattern on image 


CUcontext : is specific to a thread
-------------------------------------

* https://stackoverflow.com/questions/7534892/cuda-context-creation-and-resource-association-in-runtime-api-applications


optixProgramGroup PG and SBT
-------------------------------

* PG is just the bare program
* SBT collects records for each program that hold associated data 
  and reference to the program


github search for optix7
----------------------------

* https://github.com/search?q=optix7
* https://github.com/Hurleyworks/Optix7Sandbox

  mesh instancing example (gltf duck) 

* https://github.com/SpringPie/mov_ass3/tree/master/lighthouse2

* https://github.com/jbikker/lighthouse2/blob/master/lib/RenderCore_Optix7Filter/rendercore.cpp


CUSTOM_PRIMITIVES for analytic
------------------------------------

::

    [blyth@localhost SDK]$ find . -type f -exec grep -H CUSTOM_PRIMITIVES {} \;
    ./optixWhitted/optixWhitted.cpp:    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    ./optixSphere/optixSphere.cpp:            aabb_input.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    ./optixDemandTextureAdvanced/optixDemandTexture.cpp:    aabb_input.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    ./optixSimpleMotionBlur/optixSimpleMotionBlur.cpp:        sphere_input.type                     = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    ./optixCutouts/optixCutouts.cpp:        sphere_input.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    ./optixDemandTexture/optixDemandTexture.cpp:            aabb_input.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;





EOU
}


optix7-pdf(){ open $OPTICKS_INSTALL_PREFIX/externals/OptiX_700/doc/OptiX_Programming_Guide_7.0.0.pdf ; }
optix7-icd(){ cd $OPTICKS_INSTALL_PREFIX/externals/OptiX_700/include ; }
optix7-cd(){ cd $OPTICKS_INSTALL_PREFIX/externals/OptiX_700/SDK ; }





