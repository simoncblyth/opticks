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


18:01 8 32-bit registers for payload and attribs


EOU
}


