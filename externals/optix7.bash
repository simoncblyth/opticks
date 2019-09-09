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

* https://devtalk.nvidia.com/default/topic/1058310/optix/optix-7-0-release/
* https://devtalk.nvidia.com/default/topic/1061831/optix/optix-debugging-and-profiling-tools/

* https://raytracing-docs.nvidia.com/optix7/index.html
* https://raytracing-docs.nvidia.com/optix7/guide/index.html#introduction#


* https://gitlab.com/ingowald/optix7course
* ~/opticks_refs/SIG19_OptiX7_Main_Talk.pdf


* http://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.5.pdf

* ~/opticks_refs/unofficial_RayTracingGems_v1.5.pdf


EOU
}


