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

mgpu-source(){   echo ${BASH_SOURCE} ; }
mgpu-vi(){       vi $(mgpu-source) ; }
mgpu-usage(){ cat << EOU

MGPU : Modern GPU
======================

* https://moderngpu.github.io/intro.html
* https://github.com/moderngpu/moderngpu
* https://github.com/simoncblyth/moderngpu


* https://moderngpu.github.io/loadbalance.html



* http://on-demand.gputechconf.com/gtc/2014/video/S4674-parallel-decomposition-strategies-gpus.mp4

  Parallel Decomposition Strategies in Modern GPU
  Sean Baxter (NVIDIA)
  GTC Silicon Valley, 2014 

  Two stage:

  1. parallel partitioning : work assignment, no actual work 
  2. sequential logic : do the work 


  CTA : Cooperative Thread Array : aka a CUDA block (or an OpenCL workgroup) 
  ILP : Instruction Level Parallelism 
  TLP : Thread Level Parallelism across entire chip

  VT : values per thread "grain-size"
 
  * larger grains : more ILP
  * smaller grains : less state per thread -> more CTAs at once, more TLP  

  Treat the grain size as a static parameter, so can optimize with it 



Comparing GPU Parallel Primitives Libs
------------------------------------------


* https://ieeexplore.ieee.org/document/7447754

  Pro++: A Profiling Framework for Primitive-Based GPU Programming

::

    ArrayFire
    CUB
    CUDPP
    MGPU
    Thrust 



* https://docs.nvidia.com/cuda/cusolver/index.html





EOU
}


mgpu-edir(){ echo $(opticks-home)/numerics/moderngpu ; }
mgpu-sdir(){ echo $(local-base)/env/numerics/moderngpu ; }
mgpu-idir(){ echo $(local-base)/env/numerics/moderngpu/include ; }

mgpu-ecd(){  cd $(mgpu-edir) ; }
mgpu-scd(){  cd $(mgpu-sdir) ; }
mgpu-icd(){  cd $(mgpu-idir) ; }

mgpu-cd(){   cd $(mgpu-idir) ; }

mgpu-get()
{
    local dir=$(dirname $(mgpu-sdir)) &&  mkdir -p $dir && cd $dir
    [ ! -d moderngpu ] && git clone git@github.com:simoncblyth/moderngpu.git
}

mgpu-update()
{
    mgpu-scd
    git pull
}


mgpu-env(){      
   olocal- ; 
}

