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

ptx-source(){ echo $BASH_SOURCE ; }
ptx-vi(){ vi $(ptx-source)  ; }
ptx-env(){  olocal- ; opticks- ; }
ptx-usage(){ cat << EOU

PTX : low-level parallel thread execution virtual machine and instruction set architecture (ISA)
==================================================================================================

* https://en.wikipedia.org/wiki/Parallel_Thread_Execution

* https://docs.nvidia.com/cuda/pdf/ptx_isa_6.4.pdf

* https://www.cs.uaf.edu/2011/spring/cs641/lecture/03_03_CUDA_PTX.html


See also
----------

* ~/opticks/bin/ptx.py 
* notes/issues/oxrap-hunt-for-f64-in-ptx.rst


Manual
---------

p14 PTX statement is either

directive
     begins with . ends with ;   

instruction


Sources of .f64 in OptiX 6.0.0. PTX 
----------------------------------------

1. rtPrintExceptionDetails
2. rtPrintf of floats    
    ## aha : that explains why i see it in bounds at lot, I have a habit of leaving rtPrintf in bounds progs
    as they only get run one... 




EOU
}



