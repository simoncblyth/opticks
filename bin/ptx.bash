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



