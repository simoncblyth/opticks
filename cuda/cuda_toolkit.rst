CUDA Toolkit
==============

* https://developer.nvidia.com/cuda-toolkit
* http://graphics.im.ntu.edu.tw/~bossliaw/nvCuda_doxygen/html/index.html
* http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/index.html



Detect CUDA version
---------------------

The CUDA compiler version matches the toolkit version.

::

    aracity@aracity-desktop:~$ which nvcc
    /usr/local/cuda/bin/nvcc

::

    aracity@aracity-desktop:~$ nvcc --version
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2012 NVIDIA Corporation
    Built on Fri_Sep_21_17:28:58_PDT_2012
    Cuda compilation tools, release 5.0, V0.2.1221
    aracity@aracity-desktop:~$ 


Detect CUDA capability
------------------------

* http://stackoverflow.com/questions/12699455/can-i-get-cuda-compute-capability-version-in-compile-time-by-define

* :google:`__CUDA_ARCH__`

* http://stackoverflow.com/questions/8796369/cuda-and-nvcc-using-the-preprocessor-to-choose-between-float-or-double/8809924#8809924



