OptiX_6_5_CUDA_10_2_incompatibility_possibly
=============================================

Overview
----------

CUDA nvcc link error recompile with fPIC


Raja Reports
--------------

::

    Hi Simon,

    I paste below, two warnings and finally an error with opticks-full which happens in optixrap (predictably I suppose). If you think it is a problem with the CUDA version, I will try to get cuda v10.1 installed for this. Back to debugging for me I suppose.
    Cheers,
    Raja.


    [ 39%] Linking CXX shared library libOptiXRap.so
    /usr/bin/ld: CMakeFiles/OptiXRap.dir/OptiXRap_generated_OBuf_.cu.o: relocation R_X86_64_32 against `.rodata' can not be used when making a shared object; recompile with -fPIC
    /usr/bin/ld: CMakeFiles/OptiXRap.dir/OptiXRap_generated_OBufBase_.cu.o: relocation R_X86_64_32 against symbol `_ZTV8OBufBase' can not be used when making a shared object; recompile with -fPIC
    /usr/bin/ld: CMakeFiles/OptiXRap.dir/OptiXRap_generated_OBufPair_.cu.o: relocation R_X86_64_32 against `.bss' can not be used when making a shared object; recompile with -fPIC
    /usr/bin/ld: final link failed: Nonrepresentable section on output
    collect2: error: ld returned 1 exit status
    make[2]: *** [libOptiXRap.so] Error 1
    make[1]: *** [CMakeFiles/OptiXRap.dir/all] Error 2
    make: *** [all] Error 2
    === om-one-or-all install : non-zero rc 2
    === om-all om-install : ERROR bdir --snip--/build/optixrap : non-zero rc 2
    === om-one-or-all install : non-zero rc 2


Tim Reports the same problem with CUDA 10.2
------------------------------------------------

* https://groups.io/g/opticks/topic/71679550#146


Searches
----------

* :google:`CUDA link error fPIC`

* https://forums.developer.nvidia.com/search?q=fPIC

* https://forums.developer.nvidia.com/t/static-cpu-library-build-failure-linker-requesting-fpic-again/29585

This unanswered question has some commandlines that could provide a starting point for low level debugging.


* https://gitlab.kitware.com/cmake/cmake/-/issues/18504

Some CMake version dependency in nvcc linking ?


* https://stackoverflow.com/questions/30642229/fail-to-build-shared-library-using-cmake-and-cuda


* https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=861878



