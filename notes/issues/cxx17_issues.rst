cxx17_issues
===============

After upping the standard from 14 to 17 it is necessary to cleaninstall all packages. 
If you forgot to do that in a package::

    om
    ...
    === om-make-one : CSG_GGeo        /home/blyth/opticks/CSG_GGeo                                 /data/blyth/junotop/ExternalLibs/opticks/head/build/CSG_GGeo 
    -- Configuring CSG_GGeo
    -- CSG_FOUND      : 1      CSG_INCLUDE_DIRS      : 
    -- GGeo_FOUND     : 1     GGeo_INCLUDE_DIRS     : 
    -- Configuring done
    CMake Error in CMakeLists.txt:
      Target "CSG_GGeo" requires the language dialect "CXX17" (with compiler
      extensions), but CMake does not know the compile flags to use to enable it.


    -- Generating done
    CMake Generate step failed.  Build files cannot be regenerated correctly.


Check versions shows the expected ones::

    O[blyth@localhost CSG_GGeo]$ gcc --version
    gcc (GCC) 8.3.1 20190311 (Red Hat 8.3.1-3)
    Copyright (C) 2018 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    O[blyth@localhost CSG_GGeo]$ cmake --version
    cmake version 3.19.6

    CMake suite maintained and supported by Kitware (kitware.com/cmake).


So cleaninstall::

    O[blyth@localhost CSG_GGeo]$ om-cleaninstall
    rm -rf /data/blyth/junotop/ExternalLibs/opticks/head/build/CSG_GGeo && mkdir -p /data/blyth/junotop/ExternalLibs/opticks/head/build/CSG_GGeo
    === om-visit-one : CSG_GGeo        /home/blyth/opticks/CSG_GGeo                                 /data/blyth/junotop/ExternalLibs/opticks/head/build/CSG_GGeo 
    === om-one-or-all cleaninstall : CSG_GGeo        /home/blyth/opticks/CSG_GGeo                                 /data/blyth/junotop/ExternalLibs/opticks/head/build/CSG_GGeo 
    -- The C compiler identification is GNU 8.3.1
    -- The CXX compiler identification is GNU 8.3.1
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Check for working C compiler: /opt/rh/devtoolset-8/root/usr/bin/cc - skipped
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Check for working CXX compiler: /opt/rh/devtoolset-8/root/usr/bin/c++ - skipped
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Configuring CSG_GGeo
    -- Looking for pthread.h
    -- Looking for pthread.h - found
    -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
    -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
    -- Looking for pthread_create in pthreads
    -- Looking for pthread_create in pthreads - not found
    -- Looking for pthread_create in pthread
    -- Looking for pthread_create in pthread - found
    -- Found Threads: TRUE  
    -- Found CUDA: /usr/local/cuda (found version "10.1") 
    -- CSG_FOUND      : 1      CSG_INCLUDE_DIRS      : 
    -- GGeo_FOUND     : 1     GGeo_INCLUDE_DIRS     : 
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /data/blyth/junotop/ExternalLibs/opticks/head/build/CSG_GGeo
    === om-make-one : CSG_GGeo        /home/blyth/opticks/CSG_GGeo                                 /data/blyth/junotop/ExternalLibs/opticks/head/build/CSG_GGeo 
    Scanning dependencies of target CSG_GGeo
    [ 33%] Building CXX object CMakeFiles/CSG_GGeo.dir/CSG_GGeo.cc.o
    [ 66%] Building CXX object CMakeFiles/CSG_GGeo.dir/CSG_GGeo_Convert.cc.o
    [100%] Linking CXX executable CSG_GGeo
    [100%] Built target CSG_GGeo
    [100%] Built target CSG_GGeo
    Install the project...
    -- Install configuration: "Debug"
    -- Installing: /data/blyth/junotop/ExternalLibs/opticks/head/lib/CSG_GGeo
    -- Set runtime path of "/data/blyth/junotop/ExternalLibs/opticks/head/lib/CSG_GGeo" to "$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64"
    O[blyth@localhost CSG_GGeo]$ 




Bizarre : error: cannot call member function ‘void QBuf<T>::device_alloc(unsigned int) [with T = int]’ without object
----------------------------------------------------------------------------------------------------------------------------

::

     67     static QBuf<T>* Upload( const T* data, unsigned num_items )
     68     {   
     69         QBuf<T>* buf = new QBuf<T>() ;
     //                                   ^^^   had omitted the ctor brackets 
     70         buf->device_alloc(num_items);
     71         buf->upload( data, num_items );
     72         return buf ;
     73     }   
     74 




::

    /home/blyth/opticks/qudarap/QBuf.hh: In instantiation of ‘static QBuf<T>* QBuf<T>::Alloc(unsigned int) [with T = int]’:
    /home/blyth/opticks/qudarap/QSeed.cu:36:22:   required from here
    /home/blyth/opticks/qudarap/QBuf.hh:90:1: error: cannot call member function ‘void QBuf<T>::device_alloc(unsigned int) [with T = int]’ without object
             buf->device_alloc(num_items);
     ^       ~~~~
    /home/blyth/opticks/qudarap/QBuf.hh:91:1: error: cannot call member function ‘void QBuf<T>::device_set(int) [with T = int]’ without object
             buf->device_set(0);
     ^       ~~
    CMake Error at QUDARap_generated_QSeed.cu.o.Debug.cmake:276 (message):
      Error generating file
      /data/blyth/junotop/ExternalLibs/opticks/head/build/qudarap/CMakeFiles/QUDARap.dir//./QUDARap_generated_QSeed.cu.o



Without CUDA/nvcc in the mix it compiles fine, eg with QBufTest
------------------------------------------------------------------


* https://stackoverflow.com/questions/36551469/triggering-c11-support-in-nvcc-with-cmake

* https://github.com/microsoft/onnxruntime/issues/661

* https://forums.developer.nvidia.com/t/cuda-10-1-nvidia-youre-now-fixing-gcc-bugs-that-gcc-doesnt-even-have/71063




::

    [ 28%] Building CXX object CMakeFiles/QUDARap.dir/QSeed.cc.o
    In file included from /home/blyth/opticks/qudarap/QRng.cc:9:
    /home/blyth/opticks/qudarap/QRng.cc: In destructor ‘virtual QRng::~QRng()’:
    /home/blyth/opticks/qudarap/QUDA_CHECK.h:19:52: warning: throw will always call terminate() [-Wterminate]
                 throw QUDA_Exception( ss.str().c_str() );                        \
                                                        ^
    /home/blyth/opticks/qudarap/QRng.cc:32:5: note: in expansion of macro ‘QUDA_CHECK’
         QUDA_CHECK(cudaFree(qr->rng_states));
         ^~~~~~~~~~
    /home/blyth/opticks/qudarap/QUDA_CHECK.h:19:52: note: in C++11 destructors default to noexcept
                 throw QUDA_Exception( ss.str().c_str() );                        \
                                                        ^
    /home/blyth/opticks/qudarap/QRng.cc:32:5: note: in expansion of macro ‘QUDA_CHECK’
         QUDA_CHECK(cudaFree(qr->rng_states));
         ^~~~~~~~~~
    In file included from /home/blyth/opticks/qudarap/QProp.cc:7:



Alarming that the below compilation fails with normal pointer arrow style::

    088     /**
     89     method tickles CUDA/cxx17/devtoolset-8 bug causing compilation to fail with 
     90     error: cannot call member function without object
     91 
     92     See notes/issues/cxx17_issues.rst
     93 
     94     https://forums.developer.nvidia.com/t/cuda-10-1-nvidia-youre-now-fixing-gcc-bugs-that-gcc-doesnt-even-have/71063
     95 
     96     **/
     97 
     98     static QBuf<T>* Alloc( unsigned num_items  )
     99     {   
    100         QBuf<T>* buf = new QBuf<T> ; 
    101         (*buf).device_alloc(num_items); 
    102         (*buf).device_set(0); 
    103         return buf ; 
    104     }   


cxx17 throwing up new templated undefined errors:: 


    [ 76%] Built target QTexRotateTest
    [ 77%] Built target QSeedTest
    CMakeFiles/QTexMakerTest.dir/QTexMakerTest.cc.o: In function `main':
    /home/blyth/opticks/qudarap/tests/QTexMakerTest.cc:37: undefined reference to `QTex<float4>::setHDFactor(unsigned int)'
    /home/blyth/opticks/qudarap/tests/QTexMakerTest.cc:38: undefined reference to `QTex<float4>::uploadMeta()'
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/QTexMakerTest] Error 1
    make[1]: *** [tests/CMakeFiles/QTexMakerTest.dir/all] Error 2
    make[1]: *** Waiting for unfinished jobs....
    [ 80%] Built target QCerenkovTest
    [ 80%] Built target QBufTest
    [ 82%] Built target QPropTest




Try with CUDA 11 on Precision
---------------------------------

* Hmh, defer that updating to 11.4 will require a new driver, I dont like doing that remotely 

::

    O[blyth@localhost qudarap]$ l /usr/local/
    total 2261216
         0 drwxr-xr-x.  7 blyth blyth        75 May  2 02:05 csg
         4 drwxr-xr-x. 18 root  root       4096 Apr 30 18:03 .
     44500 -rw-r--r--.  1 blyth blyth  45564234 Feb  6  2021 NVIDIA-OptiX-SDK-7.1.0-linux64-x86_64.sh
     43532 -rw-r--r--.  1 blyth blyth  44573802 Feb  6  2021 NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64.sh
    141004 -rw-rw-r--.  1 blyth blyth 144387574 Sep 19  2019 NVIDIA-Linux-x86_64-435.21.run
     28256 -rw-rw-r--.  1 blyth blyth  28930132 Sep 10  2019 NVIDIA-OptiX-SDK-7.0.0-linux64.sh
    124648 -rw-r--r--.  1 blyth blyth 127636318 Sep 10  2019 NVIDIA-OptiX-SDK-6.5.0-linux64.sh
         0 lrwxrwxrwx.  1 root  root         30 May 22  2019 OptiX_511 -> NVIDIA-OptiX-SDK-5.1.1-linux64
         0 drwxr-xr-x.  7 root  root         87 May 22  2019 NVIDIA-OptiX-SDK-5.1.1-linux64
    620024 -rw-rw-r--.  1 blyth blyth 634901022 May  9  2019 NVIDIA-OptiX-SDK-5.1.1-linux64-25109142.sh
         0 drwxr-xr-x.  9 root  root        123 Apr 10  2019 NVIDIA-OptiX-SDK-6.0.0-linux64
         0 lrwxrwxrwx.  1 root  root         30 Apr  9  2019 OptiX_600 -> NVIDIA-OptiX-SDK-6.0.0-linux64
    627268 -rw-rw-r--.  1 blyth blyth 642319364 Apr  9  2019 NVIDIA-OptiX-SDK-6.0.0-linux64-25650775.sh
         0 lrwxrwxrwx.  1 root  root         30 Jul  6  2018 OptiX_510 -> NVIDIA-OptiX-SDK-5.1.0-linux64
         0 drwxr-xr-x.  7 root  root         87 Jul  6  2018 NVIDIA-OptiX-SDK-5.1.0-linux64
    631976 -rw-r--r--.  1 root  root  647141137 Jul  5  2018 NVIDIA-OptiX-SDK-5.1.0-linux64_24109458.sh
         0 drwxr-xr-x. 13 root  root        155 Jul  5  2018 ..



         0 drwxr-xr-x. 18 root  root        249 Jul  5  2018 cuda-9.2
         4 drwxr-xr-x. 18 root  root       4096 Apr  8  2019 cuda-10.1
         0 lrwxrwxrwx.  1 root  root         21 Apr  8  2019 cuda -> /usr/local/cuda-10.1/



Confirm the issue is coming from cxx17/devtoolset-8 by downgrading
----------------------------------------------------------------------

cmake/Modules/OpticksCXXFlags.cmake switch back from 17 to 14 on Linux::

     68 else()
     69 
     70   if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
     71      set(CMAKE_CXX_STANDARD 14)
     72      set(CMAKE_CXX_STANDARD_REQUIRED on)
     73   else ()
     74      set(CMAKE_CXX_STANDARD 14)
     75      #set(CMAKE_CXX_STANDARD 17)   ## Geant4 1100 forces c++17 gcc 5+ devtoolset-8 on centos7 : dangerous for CUDA 
     76      set(CMAKE_CXX_STANDARD_REQUIRED on)
     77   endif ()



.local.bash comment out devtoolset-8::

     25 # default gcc is 4.8.5 
     26 #source /opt/rh/devtoolset-9/enable    ## gcc 9.3.1 : cannot be used with CUDA 10.1
     27 #source /opt/rh/devtoolset-8/enable    ## gcc 8.3.1 
     28 #source /opt/rh/devtoolset-7/enable    ## gcc 7.3.1 
     29 


start new session and check gcc version is back to 4.8.5::

    epsilon:tests blyth$ O
    mo .bashrc VIP_MODE:dev O : ordinary opticks dev ontop of juno externals CMTEXTRATAGS:opticks

    O[blyth@localhost ~]$ gcc --version
    gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-44)
    Copyright (C) 2015 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    O[blyth@localhost ~]$ 

Cleaninstall::

    O[blyth@localhost opticks]$ om- ; om-cleaninstall



Hmm even back with cxx14 getting problems from QTex : maybe the template specializations ?
Try avoiding the complexity by moving rotation to different struct.


