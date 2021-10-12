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



