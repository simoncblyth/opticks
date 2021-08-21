QUDARap symbol_visibility_warnings
======================================


::

    ld: warning: direct access in function 'QBuf<float>::device_alloc(unsigned int)' from file 'CMakeFiles/QUDARap.dir/QBuf.cc.o' to global weak symbol 'typeinfo for sutil::QUDA_Exception' from file 'CMakeFiles/QUDARap.dir/QUDARap_generated_QSeed.cu.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'QBuf<float>::device_set(int)' from file 'CMakeFiles/QUDARap.dir/QBuf.cc.o' to global weak symbol 'typeinfo for sutil::QUDA_Exception' from file 'CMakeFiles/QUDARap.dir/QUDARap_generated_QSeed.cu.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'QBuf<float>::device_free()' from file 'CMakeFiles/QUDARap.dir/QBuf.cc.o' to global weak symbol 'typeinfo for sutil::QUDA_Exception' from file 'CMakeFiles/QUDARap.dir/QUDARap_generated_QSeed.cu.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'QBuf<float>::upload(float const*, unsigned int)' from file 'CMakeFiles/QUDARap.dir/QBuf.cc.o' to global weak symbol 'typeinfo for sutil::QUDA_Exception' from file 'CMakeFiles/QUDARap.dir/QUDARap_generated_QSeed.cu.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'QBuf<float>::download(std::__1::vector<float, std::__1::allocator<float> >&)' from file 'CMakeFiles/QUDARap.dir/QBuf.cc.o' to global weak symbol 'typeinfo for sutil::QUDA_Exception' from file 'CMakeFiles/QUDARap.dir/QUDARap_generated_QSeed.cu.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'QBuf<quad6>::device_alloc(unsigned int)' from file 'CMakeFiles/QUDARap.dir/QBuf.cc.o' to global weak symbol 'typeinfo for sutil::QUDA_Exception' from file 'CMakeFiles/QUDARap.dir/QUDARap_generated_QSeed.cu.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'QBuf<quad6>::device_set(int)' from file 'CMakeFiles/QUDARap.dir/QBuf.cc.o' to global weak symbol 'typeinfo for sutil::QUDA_Exception' from file 'CMakeFiles/QUDARap.dir/QUDARap_generated_QSeed.cu.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'QBuf<quad6>::device_free()' from file 'CMakeFiles/QUDARap.dir/QBuf.cc.o' to global weak symbol 'typeinfo for sutil::QUDA_Exception' from file 'CMakeFiles/QUDARap.dir/QUDARap_generated_QSeed.cu.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'QBuf<quad6>::upload(quad6 const*, unsigned int)' from file 'CMakeFiles/QUDARap.dir/QBuf.cc.o' to global weak symbol 'typeinfo for sutil::QUDA_Exception' from file 'CMakeFiles/QUDARap.dir/QUDARap_generated_QSeed.cu.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'QBuf<quad6>::download(std::__1::vector<quad6, std::__1::allocator<quad6> >&)' from file 'CMakeFiles/QUDARap.dir/QBuf.cc.o' to global weak symbol 'typeinfo for sutil::QUDA_Exception' from file 'CMakeFiles/QUDARap.dir/QUDARap_generated_QSeed.cu.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.


::

    ld: warning: direct access in function 'QBuf<float>::device_alloc(unsigned int)' from file 'CMakeFiles/QUDARap.dir/QBuf.cc.o' 
    to global weak symbol 
    'typeinfo for sutil::QUDA_Exception' 
    from file 
    'CMakeFiles/QUDARap.dir/QUDARap_generated_QSeed.cu.o' 
    means the weak symbol cannot be overridden at runtime. 
    This was likely caused by different translation units being compiled with different visibility settings.
 

* https://stackoverflow.com/questions/36567072/why-do-i-get-ld-warning-direct-access-in-main-to-global-weak-symbol-in-this


POssibly the solution is to get nvcc to use ``-fvisibility=hidden``


* https://gitlab.kitware.com/cmake/cmake/-/issues/17533

* see QUDARap/CMakeLists.txt failed to avoid the warnings 



::

    epsilon:issues blyth$ cmake --version
    cmake version 3.17.1

    CMake suite maintained and supported by Kitware (kitware.com/cmake).
    epsilon:issues blyth$ 

    O[blyth@localhost CSGOptiX]$ cmake --version
    cmake version 3.19.6

    CMake suite maintained and supported by Kitware (kitware.com/cmake).
    O[blyth@localhost CSGOptiX]$ 



Doing this in QUDARap/CMakeLists.txt made no difference::

    # see notes/issues/QUDARap_symbol_visibility_warnings.rst 
    #set_target_properties( ${name} PROPERTIES CXX_VISIBILITY_PRESET hidden )


Fix was to add QUDARAP_API to the QUAD_Exception class in QUDA_CHECK.h::

     42 class QUDARAP_API QUDA_Exception : public std::runtime_error
     43 {
     44  public:
     45      QUDA_Exception( const char* msg )
     46          : std::runtime_error( msg )
     47      { }
     48 
     49 };
     50 



