nvcc_fatal_the_version_of_the_host_compiler_is_not_supported
===================================================================

* hmm some conda install (probably root_numpy) has installed a unsupported clang 

::

    epsilon:optixrap blyth$ om
    === om-make-one : optixrap        /Users/blyth/opticks/optixrap                                /usr/local/opticks/build/optixrap                            
    [  2%] Building NVCC (Device) object CMakeFiles/OptiXRap.dir/OptiXRap_generated_OBuf_.cu.o
    [  2%] Building NVCC (Device) object CMakeFiles/OptiXRap.dir/OptiXRap_generated_OBufBase_.cu.o
    nvcc fatal   : The version ('90001') of the host compiler ('clang') is not supported
    nvcc fatal   : The version ('90001') of the host compiler ('clang') is not supported
    CMake Error at OptiXRap_generated_OBuf_.cu.o.Debug.cmake:216 (message):


Commenting the miniconda environment setup regains the build with the Xcode cc::

    epsilon:optixrap blyth$ om-install
    === om-visit-one : optixrap        /Users/blyth/opticks/optixrap                                /usr/local/opticks/build/optixrap                            
    === om-one-or-all install : optixrap        /Users/blyth/opticks/optixrap                                /usr/local/opticks/build/optixrap                            
    -- The C compiler identification is AppleClang 9.0.0.9000039
    -- The CXX compiler identification is AppleClang 9.0.0.9000039
    -- Check for working C compiler: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
    -- Check for working C compiler: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc - works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done




