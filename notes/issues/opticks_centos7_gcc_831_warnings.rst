opticks_centos7_gcc_831_warnings
===================================


Following 


Most prolific is glm : TODO see if can fix by updating to 0.9.9.8 
--------------------------------------------------------------------

* https://github.com/g-truc/glm/releases

::

    epsilon:glm-0.9.9.5 blyth$ vi ./glm/glm/detail/setup.hpp

    GLM 0.9.9.5 released Apr 2, 2019 
    GLM 0.9.9.8 released Apr 13, 2020


List the warnings
-------------------


::

    S(){ ssh simon@P ; }

::

    [simon@localhost opticks]$ om-cleaninstall
    rm -rf /home/simon/local/opticks/build/okconf && mkdir -p /home/simon/local/opticks/build/okconf
    === om-visit-one : okconf          /home/simon/opticks/okconf                                   /home/simon/local/opticks/build/okconf                       
    === om-one-or-all cleaninstall : okconf          /home/simon/opticks/okconf                                   /home/simon/local/opticks/build/okconf                       
    -- The C compiler identification is GNU 8.3.1
    -- The CXX compiler identification is GNU 8.3.1
    -- Check for working C compiler: /opt/rh/devtoolset-8/root/usr/bin/cc
    -- Check for working C compiler: /opt/rh/devtoolset-8/root/usr/bin/cc -- works
    -- Detecting C compiler ABI info

    ...


cudarap FIXED::

    [  8%] Building NVCC (Device) object CMakeFiles/CUDARap.dir/CUDARap_generated_CDevice.cu.o
    [ 13%] Building NVCC (Device) object CMakeFiles/CUDARap.dir/CUDARap_generated_cuRANDWrapper_kernel.cu.o
    /home/simon/opticks/cudarap/CDevice.cu: In static member function ‘static void CDevice::Collect(std::vector<CDevice>&, bool)’:
    /home/simon/opticks/cudarap/CDevice.cu:71:25: warning: argument to ‘sizeof’ in ‘char* strncpy(char*, const char*, size_t)’ call is the same expression as the source; did you mean to use the size of the destination? [-Wsizeof-pointer-memaccess]
             strncpy( d.name, p.name, sizeof(p.name) );
                             ^~~~~~~~~~~~~~~
    Scanning dependencies of target CUDARap
    [ 17%] Building CXX object CMakeFiles/CUDARap.dir/CResource.cc.o
    [ 21%] Building CXX object CMakeFiles/CUDARap.dir/CUDARAP_LOG.cc.o

    

thrap loadsa glm warnings TODO try newer glm::

    [ 13%] Building NVCC (Device) object CMakeFiles/ThrustRap.dir/ThrustRap_generated_TRngBuf_.cu.o
    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(94): warning: __device__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(94): warning: __host__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(95): warning: __device__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(95): warning: __host__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(143): warning: __device__ annotation is ignored on a function("operator=") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(143): warning: __host__ annotation is ignored on a function("operator=") that is explicitly defaulted on its first declaration



optixrap a few of this old warning::

    /home/simon/opticks/optixrap/cu/compactionTest.cu(39): warning: variable "q3" was set but never used

    /home/blyth/local/opticks/externals/OptiX_650/include/optixu/optixpp_namespace.h(647): warning: overloaded virtual function "optix::APIObj::checkError" is only partially overridden in class "optix::ContextObj"

    /home/blyth/local/opticks/externals/OptiX_650/include/optixu/optixpp_namespace.h(647): warning: overloaded virtual function "optix::APIObj::checkError" is only partially overridden in class "optix::ContextObj"

    /home/blyth/local/opticks/externals/OptiX_650/include/optixu/optixpp_namespace.h(647): warning: overloaded virtual function "optix::APIObj::checkError" is only partially overridden in class "optix::ContextObj"

    /home/blyth/local/opticks/externals/OptiX_650/include/optixu/optixpp_namespace.h(647): warning: overloaded virtual function "optix::APIObj::checkError" is only partially overridden in class "optix::ContextObj"

    /home/blyth/local/opticks/externals/OptiX_650/include/optixu/optixpp_namespace.h(647): warning: overloaded virtual function "optix::APIObj::checkError" is only partially overridden in class "optix::ContextObj"

    /home/blyth/local/opticks/externals/OptiX_650/include/optixu/optixpp_namespace.h(647): warning: overloaded virtual function "optix::APIObj::checkError" is only partially overridden in class "optix::ContextObj"

    Scanning dependencies of target OptiXRap
    [ 17%] Building CXX object CMakeFiles/OptiXRap.di



okop again loadsa glm warnings::

    -- Build files have been written to: /home/simon/local/opticks/build/okop
    === om-make-one : okop            /home/simon/opticks/okop                                     /home/simon/local/opticks/build/okop                         
    [  4%] Building NVCC (Device) object CMakeFiles/OKOP.dir/OKOP_generated_OpIndexer_.cu.o
    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(94): warning: __device__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(94): warning: __host__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(95): warning: __device__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(95): warning: __host__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(143): warning: __device__ annotation is ignored on a function("operator=") that is explicitly defaulted on its first declaration



oglrap FIXED::

    [ 63%] Building CXX object CMakeFiles/OGLRap.dir/OpticksViz.cc.o
    [ 65%] Building CXX object CMakeFiles/OGLRap.dir/AxisApp.cc.o
    /home/simon/opticks/oglrap/Prog.cc: In member function ‘void Prog::traverseActive(Prog::Obj_t, bool)’:
    /home/simon/opticks/oglrap/Prog.cc:394:37: warning: ‘%i’ directive writing between 1 and 11 bytes into a region of size between 0 and 63 [-Wformat-overflow=]
                     sprintf (long_name, "%s[%i]", name, j);
                                         ^~~~~~~~
    /home/simon/opticks/oglrap/Prog.cc:394:25: note: ‘sprintf’ output between 4 and 77 bytes into a destination of size 64
                     sprintf (long_name, "%s[%i]", name, j);
                     ~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    [ 68%] Linking CXX shared library libOGLRap.so
    [ 68%] Built target OGLRap
    Scanning dependencies of target AxisAppCheck
    Scanning dependencies of target InteractorKeys


x4 FIXED::

    [ 43%] Building CXX object CMakeFiles/ExtG4.dir/X4GDMLMatrix.cc.o
    [ 44%] Building CXX object CMakeFiles/ExtG4.dir/X4_LOG.cc.o
    /home/simon/opticks/extg4/X4OpticalSurface.cc: In static member function ‘static GOpticalSurface* X4OpticalSurface::Convert(const G4OpticalSurface*)’:
    /home/simon/opticks/extg4/X4OpticalSurface.cc:87:10: warning: variable ‘specular’ set but not used [-Wunused-but-set-variable]
         bool specular = false ;    // HUH: not used, TODO:check cfg4
              ^~~~~~~~
    /home/simon/opticks/extg4/X4PhysicalVolume.cc: In member function ‘void X4PhysicalVolume::convertSolids_r(const G4VPhysicalVolume*, int)’:
    /home/simon/opticks/extg4/X4PhysicalVolume.cc:507:22: warning: comparison of integer expressions of different signedness: ‘int’ and ‘size_t’ {aka ‘long unsigned int’} [-Wsign-compare]
         for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
                        ~~^~~~~~~~~~~~~~~~~~~~~~
    /home/simon/opticks/extg4/X4PhysicalVolume.cc: In member function ‘GVolume* X4PhysicalVolume::convertStructure_r(const G4VPhysicalVolume*, GVolume*, int, const G4VPhysicalVolume*, bool&)’:
    /home/simon/opticks/extg4/X4PhysicalVolume.cc:889:23: warning: comparison of integer expressions of different signedness: ‘int’ and ‘size_t’ {aka ‘long unsigned int’} [-Wsign-compare]
          for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
                         ~~^~~~~~~~~~~~~~~~~~~~~~
    [ 45%] Linking CXX shared library libExtG4.so
    [ 45%] Built target ExtG4
    Scanning dependencies of target X4GDMLParserTest


cfg4 FIXED the last one::

    [ 36%] Building CXX object CMakeFiles/CFG4.dir/CWriter.cc.o
    /home/simon/opticks/cfg4/DsG4Scintillation.cc: In member function ‘virtual G4VParticleChange* DsG4Scintillation::PostStepDoIt(const G4Track&, const G4Step&)’:
    /home/simon/opticks/cfg4/DsG4Scintillation.cc:222:14: warning: variable ‘vertenergy’ set but not used [-Wunused-but-set-variable]
         G4double vertenergy=0.0;  // tis used : but on the other side of this monolith
                  ^~~~~~~~~~
    /home/simon/opticks/cfg4/DsG4Scintillation.cc:223:14: warning: variable ‘reem_d’ set but not used [-Wunused-but-set-variable]
         G4double reem_d=0.0;      // tis used : but on the other side of this monolith
                  ^~~~~~
    [ 36%] Building CXX object CMakeFiles/CFG4.dir/CDetector.cc.o
    [ 37%] Building CXX object CMakeFiles/CFG4.dir/CGDMLDetector.cc.o

    [ 50%] Building CXX object CMakeFiles/CFG4.dir/CPrimaryCollector.cc.o
    /home/simon/opticks/cfg4/CTraverser.cc: In member function ‘void CTraverser::AncestorTraverse(std::vector<const G4VPhysicalVolume*>, const G4VPhysicalVolume*, unsigned int, bool)’:
    /home/simon/opticks/cfg4/CTraverser.cc:210:22: warning: comparison of integer expressions of different signedness: ‘int’ and ‘size_t’ {aka ‘long unsigned int’} [-Wsign-compare]
          for (int i=0 ; i<lv->GetNoDaughters() ;i++) AncestorTraverse(ancestors, lv->GetDaughter(i), depth+1, recursive_select );
                         ~^~~~~~~~~~~~~~~~~~~~~
    [ 51%] Building CXX object CMakeFiles/CFG4.dir/CPhotonCollector.cc.o



g4ok::

    -- Looking for pthread_create in pthread - found
    -- Found Threads: TRUE  
    -- FindOpticksXercesC.cmake. Found Geant4::G4persistency AND XercesC::XercesC target _lll Geant4::G4geometry;Geant4::G4global;Geant4::G4graphics_reps;Geant4::G4intercoms;Geant4::G4materials;Geant4::G4particles;Geant4::G4digits_hits;Geant4::G4event;Geant4::G4processes;Geant4::G4run;Geant4::G4track;Geant4::G4tracking;XercesC::XercesC 
    CMake Warning (dev) at /usr/share/cmake3/Modules/FindCUDA.cmake:576 (option):
      Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
      --help-policy CMP0077" for policy details.  Use the cmake_policy command to
      set the policy and suppress this warning.

      For compatibility with older versions of CMake, option is clearing the
      normal variable 'CUDA_PROPAGATE_HOST_FLAGS'.
    Call Stack (most recent call first):
      /home/simon/opticks/cmake/Modules/FindOpticksCUDA.cmake:29 (find_package)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /home/simon/local/opticks/lib64/cmake/cudarap/cudarap-config.cmake:18 (find_dependency)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /home/simon/local/opticks/lib64/cmake/thrustrap/thrustrap-config.cmake:18 (find_dependency)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /home/simon/local/opticks/lib64/cmake/cfg4/cfg4-config.cmake:16 (find_dependency)
      CMakeLists.txt:11 (find_package)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    CMake Warning (dev) at /usr/share/cmake3/Modules/FindCUDA.cmake:582 (option):
      Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
      --help-policy CMP0077" for policy details.  Use the cmake_policy command to
      set the policy and suppress this warning.

      For compatibility with older versions of CMake, option is clearing the
      normal variable 'CUDA_VERBOSE_BUILD'.
    Call Stack (most recent call first):
      /home/simon/opticks/cmake/Modules/FindOpticksCUDA.cmake:29 (find_package)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /home/simon/local/opticks/lib64/cmake/cudarap/cudarap-config.cmake:18 (find_dependency)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /home/simon/local/opticks/lib64/cmake/thrustrap/thrustrap-config.cmake:18 (find_dependency)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /home/simon/local/opticks/lib64/cmake/cfg4/cfg4-config.cmake:16 (find_dependency)
      CMakeLists.txt:11 (find_package)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    -- Found CUDA: /usr/local/cuda-10.1 (found version "10.1") 
    -- Configuring G4OKTest



This has appeared before for details

* notes/issues/cmake-3.13.4-FindCUDA-warnings.rst
* :doc:`cmake-3.13.4-FindCUDA-warnings`


