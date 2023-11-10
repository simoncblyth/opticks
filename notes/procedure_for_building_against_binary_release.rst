procedure_for_building_against_binary_release
===============================================

Overview
-----------

Developing a CMakeLists.txt to use Opticks packages 
based off a binary release. 

For the finished example see : examples/UseRelease


Issue 1 : caused by omitting "-DOPTICKS_PREFIX=$OPTICKS_PREFIX" 
-----------------------------------------------------------------

::

    -- Detecting CXX compile features - done
    CMake Error at /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package):
      Found package configuration file:

        /data/simon/local/opticks/lib64/cmake/sysrap/sysrap-config.cmake

      but it set SysRap_FOUND to FALSE so package "SysRap" is considered to be
      NOT FOUND.  Reason given by package:

      SysRap could not be found because dependency PLog could not be found.

    Call Stack (most recent call first):
      /data/simon/local/opticks/lib64/cmake/u4/u4-config.cmake:10 (find_dependency)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /data/simon/local/opticks/lib64/cmake/g4cx/g4cx-config.cmake:8 (find_dependency)
      CMakeLists.txt:12 (find_package)


Issue 2 : "set_target_properties called with incorrect number of arguments"  due to omission of "include(GNUInstallDirs)"
---------------------------------------------------------------------------------------------------------------------------

Somehow the below two are no longer defined::

   CMAKE_INSTALL_LIBDIR
   CMAKE_INSTALL_INCLUDEDIR 

Causing the error, from target properties of all packages::

  2 set_target_properties(Opticks::QUDARap PROPERTIES INTERFACE_PKG_CONFIG_NAME QUDARap)
  3 
  4 set_target_properties(Opticks::QUDARap PROPERTIES INTERFACE_INSTALL_CONFIGFILE_BCM ${CMAKE_CURRENT_LIST_FILE})
  5 set_target_properties(Opticks::QUDARap PROPERTIES INTERFACE_INSTALL_CONFIGDIR_BCM ${CMAKE_CURRENT_LIST_DIR})
  6 set_target_properties(Opticks::QUDARap PROPERTIES INTERFACE_INSTALL_LIBDIR_BCM ${CMAKE_INSTALL_LIBDIR})
  7 set_target_properties(Opticks::QUDARap PROPERTIES INTERFACE_INSTALL_INCLUDEDIR_BCM ${CMAKE_INSTALL_INCLUDEDIR})
  8 set_target_properties(Opticks::QUDARap PROPERTIES INTERFACE_INSTALL_PREFIX_BCM ${CMAKE_INSTALL_PREFIX})

* those variables come from https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html

So that means need to include the below early in CMakeLists.txt and customize the INCLUDEDIR::

    include(GNUInstallDirs)
    set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")  # override the GNUInstallDirs default of "include"


Issue 3 : Need to use the same C++ dialect that Opticks does
---------------------------------------------------------------

The CMake default C++ dialect is an old one::

    In file included from /Users/blyth/opticks/examples/UseRelease/UseRelease.cc:2:
    /usr/local/opticks/include/G4CX/G4CXOpticks.hh:40:12: error: unknown type name 'constexpr'
        static constexpr const char* SaveGeometry_KEY = "G4CXOpticks__SaveGeometry_DIR" ; 
               ^
    /usr/local/opticks/include/G4CX/G4CXOpticks.hh:40:22: error: expected member name or ';' after declaration specifiers
        static constexpr const char* SaveGeometry_KEY = "G4CXOpticks__SaveGeometry_DIR" ; 
        ~~~~~~~~~~~~~~~~ ^

Add::

   set(CMAKE_CXX_STANDARD 17)   ## Geant4 1100+ (and Opticks) forcing c++17 which restricts to gcc 5+
   set(CMAKE_CXX_STANDARD_REQUIRED on) 


Issue 4 : HMM this one could be Darwin specific that uses very old CUDA and OptiX 5.5 
----------------------------------------------------------------------------------------

OptiX 7+ has no lib, just headers::


    -- Installing: /data/blyth/opticks/UseRelease.install/lib/UseRelease
    === ./build.sh : /data/blyth/opticks/UseRelease.install/lib/UseRelease
    dyld: Library not loaded: @rpath/liboptix.1.dylib
      Referenced from: /data/blyth/opticks/UseRelease.install/lib/UseRelease
      Reason: image not found
    ./build.sh: line 39: 79738 Abort trap: 6           $bin
    epsilon:UseRelease blyth$ 






