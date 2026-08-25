Geant4_1140_compatibility_fixes
===============================


InsideNoVoxels now private
---------------------------

::

    [ 17%] Building CXX object CMakeFiles/U4.dir/ShimG4OpRayleigh.cc.o
    [ 19%] Building CXX object CMakeFiles/U4.dir/Local_DsG4Scintillation.cc.o
    [ 19%] Building CXX object CMakeFiles/U4.dir/U4Physics.cc.o
    In file included from /home/blyth/opticks/u4/U4Recorder.cc:30:
    /data1/blyth/local/opticks_Debug_g411/include/SysRap/ssolid.h: In static member function ‘static G4double ssolid::DistanceMultiUnionNoVoxels_(const G4MultiUnion*, const G4ThreeVector&, const G4ThreeVector&, EInside&)’:
    /data1/blyth/local/opticks_Debug_g411/include/SysRap/ssolid.h:91:32: error: ‘EInside G4MultiUnion::InsideNoVoxels(const G4ThreeVector&) const’ is private within this context
       91 |     in =  solid->InsideNoVoxels(pos) ;
          |           ~~~~~~~~~~~~~~~~~~~~~^~~~~
    In file included from /data1/blyth/local/opticks_Debug_g411/include/SysRap/ssolid.h:18:
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/include/Geant4/G4MultiUnion.hh:268:13: note: declared private here
      268 |     EInside InsideNoVoxels(const G4ThreeVector& aPoint) const;
          |             ^~~~~~~~~~~~~~
    make[2]: *** [CMakeFiles/U4.dir/build.make:160: CMakeFiles/U4.dir/U4Recorder.cc.o] Error 1
    make[2]: *** Waiting for unfinished jobs....
    make[1]: *** [CMakeFiles/Makefile2:934: CMakeFiles/U4.dir/all] Error 2
    make: *** [Makefile:146: all] Error 2
    === om-one-or-all cleaninstall : non-zero rc 2
    === om-all om-cleaninstall : ERROR bdir /data1/blyth/local/opticks_Debug_g411/build/u4 : non-zero rc 2
    === om-one-or-all cleaninstall : non-zero rc 2
    === opticks-full : ERR from opticks-full-make
    [lob] A[blyth@localhost opticks]$



libs rearranged
-----------------


::

    [ 94%] Built target U4SolidTest
    [ 95%] Linking CXX executable U4SolidMakerTest
    [ 95%] Built target U4SolidMakerTest
    [ 96%] Linking CXX executable U4NavigatorTest
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/binutils/2.40/bin/ld: CMakeFiles/U4NavigatorTest.dir/U4NavigatorTest.cc.o: undefined reference to symbol '_ZTI18G4VDiscreteProcess'
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/binutils/2.40/bin/ld: /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4processes_core.so: error adding symbols: DSO missing from command line
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/CMakeFiles/U4NavigatorTest.dir/build.make:133: tests/U4NavigatorTest] Error 1
    make[1]: *** [CMakeFiles/Makefile2:1792: tests/CMakeFiles/U4NavigatorTest.dir/all] Error 2
    [ 97%] Linking CXX executable U4SimtraceSimpleTest
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/binutils/2.40/bin/ld: CMakeFiles/U4SimtraceSimpleTest.dir/U4SimtraceSimpleTest.cc.o: undefined reference to symbol '_ZTI12G4OpRayleigh'
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/binutils/2.40/bin/ld: /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4processes_core.so: error adding symbols: DSO missing from command line
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/CMakeFiles/U4SimtraceSimpleTest.dir/build.make:133: tests/U4SimtraceSimpleTest] Error 1
    make[1]: *** [CMakeFiles/Makefile2:1922: tests/CMakeFiles/U4SimtraceSimpleTest.dir/all] Error 2
    [ 98%] Linking CXX executable U4TreeCreateSSimTest
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/binutils/2.40/bin/ld: CMakeFiles/U4TreeCreateSSimTest.dir/U4TreeCreateSSimTest.cc.o: undefined reference to symbol '_ZTI12G4OpRayleigh'
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/binutils/2.40/bin/ld: /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4processes_core.so: error adding symbols: DSO missing from command line
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/CMakeFiles/U4TreeCreateSSimTest.dir/build.make:133: tests/U4TreeCreateSSimTest] Error 1
    make[1]: *** [CMakeFiles/Makefile2:1870: tests/CMakeFiles/U4TreeCreateSSimTest.dir/all] Error 2
    [ 99%] Linking CXX executable U4TreeCreateTest
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/binutils/2.40/bin/ld: CMakeFiles/U4TreeCreateTest.dir/U4TreeCreateTest.cc.o: undefined reference to symbol '_ZTI12G4OpRayleigh'
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/binutils/2.40/bin/ld: /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4processes_core.so: error adding symbols: DSO missing from command line
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/CMakeFiles/U4TreeCreateTest.dir/build.make:133: tests/U4TreeCreateTest] Error 1
    make[1]: *** [CMakeFiles/Makefile2:1844: tests/CMakeFiles/U4TreeCreateTest.dir/all] Error 2
    [100%] Linking CXX executable U4TreeTest
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/binutils/2.40/bin/ld: CMakeFiles/U4TreeTest.dir/U4TreeTest.cc.o: undefined reference to symbol '_ZTI12G4OpRayleigh'
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/binutils/2.40/bin/ld: /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4processes_core.so: error adding symbols: DSO missing from command line
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/CMakeFiles/U4TreeTest.dir/build.make:133: tests/U4TreeTest] Error 1
    make[1]: *** [CMakeFiles/Makefile2:1818: tests/CMakeFiles/U4TreeTest.dir/all] Error 2
    make: *** [Makefile:146: all] Error 2
    === om-one-or-all make : non-zero rc 2
    === om-all om-make : ERROR bdir /data1/blyth/local/opticks_Debug_g411/build/u4 : non-zero rc 2
    === om-one-or-all make : non-zero rc 2
    === opticks-check-compute-capability : OPTICKS_COMPUTE_CAPABILITY 89 : looking good it is an integer expression of 30 or more
    === opticks-setup-generate : writing /data1/blyth/local/opticks_Debug_g411/bin/opticks-setup.sh



Enable G4_VERBOSE in U4::

    u4
    om-cleaninstall


    -- Found Geant4: /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/cmake/Geant4/Geant4Config.cmake (found version "11.4.0")
    --  G4_VERSION_INTEGER : 1140
    --
    -- G4_MODULE                : /home/blyth/opticks/cmake/Modules/FindG4.cmake
    -- G4_DIR                   : /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/cmake/Geant4
    -- G4_DIRDIR                : /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/cmake
    -- G4_VERSION_INTEGER       : 1140
    --
    -- Geant4_DIR               : /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/cmake/Geant4
    -- Geant4_VERSION           : 11.4.0
    -- Geant4_LIBRARIES         : Geant4::G4Tree;Geant4::G4FR;Geant4::G4GMocren;Geant4::G4RayTracer;Geant4::G4VRML;Geant4::G4ToolsSG;Geant4::G4vis_management;Geant4::G4modeling;Geant4::G4interfaces;Geant4::G4mctruth;Geant4::G4geomtext;Geant4::G4gdml;Geant4::G4analysis;Geant4::G4error_propagation;Geant4::G4readout;Geant4::G4physicslists;Geant4::G4run;Geant4::G4event;Geant4::G4tracking;Geant4::G4parmodels;Geant4::G4processes;Geant4::G4digits_hits;Geant4::G4track;Geant4::G4particles;Geant4::G4geometry;Geant4::G4materials;Geant4::G4graphics_reps;Geant4::G4intercoms;Geant4::G4global;Geant4::G4tools;Geant4::G4zlib;Geant4::G4ptl
    -- Geant4_INCLUDE_DIRS      : /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/include/Geant4
    -- Geant4_DEFINITIONS       :
    --
    -- CMAKE_INSTALL_INCLUDEDIR : include/U4
    --
    -- _lib Geant4::G4Tree _type SHARED_LIBRARY
    -- _lib G4Tree _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4Tree.so
    -- _lib Geant4::G4FR _type SHARED_LIBRARY
    -- _lib G4FR _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4FR.so
    -- _lib Geant4::G4GMocren _type SHARED_LIBRARY
    -- _lib G4GMocren _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4GMocren.so
    -- _lib Geant4::G4RayTracer _type SHARED_LIBRARY
    -- _lib G4RayTracer _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4RayTracer.so
    -- _lib Geant4::G4VRML _type SHARED_LIBRARY
    -- _lib G4VRML _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4VRML.so
    -- _lib Geant4::G4ToolsSG _type SHARED_LIBRARY
    -- _lib G4ToolsSG _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4ToolsSG.so
    -- _lib Geant4::G4vis_management _type SHARED_LIBRARY
    -- _lib G4vis_management _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4vis_management.so
    -- _lib Geant4::G4modeling _type SHARED_LIBRARY
    -- _lib G4modeling _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4modeling.so
    -- _lib Geant4::G4interfaces _type SHARED_LIBRARY
    -- _lib G4interfaces _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4interfaces.so
    -- _lib Geant4::G4mctruth _type SHARED_LIBRARY
    -- _lib G4mctruth _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4mctruth.so
    -- _lib Geant4::G4geomtext _type SHARED_LIBRARY
    -- _lib G4geomtext _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4geomtext.so
    -- _lib Geant4::G4gdml _type SHARED_LIBRARY
    -- _lib G4gdml _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4gdml.so
    -- _lib Geant4::G4analysis _type SHARED_LIBRARY
    -- _lib G4analysis _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4analysis.so
    -- _lib Geant4::G4error_propagation _type SHARED_LIBRARY
    -- _lib G4error_propagation _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4error_propagation.so
    -- _lib Geant4::G4readout _type SHARED_LIBRARY
    -- _lib G4readout _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4readout.so
    -- _lib Geant4::G4physicslists _type SHARED_LIBRARY
    -- _lib G4physicslists _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4physicslists.so
    -- _lib Geant4::G4run _type SHARED_LIBRARY
    -- _lib G4run _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4run.so
    -- _lib Geant4::G4event _type SHARED_LIBRARY
    -- _lib G4event _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4event.so
    -- _lib Geant4::G4tracking _type SHARED_LIBRARY
    -- _lib G4tracking _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4tracking.so
    -- _lib Geant4::G4parmodels _type SHARED_LIBRARY
    -- _lib G4parmodels _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4parmodels.so

    -- _lib Geant4::G4processes _type INTERFACE_LIBRARY
    --  _lib Geant4::G4processes _icd _icd-NOTFOUND : CURRENTLY IGNORING THIS INTERFACE_LIBRARY

    -- _lib Geant4::G4digits_hits _type SHARED_LIBRARY
    -- _lib G4digits_hits _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4digits_hits.so
    -- _lib Geant4::G4track _type SHARED_LIBRARY
    -- _lib G4track _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4track.so
    -- _lib Geant4::G4particles _type SHARED_LIBRARY
    -- _lib G4particles _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4particles.so
    -- _lib Geant4::G4geometry _type SHARED_LIBRARY
    -- _lib G4geometry _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4geometry.so
    -- _lib Geant4::G4materials _type SHARED_LIBRARY
    -- _lib G4materials _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4materials.so
    -- _lib Geant4::G4graphics_reps _type SHARED_LIBRARY
    -- _lib G4graphics_reps _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4graphics_reps.so
    -- _lib Geant4::G4intercoms _type SHARED_LIBRARY
    -- _lib G4intercoms _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4intercoms.so
    -- _lib Geant4::G4global _type SHARED_LIBRARY
    -- _lib G4global _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4global.so
    -- _lib Geant4::G4tools _type INTERFACE_LIBRARY
    --  _lib Geant4::G4tools _icd _icd-NOTFOUND : CURRENTLY IGNORING THIS INTERFACE_LIBRARY
    -- _lib Geant4::G4zlib _type SHARED_LIBRARY
    -- _lib G4zlib _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4zlib.so

    -- _lib Geant4::G4ptl _type INTERFACE_LIBRARY
    --  _lib Geant4::G4ptl _icd _icd-NOTFOUND : CURRENTLY IGNORING THIS INTERFACE_LIBRARY

    -- _defs
    -- /home/blyth/opticks/u4/CMakeLists.txt : ====== 1 ======= find GDXML CLHEP OpticksXercesC
    -- /home/blyth/opticks/u4/CMakeLists.txt : ====== 2 =======  find Custom4 PMTSim
    -- Could NOT find PMTSim_standalone (missing: PMTSim_standalone_DIR)
    -- /home/blyth/opticks/u4/CMakeLists.txt : PMTSim_standalone_FOUND       : 0




After adjust cmake/Modules/FindG4.cmake to handle ILL::


    -- Found Geant4: /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/cmake/Geant4/Geant4Config.cmake (found version "11.4.0")
    --  G4_VERSION_INTEGER : 1140
    --
    -- G4_MODULE                : /home/blyth/opticks/cmake/Modules/FindG4.cmake
    -- G4_DIR                   : /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/cmake/Geant4
    -- G4_DIRDIR                : /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/cmake
    -- G4_VERSION_INTEGER       : 1140
    --
    -- Geant4_DIR               : /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/cmake/Geant4
    -- Geant4_VERSION           : 11.4.0
    -- Geant4_LIBRARIES         : Geant4::G4Tree;Geant4::G4FR;Geant4::G4GMocren;Geant4::G4RayTracer;Geant4::G4VRML;Geant4::G4ToolsSG;Geant4::G4vis_management;Geant4::G4modeling;Geant4::G4interfaces;Geant4::G4mctruth;Geant4::G4geomtext;Geant4::G4gdml;Geant4::G4analysis;Geant4::G4error_propagation;Geant4::G4readout;Geant4::G4physicslists;Geant4::G4run;Geant4::G4event;Geant4::G4tracking;Geant4::G4parmodels;Geant4::G4processes;Geant4::G4digits_hits;Geant4::G4track;Geant4::G4particles;Geant4::G4geometry;Geant4::G4materials;Geant4::G4graphics_reps;Geant4::G4intercoms;Geant4::G4global;Geant4::G4tools;Geant4::G4zlib;Geant4::G4ptl
    -- Geant4_INCLUDE_DIRS      : /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/include/Geant4
    -- Geant4_DEFINITIONS       :
    --
    -- CMAKE_INSTALL_INCLUDEDIR : include/U4
    --
    -- _lib Geant4::G4Tree _type SHARED_LIBRARY
    -- _lib Geant4::G4Tree _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4Tree.so
    -- _lib Geant4::G4FR _type SHARED_LIBRARY
    -- _lib Geant4::G4FR _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4FR.so
    -- _lib Geant4::G4GMocren _type SHARED_LIBRARY
    -- _lib Geant4::G4GMocren _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4GMocren.so
    -- _lib Geant4::G4RayTracer _type SHARED_LIBRARY
    -- _lib Geant4::G4RayTracer _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4RayTracer.so
    -- _lib Geant4::G4VRML _type SHARED_LIBRARY
    -- _lib Geant4::G4VRML _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4VRML.so
    -- _lib Geant4::G4ToolsSG _type SHARED_LIBRARY
    -- _lib Geant4::G4ToolsSG _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4ToolsSG.so
    -- _lib Geant4::G4vis_management _type SHARED_LIBRARY
    -- _lib Geant4::G4vis_management _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4vis_management.so
    -- _lib Geant4::G4modeling _type SHARED_LIBRARY
    -- _lib Geant4::G4modeling _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4modeling.so
    -- _lib Geant4::G4interfaces _type SHARED_LIBRARY
    -- _lib Geant4::G4interfaces _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4interfaces.so
    -- _lib Geant4::G4mctruth _type SHARED_LIBRARY
    -- _lib Geant4::G4mctruth _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4mctruth.so
    -- _lib Geant4::G4geomtext _type SHARED_LIBRARY
    -- _lib Geant4::G4geomtext _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4geomtext.so
    -- _lib Geant4::G4gdml _type SHARED_LIBRARY
    -- _lib Geant4::G4gdml _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4gdml.so
    -- _lib Geant4::G4analysis _type SHARED_LIBRARY
    -- _lib Geant4::G4analysis _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4analysis.so
    -- _lib Geant4::G4error_propagation _type SHARED_LIBRARY
    -- _lib Geant4::G4error_propagation _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4error_propagation.so
    -- _lib Geant4::G4readout _type SHARED_LIBRARY
    -- _lib Geant4::G4readout _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4readout.so
    -- _lib Geant4::G4physicslists _type SHARED_LIBRARY
    -- _lib Geant4::G4physicslists _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4physicslists.so
    -- _lib Geant4::G4run _type SHARED_LIBRARY
    -- _lib Geant4::G4run _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4run.so
    -- _lib Geant4::G4event _type SHARED_LIBRARY
    -- _lib Geant4::G4event _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4event.so
    -- _lib Geant4::G4tracking _type SHARED_LIBRARY
    -- _lib Geant4::G4tracking _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4tracking.so
    -- _lib Geant4::G4parmodels _type SHARED_LIBRARY
    -- _lib Geant4::G4parmodels _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4parmodels.so
    -- _lib Geant4::G4processes _type INTERFACE_LIBRARY
    -- Wrapped INTERFACE_LIBRARY Geant4::G4processes -> Opticks::G4processes
    -- _lib Geant4::G4digits_hits _type SHARED_LIBRARY
    -- _lib Geant4::G4digits_hits _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4digits_hits.so
    -- _lib Geant4::G4track _type SHARED_LIBRARY
    -- _lib Geant4::G4track _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4track.so
    -- _lib Geant4::G4particles _type SHARED_LIBRARY
    -- _lib Geant4::G4particles _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4particles.so
    -- _lib Geant4::G4geometry _type SHARED_LIBRARY
    -- _lib Geant4::G4geometry _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4geometry.so
    -- _lib Geant4::G4materials _type SHARED_LIBRARY
    -- _lib Geant4::G4materials _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4materials.so
    -- _lib Geant4::G4graphics_reps _type SHARED_LIBRARY
    -- _lib Geant4::G4graphics_reps _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4graphics_reps.so
    -- _lib Geant4::G4intercoms _type SHARED_LIBRARY
    -- _lib Geant4::G4intercoms _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4intercoms.so
    -- _lib Geant4::G4global _type SHARED_LIBRARY
    -- _lib Geant4::G4global _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4global.so
    -- _lib Geant4::G4tools _type INTERFACE_LIBRARY
    -- Wrapped INTERFACE_LIBRARY Geant4::G4tools -> Opticks::G4tools
    -- _lib Geant4::G4zlib _type SHARED_LIBRARY
    -- _lib Geant4::G4zlib _loc /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/Geant4/11.4.0/lib64/libG4zlib.so
    -- _lib Geant4::G4ptl _type INTERFACE_LIBRARY
    -- Wrapped INTERFACE_LIBRARY Geant4::G4ptl -> Opticks::G4ptl
    -- _defs
    -- /home/blyth/opticks/u4/CMakeLists.txt : ====== 1 ======= find GDXML CLHEP OpticksXercesC
    -- /home/blyth/opticks/u4/CMakeLists.txt : ====== 2 =======  find Custom4 PMTSim
    -- Could NOT find PMTSim_standalone (missing: PMTSim_standalone_DIR)







fixed test fail - gcc15 gives error when setting value beyond vector bounds that gcc11 didnt notice
-----------------------------------------------------------------------------------------------------

::


    FAILS:  1   / 223   :  Mon Aug 24 17:38:50 2026  :  GEOM RaindropRockAirWater
      43 /110 Test #43 : SysRapTest.ArrayTest                                    Subprocess aborted***Exception:   0.26



            Start  43: SysRapTest.ArrayTest
     43/110 Test  #43: SysRapTest.ArrayTest .....................................Subprocess aborted***Exception:   0.09 sec
    2026-08-24 17:39:59.646 INFO  [1695224] [main@62] 0 : -
    2026-08-24 17:39:59.646 INFO  [1695224] [main@62] 1 : -
    2026-08-24 17:39:59.646 INFO  [1695224] [main@62] 2 : -
    2026-08-24 17:39:59.646 INFO  [1695224] [main@62] 3 : -
    2026-08-24 17:39:59.646 INFO  [1695224] [main@62] 4 : -
    2026-08-24 17:39:59.646 INFO  [1695224] [main@62] 5 :         yo :         42
    2026-08-24 17:39:59.646 INFO  [1695224] [main@62] 6 : -
    2026-08-24 17:39:59.646 INFO  [1695224] [main@62] 7 : -
    2026-08-24 17:39:59.646 INFO  [1695224] [main@62] 8 : -
    2026-08-24 17:39:59.646 INFO  [1695224] [main@62] 9 : -
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/gcc/15.1.0/include/c++/15.1.0/array:210: constexpr std::array<_Tp, _Nm>::value_type& std::array<_Tp, _Nm>::operator[](size_type) [with _Tp = Demo*; long unsigned int _Nm = 10; reference = Demo*&; size_type = long unsigned int]: Assertion '__n < this->size()' failed.

            Start  44: SysRapTest.SBacktraceTest
     44/110 Test  #44: SysRapTest.SBacktraceTest ................................   Passed    0.01 sec
            Start  45: SysRapTest.SStackFrameTest
     45/110 Test  #45: SysRapTest.SStackFrameTest ...............................   Passed    0.01 sec






    [lob] A[blyth@localhost sysrap]$ which ArrayTest
    /data1/blyth/local/opticks_Debug_g411/lib/ArrayTest
    [lob] A[blyth@localhost sysrap]$ ArrayTest
    2026-08-24 17:41:02.931 INFO  [1695504] [main@62] 0 : -
    2026-08-24 17:41:02.931 INFO  [1695504] [main@62] 1 : -
    2026-08-24 17:41:02.931 INFO  [1695504] [main@62] 2 : -
    2026-08-24 17:41:02.931 INFO  [1695504] [main@62] 3 : -
    2026-08-24 17:41:02.931 INFO  [1695504] [main@62] 4 : -
    2026-08-24 17:41:02.931 INFO  [1695504] [main@62] 5 :         yo :         42
    2026-08-24 17:41:02.931 INFO  [1695504] [main@62] 6 : -
    2026-08-24 17:41:02.931 INFO  [1695504] [main@62] 7 : -
    2026-08-24 17:41:02.931 INFO  [1695504] [main@62] 8 : -
    2026-08-24 17:41:02.931 INFO  [1695504] [main@62] 9 : -
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/gcc/15.1.0/include/c++/15.1.0/array:210: constexpr std::array<_Tp, _Nm>::value_type& std::array<_Tp, _Nm>::operator[](size_type) [with _Tp = Demo*; long unsigned int _Nm = 10; reference = Demo*&; size_type = long unsigned int]: Assertion '__n < this->size()' failed.
    Aborted (core dumped)
    [lob] A[blyth@localhost sysrap]$




Test Geant4 1140 and gcc15 compatible Opticks changes building against with old Geant4 1042 libs and gcc11
-----------------------------------------------------------------------------------------------------------

1. change the config ~/j/opticks_config.sh::

     20 export OPTICKS_HOME=$HOME/opticks
     21
     22 config=Debug
     23 #config=Debug_g411
     24 export OPTICKS_CONFIG=${OPTICKS_CONFIG:-$config}

2. clean build::

    lobbc

3. test::

    SLOW: tests taking longer that 15.0 seconds

    FAILS:  0   / 223   :  Mon Aug 24 18:35:35 2026  :  GEOM RaindropRockAirWater

    test_secs  :  19                       ## small GEOM like RaindropRockAirWater are ~5x faster that full ones
    test_start :  2026-08-24 18:35:16
    test_end   :  2026-08-24 18:35:35





