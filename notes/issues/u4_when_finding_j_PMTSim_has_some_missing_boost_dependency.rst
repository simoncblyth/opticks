u4_when_finding_j_PMTSim_has_some_missing_boost_dependency
==============================================================


Unfortunately boost dependency of PMTSim(_standalone) is unavoidable
without more work than it merits to remove it. 

::

    u4
    om


    [ 19%] Building CXX object CMakeFiles/U4.dir/Local_DsG4Scintillation.cc.o
    [ 19%] Building CXX object CMakeFiles/U4.dir/U4Physics.cc.o
    [ 20%] Linking CXX shared library libU4.so
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: cannot find -lBoost::system
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: cannot find -lBoost::program_options
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: cannot find -lBoost::filesystem
    collect2: error: ld returned 1 exit status
    make[2]: *** [libU4.so] Error 1
    make[1]: *** [CMakeFiles/U4.dir/all] Error 2
    make: *** [all] Error 2
    === om-one-or-all make : non-zero rc 2
    [blyth@localhost u4]$ 


::

    [blyth@localhost PMTSim]$ ldd /data/blyth/opticks_Debug/lib64/libPMTSim_standalone.so
        linux-vdso.so.1 =>  (0x00007ffe5d52c000)
        libCLHEP-2.4.1.0.so => /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/CLHEP/2.4.1.0/lib/libCLHEP-2.4.1.0.so (0x00007f22da58a000)
        libSysRap.so => /data/blyth/opticks_Debug/lib64/../lib64/libSysRap.so (0x00007f22da249000)
        libboost_system-mt.so.1.53.0 => /usr/lib64/libboost_system-mt.so.1.53.0 (0x00007f22da045000)
        libboost_program_options-mt.so.1.53.0 => /usr/lib64/libboost_program_options-mt.so.1.53.0 (0x00007f22d9dd3000)
        libboost_filesystem-mt.so.1.53.0 => /usr/lib64/libboost_filesystem-mt.so.1.53.0 (0x00007f22d9bbc000)
        libCustom4.so => /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc3.dc1/ExternalLibs/custom4/0.1.8/lib64/libCustom4.so (0x00007f22da7d7000)
        libG4Tree.so => /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4Tree.so (0x00007f22da7c2000)
        libG4FR.so => /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4FR.so (0x00007f22da7aa000)
        libG4GMocren.so => /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4GMocren.so (0x00007f22da759000)
        libG4visHepRep.so => /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4visHepRep.so (0x00007f22d9b1d000)
        libG4RayTracer.so => /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4RayTracer.so (0x00007f22da728000)
        libG4VRML.so => /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4VRML.so (0x00007f22d9b01000)
        libG4vis_management.so => /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4vis_management.so (0x00007f22d99f1000)
        libG4modeling.so => /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4modeling.so (0x00007f22d9918000)


::

    om-clean
    om-conf


    -- cmake/Modules/OpticksCUDAFlags.cmake : using default OPTICKS_CUDA_NVCC_DIALECT variable c++17
    -- /home/blyth/opticks/u4/tests/CMakeLists.txt : PMTSim_FOUND     : 1
    -- /home/blyth/opticks/u4/tests/CMakeLists.txt : Custom4_FOUND    : 1
    -- Configuring done
    CMake Error at /data/blyth/opticks_Debug/lib64/cmake/pmtsim/pmtsim-targets.cmake:61 (set_target_properties):
      The link interface of target "Opticks::PMTSim" contains:

        Boost::system

      but the target was not found.  Possible reasons include:

        * There is a typo in the target name.
        * A find_package call is missing for an IMPORTED target.
        * An ALIAS target is missing.

    Call Stack (most recent call first):
      /data/blyth/opticks_Debug/lib64/cmake/pmtsim/pmtsim-config.cmake:20 (include)
      CMakeLists.txt:33 (find_package)


    CMake Error at tests/CMakeLists.txt:170 (target_link_libraries):
      Target "U4SimulateTest" links to:

        Opticks::PMTSim_standalone

      but the target was not found.  Possible reasons include:

        * There is a typo in the target name.
        * A find_package call is missing for an IMPORTED target.
        * An ALIAS target is missing.




Adhoc install boost::

    boost-
    boost--

    ...

    ln-UNIX /data/blyth/opticks_Debug_externals/boost/lib/libboost_system.so.1.70
    boost-install.generate-cmake-config- /data/blyth/opticks_Debug_externals/boost/lib/cmake/boost_system-1.70.0/boost_system-config.cmake
    boost-install.generate-cmake-config-version- /data/blyth/opticks_Debug_externals/boost/lib/cmake/boost_system-1.70.0/boost_system-config-version.cmake
    boost-install.generate-cmake-variant- /data/blyth/opticks_Debug_externals/boost/lib/cmake/boost_system-1.70.0/libboost_system-variant-shared.cmake
    gcc.compile.c++ /data/blyth/opticks_Debug_externals/boost.build/boost_1_70_0.build/boost/bin.v2/libs/filesystem/build/gcc-11.2.0/release/link-static/threading-multi/visibility-hidden/codecvt_error_category.o
    gcc.compile.c++ /data/blyth/opticks_Debug_externals/boost.build/boost_1_70_0.build/boost/bin.v2/libs/filesystem/build/gcc-11.2.0/release/link-static/threading-multi/visibility-hidden/operations.o
    gcc.compile.c++ /data/blyth/opticks_Debug_externals/boost.build/boost_1_70_0.build/boost/bin.v2/libs/filesystem/build/gcc-11.2.0/release/link-static/threading-multi/visibility-hidden/path.o
    ...on 15400th target...
    gcc.compile.c++ /data/blyth/opticks_Debug_externals/boost.build/boost_1_70_0.build/boost/bin.v2/libs/filesystem/build/gcc-11.2.0/release/link-static/threading-multi/visibility-hidden/path_traits.o
    gcc.compile.c++ /data/blyth/opticks_Debug_externals/boost.build/boost_1_70_0.build/boost/bin.v2/libs/filesystem/build/gcc-11.2.0/release/link-static/threading-multi/visibility-hidden/portability.o
    gcc.compile.c++ /data/blyth/opticks_Debug_externals/boost.build/boost_1_70_0.build/boost/bin.v2/libs/filesystem/build/gcc-11.2.0/release/link-static/threading-multi/visibility-hidden/unique_path.o
    gcc.compile.c++ /data/blyth/opticks_Debug_externals/boost.build/boost_1_70_0.build/boost/bin.v2/libs/filesystem/build/gcc-11.2.0/release/link-static/threading-multi/visibility-hidden/utf8_codecvt_facet.o
    gcc.compile.c++ /data/blyth/opticks_Debug_externals/boost.build/boost_1_70_0.build/boost/bin.v2/libs/filesystem/build/gcc-11.2.0/release/link-static/threading-multi/visibility-hidden/windows_file_codecvt.o
    gcc.archive /data/blyth/opticks_Debug_externals/boost.build/boost_1_70_0.build/boost/bin.v2/libs/filesystem/build/gcc-11.2.0/release/link-static/threading-multi/visibility-hidden/libboost_filesystem.a
    common.copy /data/blyth/opticks_Debug_externals/boost/lib/libboost_filesystem.a
    boost-install.generate-cmake-variant- /data/blyth/opticks_Debug_externals/boost/lib/cmake/boost_filesystem-1.70.0/libboost_filesystem-variant-static.cmake
    ...updated 15408 targets...
    [blyth@localhost boost_1_70_0]$ 



