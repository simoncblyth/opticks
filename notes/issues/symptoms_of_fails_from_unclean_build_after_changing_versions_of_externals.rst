symptoms_of_fails_from_unclean_build_after_changing_versions_of_externals
=============================================================================

Overview
---------

Update builds do not handle changes in the versions of externals 
that could for example be triggered by env changes that cause the system xerces-c to 
be built against as opposed to the controlled version. 

::

    opticks-full-make 




Issue
-------


Build of GDXML fails with XercesC 3.2 version/linking issue::

    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213XMLAttDefListE[_ZTVN11xercesc_3_213XMLAttDefListE]+0x30): undefined reference to `xercesc_3_2::XMLAttDefList::getProtoType() const'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTIN11xercesc_3_213DTDEntityDeclE[_ZTIN11xercesc_3_213DTDEntityDeclE]+0x10): undefined reference to `typeinfo for xercesc_3_2::XMLEntityDecl'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::XMLUni::fgXercescDefaultLocale'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::SAXParseException::getLineNumber() const'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::setDoSchema(bool)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `typeinfo for xercesc_3_2::XMLException'


Investigate::

    [blyth@localhost opticks]$ cd gdxml
    [blyth@localhost gdxml]$ om-clean
    rm -rf /data/blyth/opticks_Debug/build/gdxml && mkdir -p /data/blyth/opticks_Debug/build/gdxml
    [blyth@localhost gdxml]$ om-conf

Uncomment verbose switches in CMakeLists.txt::

    -- _lib G4zlib _loc /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4zlib.so 
    -- _defs G4_STORE_TRAJECTORY;G4VERBOSE;G4UI_USE;G4VIS_USE;G4MULTITHREADED 
    -- OpticksXercesC_MODULE : /home/blyth/opticks/cmake/Modules/FindOpticksXercesC.cmake 
    -- TARGET Geant4::G4persistency : NOT-FOUND 
    -- TARGET G4persistency : FOUND 
    -- TARGET Geant4::G4gdml : NOT-FOUND 
    -- TARGET G4gdml : NOT-FOUND 
    -- TARGET XercesC : NOT-FOUND 
    -- TARGET XercesC::XercesC : NOT-FOUND 
    -- FindOpticksXercesC.cmake. Found G4persistency target _lll G4geometry;G4global;G4graphics_reps;G4intercoms;G4materials;G4particles;G4digits_hits;G4event;G4processes;G4run;G4track;G4tracking;/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J24.1.x/ExternalLibs/Xercesc/3.2.4/lib/libxerces-c.so
    --  G4persistency.xercesc_lib         : /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J24.1.x/ExternalLibs/Xercesc/3.2.4/lib/libxerces-c.so 
    --  G4persistency.xercesc_include_dir : /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J24.1.x/ExternalLibs/Xercesc/3.2.4/include 
    -- FindOpticksXercesC.cmake OpticksXercesC_MODULE      : /home/blyth/opticks/cmake/Modules/FindOpticksXercesC.cmake  
    -- FindOpticksXercesC.cmake OpticksXercesC_INCLUDE_DIR : /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J24.1.x/ExternalLibs/Xercesc/3.2.4/include  
    -- FindOpticksXercesC.cmake OpticksXercesC_LIBRARY     : /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J24.1.x/ExternalLibs/Xercesc/3.2.4/lib/libxerces-c.so  
    -- FindOpticksXercesC.cmake OpticksXercesC_FOUND       : YES  
    -- Configuring GDXMLTest


HMM this symbol is there::

    [blyth@localhost gdxml]$ nm /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J24.1.x/ExternalLibs/Xercesc/3.2.4/lib/libxerces-c.so | c++filt | grep xercesc_3_2::AbstractDOMParser::setDoSchema
    0000000000225970 T xercesc_3_2::AbstractDOMParser::setDoSchema(bool)

::

    cd ~/opticks/gdxml
    om-clean
    om-conf
    om    # works ? 


U4 issue::

    [ 17%] Building CXX object CMakeFiles/U4.dir/Local_DsG4Scintillation.cc.o
    [ 18%] Building CXX object CMakeFiles/U4.dir/U4Physics.cc.o
    [ 19%] Building CXX object CMakeFiles/U4.dir/Local_G4Cerenkov_modified.cc.o
    [ 20%] Linking CXX shared library libU4.so
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: aTrackAllocator: TLS reference in /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4error_propagation.so mismatches non-TLS reference in CMakeFiles/U4.dir/Local_G4Cerenkov_modified.cc.o
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4error_propagation.so: error adding symbols: bad value
    collect2: error: ld returned 1 exit status
    make[2]: *** [libU4.so] Error 1
    make[1]: *** [CMakeFiles/U4.dir/all] Error 2
    make: *** [all] Error 2
    === om-one-or-all install : non-zero rc 2


Again doing a clean makes the build succeed. Suggests should automatically clean ?::

    opticks-full-make does om-install 

G4CX::

    [ 20%] Built target G4CX
    [ 26%] Building CXX object tests/CMakeFiles/G4CXOpticks_SetGeometry_GetInputPhoton_Test.dir/G4CXOpticks_SetGeometry_GetInputPhoton_Test.cc.o
    [ 33%] Building CXX object tests/CMakeFiles/G4CXTest.dir/G4CXTest.cc.o
    [ 40%] Building CXX object tests/CMakeFiles/G4CXOpticks_setGeometry_Test.dir/G4CXOpticks_setGeometry_Test.cc.o
    [ 46%] Building CXX object tests/CMakeFiles/G4CXSimtraceTest.dir/G4CXSimtraceTest.cc.o
    [ 53%] Building CXX object tests/CMakeFiles/G4CXSimulateTest.dir/G4CXSimulateTest.cc.o
    [ 60%] Building CXX object tests/CMakeFiles/G4CXRenderTest.dir/G4CXRenderTest.cc.o
    [ 66%] Linking CXX executable G4CXOpticks_setGeometry_Test
    [ 73%] Linking CXX executable G4CXSimtraceTest
    [ 86%] Linking CXX executable G4CXRenderTest
    [ 86%] Linking CXX executable G4CXSimulateTest
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /data/blyth/opticks_externals/custom4/0.1.9/lib64/libCustom4.so: undefined reference to `G4cerr'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /data/blyth/opticks_externals/custom4/0.1.9/lib64/libCustom4.so: undefined reference to `G4cout'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /data/blyth/opticks_externals/custom4/0.1.9/lib64/libCustom4.so: undefined reference to `G4cerr'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /data/blyth/opticks_externals/custom4/0.1.9/lib64/libCustom4.so: undefined reference to `G4cout'
    collect2: error: ld returned 1 exit status
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /data/blyth/opticks_externals/custom4/0.1.9/lib64/libCustom4.so: undefined reference to `G4cerr'
    make[2]: *** [tests/G4CXSimtraceTest] Error 1
    make[1]: *** [tests/CMakeFiles/G4CXSimtraceTest.dir/all] Error 2
    make[1]: *** Waiting for unfinished jobs....
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /data/blyth/opticks_externals/custom4/0.1.9/lib64/libCustom4.so: undefined reference to `collect2: error: ld returned 1 exit status
    G4cerr'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /data/blyth/opticks_externals/custom4/0.1.9/lib64/libCustom4.so: undefined reference to `G4cout'
    make[2]: *** [tests/G4CXOpticks_setGeometry_Test] Error 1
    make[1]: *** [tests/CMakeFiles/G4CXOpticks_setGeometry_Test.dir/all] Error 2
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /data/blyth/opticks_externals/custom4/0.1.9/lib64/libCustom4.so: undefined reference to `G4cout'
    collect2: error: ld returned 1 exit status
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/G4CXSimulateTest] Error 1
    make[1]: *** [tests/CMakeFiles/G4CXSimulateTest.dir/all] Error 2
    make[2]: *** [tests/G4CXRenderTest] Error 1
    make[1]: *** [tests/CMakeFiles/G4CXRenderTest.dir/all] Error 2
    [ 93%] Linking CXX executable G4CXOpticks_SetGeometry_GetInputPhoton_Test
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /data/blyth/opticks_externals/custom4/0.1.9/lib64/libCustom4.so: undefined reference to `G4cerr'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /data/blyth/opticks_externals/custom4/0.1.9/lib64/libCustom4.so: undefined reference to `G4cout'
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/G4CXOpticks_SetGeometry_GetInputPhoton_Test] Error 1
    make[1]: *** [tests/CMakeFiles/G4CXOpticks_SetGeometry_GetInputPhoton_Test.dir/all] Error 2
    [100%] Linking CXX executable G4CXTest
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: aPrimaryParticleAllocator: TLS reference in /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4RayTracer.so mismatches non-TLS reference in CMakeFiles/G4CXTest.dir/G4CXTest.cc.o
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4RayTracer.so: error adding symbols: bad value
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/G4CXTest] Error 1
    make[1]: *** [tests/CMakeFiles/G4CXTest.dir/all] Error 2
    make: *** [all] Error 2
    === om-one-or-all install : non-zero rc 2
    === om-all om-install : ERROR bdir /data/blyth/opticks_Debug/build/g4cx : non-zero rc 2
    === om-one-or-all install : non-zero rc 2
    [blyth@localhost opticks]$ 


Notice there are paths there that should not be, like "/data/blyth/opticks_externals/" indicating 
its an unclean build. 
Yet again doing a clean first makes the build succeed. 
Where to auto-clean ?::

    opticks-full-make 
    om-install 


