geant4_juno_initialization_too_long_for_unit_tests
=====================================================

How to handle ?
----------------

* Use a different faster geometry (DYB) for unit tests. 
* Add OPTICKS_TEST_KEY which trumps OPTICKS_KEY when testing ?

  * YES : but how to distinguish "testing" ? 
  * simpler to explicity set OPTICKS_KEY to pick faster geometry within opticks-t ?
    
    * opticks-t is a bash function that will change environment of caller
    * better to create a script with its own process environment to contain the change
    * OR JUST do it inline at user level (ie not in repo)::

   opticks-t-fast(){ OPTICKS_KEY=$(geocache-;geocache-dx-v0-key) opticks-t ; }



Still a problem
---------------------

::

    SLOW: tests taking longer that 15 seconds
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    Passed                         23.41  
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Passed                         24.79  
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Passed                         23.62  
      7  /34  Test #7  : CFG4Test.CG4Test                              Passed                         1048.48 
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   Passed                         23.77  
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    Passed                         26.01  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         1036.89 


    FAILS:  1   / 420   :  Fri Dec  6 17:24:19 2019   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      9.66   



Issue CG4Test taking ages
----------------------------

::

    Fri Oct 25 11:01:52 CST 2019
    === om-test-one : cfg4            /home/blyth/opticks/cfg4                                     /home/blyth/local/opticks/build/cfg4                         
    Fri Oct 25 11:01:52 CST 2019
    Test project /home/blyth/local/opticks/build/cfg4
          Start  1: CFG4Test.CMaterialLibTest
     1/34 Test  #1: CFG4Test.CMaterialLibTest .................   Passed    0.62 sec
          Start  2: CFG4Test.CMaterialTest
     2/34 Test  #2: CFG4Test.CMaterialTest ....................   Passed    0.59 sec
          Start  3: CFG4Test.CTestDetectorTest
     3/34 Test  #3: CFG4Test.CTestDetectorTest ................   Passed   22.05 sec
          Start  4: CFG4Test.CGDMLTest
     4/34 Test  #4: CFG4Test.CGDMLTest ........................   Passed    0.10 sec
          Start  5: CFG4Test.CGDMLDetectorTest
     5/34 Test  #5: CFG4Test.CGDMLDetectorTest ................   Passed   21.83 sec
          Start  6: CFG4Test.CGeometryTest
     6/34 Test  #6: CFG4Test.CGeometryTest ....................   Passed   21.99 sec
          Start  7: CFG4Test.CG4Test

Even got a timeout, but that may be because I attached gdb::

     7/34 Test  #7: CFG4Test.CG4Test ..........................***Timeout 1510.69 sec


Connecting to the process it is G4RunManager::RunInitialization voxelizing that takes the time::

    [blyth@localhost opticks]$ gdb -p  320487
    GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-114.el7
    Copyright (C) 2013 Free Software Foundation, Inc.

    Loaded symbols for /lib64/libpcre.so.1
    Reading symbols from /usr/lib64/gconv/UTF-16.so...(no debugging symbols found)...done.
    Loaded symbols for /usr/lib64/gconv/UTF-16.so
    0x00007f41d3822808 in HepGeom::Plane3D<double>::distance (this=0x15758db0, p=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/include/CLHEP/Geometry/Plane3D.h:103
    103       return a()*p.x() + b()*p.y() + c()*p.z() + d();
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007f41d3822808 in HepGeom::Plane3D<double>::distance (this=0x15758db0, p=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/include/CLHEP/Geometry/Plane3D.h:103
    #1  0x00007f41d381fdf1 in G4BoundingEnvelope::ClipVoxelByPlanes (this=0x7fff5d919d00, pBits=3925, pBox=..., pPlanes=std::vector of length 7, capacity 8 = {...}, pAABB=..., pExtent=...)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4BoundingEnvelope.cc:1068
    #2  0x00007f41d381a641 in G4BoundingEnvelope::CalculateExtent (this=0x7fff5d919d00, pAxis=kYAxis, pVoxelLimits=..., pTransform3D=..., pMin=@0x7fff5d919fb8: 8.9999999999999999e+99, pMax=@0x7fff5d919fb0: -8.9999999999999999e+99)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4BoundingEnvelope.cc:548
    #3  0x00007f41d3942de0 in G4Polycone::CalculateExtent (this=0x17af1a20, pAxis=kYAxis, pVoxelLimit=..., pTransform=..., pMin=@0x7fff5d91a5c8: -4957.9704225729738, pMax=@0x7fff5d91a5c0: -4687.9671068287389)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/solids/specific/src/G4Polycone.cc:695
    #4  0x00007f41d384461e in G4SmartVoxelHeader::BuildNodes (this=0x15755cc0, pVolume=0x17b13860, pLimits=..., pCandidates=0x157750f0, pAxis=kYAxis)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:852
    #5  0x00007f41d384375f in G4SmartVoxelHeader::BuildVoxelsWithinLimits (this=0x15755cc0, pVolume=0x17b13860, pLimits=..., pCandidates=0x157750f0)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:476
    #6  0x00007f41d38427fc in G4SmartVoxelHeader::G4SmartVoxelHeader (this=0x15755cc0, pVolume=0x17b13860, pLimits=..., pCandidates=0x157750f0, pSlice=172)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:119
    #7  0x00007f41d384537c in G4SmartVoxelHeader::RefineNodes (this=0x157460a0, pVolume=0x17b13860, pLimits=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:1244
    #8  0x00007f41d3843ae7 in G4SmartVoxelHeader::BuildVoxelsWithinLimits (this=0x157460a0, pVolume=0x17b13860, pLimits=..., pCandidates=0xa27bdd0)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:568
    #9  0x00007f41d38427fc in G4SmartVoxelHeader::G4SmartVoxelHeader (this=0x157460a0, pVolume=0x17b13860, pLimits=..., pCandidates=0xa27bdd0, pSlice=810)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:119
    #10 0x00007f41d384537c in G4SmartVoxelHeader::RefineNodes (this=0x9f236c0, pVolume=0x17b13860, pLimits=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:1244
    #11 0x00007f41d3843ae7 in G4SmartVoxelHeader::BuildVoxelsWithinLimits (this=0x9f236c0, pVolume=0x17b13860, pLimits=..., pCandidates=0x7fff5d91b010)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:568
    #12 0x00007f41d3842cdb in G4SmartVoxelHeader::BuildVoxels (this=0x9f236c0, pVolume=0x17b13860) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:258
    #13 0x00007f41d384270d in G4SmartVoxelHeader::G4SmartVoxelHeader (this=0x9f236c0, pVolume=0x17b13860, pSlice=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:82
    #14 0x00007f41d382fd2d in G4GeometryManager::BuildOptimisations (this=0x9f18c30, allOpts=true, verbose=false) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4GeometryManager.cc:200
    #15 0x00007f41d382faa5 in G4GeometryManager::CloseGeometry (this=0x9f18c30, pOptimise=true, verbose=false, pVolume=0x0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4GeometryManager.cc:102
    #16 0x00007f41d741d589 in G4RunManagerKernel::ResetNavigator (this=0x5d70110) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManagerKernel.cc:757
    #17 0x00007f41d741d3a6 in G4RunManagerKernel::RunInitialization (this=0x5d70110, fakeRun=false) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManagerKernel.cc:699
    #18 0x00007f41d740df69 in G4RunManager::RunInitialization (this=0x5d6d340) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:313
    #19 0x00007f41d740dd0f in G4RunManager::BeamOn (this=0x5d6d340, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:272
    #20 0x00007f41dab55eee in CG4::propagate (this=0x5d68350) at /home/blyth/opticks/cfg4/CG4.cc:398
    #21 0x00000000004047fa in main (argc=1, argv=0x7fff5d91dc38) at /home/blyth/opticks/cfg4/tests/CG4Test.cc:71
    (gdb) 


