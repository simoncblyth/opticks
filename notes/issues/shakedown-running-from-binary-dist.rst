shakedown-running-from-binary-dist
=====================================

Context
----------

* :doc:`packaging-opticks-and-externals-for-use-on-gpu-cluster`
* :doc:`shakedown-running-from-expanded-binary-tarball` am earlier look at the same thing from April 2019


Workflow for testing binary dist
-----------------------------------

0. as "blyth" build Opticks, then create and explode the release binary distribution onto fake /cvmfs:: 

   o && om- && om-- && okdist- && okdist--

1. login as simon, and run some tests::

    [blyth@localhost issues]$ su - simon   # NB simon needs .bash_profile to source .bashrc for this to run ~/.opticks_setup_minimal from .bashrc
    [simon@localhost ~]$ which OKTest 
    /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/lib/OKTest

2. back to 0, fixing the problem


Minimal environment running using binary release on fake /cvmfs 
--------------------------------------------------------------------

User "simon" .bashrc::

    # ~/.opticks_setup_minimal

    export OPTICKS_INSTALL_PREFIX=/cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg
    export PATH=${OPTICKS_INSTALL_PREFIX}/lib:$PATH

    export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce   ## geocache-j1808-v5-export 


OKTest::

    [simon@localhost ~]$ OpticksResource=ERROR OKTest
    PLOG::EnvLevel adjusting loglevel by envvar   key OpticksResource level ERROR fallback DEBUG
    2019-09-15 13:42:22.344 INFO  [134648] [Opticks::init@354] INTEROP_MODE
    2019-09-15 13:42:22.345 INFO  [134648] [Opticks::configure@2022]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
    2019-09-15 13:42:22.350 ERROR [134648] [OpticksResource::init@243] OpticksResource::init
    2019-09-15 13:42:22.350 ERROR [134648] [OpticksResource::readG4Environment@503]  MISSING inipath /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    2019-09-15 13:42:22.350 ERROR [134648] [OpticksResource::readOpticksEnvironment@518]  inipath /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/opticksdata/config/opticksdata.ini
    2019-09-15 13:42:22.350 ERROR [134648] [OpticksResource::readOpticksEnvironment@527]  MISSING inipath /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/opticksdata/config/opticksdata.ini (create it with bash functions: opticksdata-;opticksdata-export-ini ) 
    2019-09-15 13:42:22.350 ERROR [134648] [OpticksResource::readEnvironment@581]  initial m_geokey OPTICKSDATA_DAEPATH_DYB
    2019-09-15 13:42:22.350 ERROR [134648] [OpticksResource::readEnvironment@587]  NO DAEPATH  geokey OPTICKSDATA_DAEPATH_DYB daepath NULL
    2019-09-15 13:42:22.350 ERROR [134648] [OpticksResource::idNameContains@205]  idname NULL 
    2019-09-15 13:42:22.350 ERROR [134648] [OpticksResource::idNameContains@205]  idname NULL 
    2019-09-15 13:42:22.350 ERROR [134648] [OpticksResource::idNameContains@205]  idname NULL 
    2019-09-15 13:42:22.350 ERROR [134648] [OpticksResource::idNameContains@205]  idname NULL 
    2019-09-15 13:42:22.350 ERROR [134648] [OpticksResource::assignDetectorName@434]  m_detector other
    2019-09-15 13:42:22.351 ERROR [134648] [OpticksResource::initRunResultsDir@279] /home/blyth/local/opticks/results/OKTest/R0_cvd_1/20190915_134222
    2019-09-15 13:42:22.351 ERROR [134648] [OpticksResource::init@267] OpticksResource::init DONE
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_S_construct null not valid
    Aborted (core dumped)
    [simon@localhost ~]$ 

    (gdb) bt
    #0  0x00007fffeaf48207 in raise () from /lib64/libc.so.6
    #1  0x00007fffeaf498f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffeb8577d5 in __gnu_cxx::__verbose_terminate_handler() () from /lib64/libstdc++.so.6
    #3  0x00007fffeb855746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00007fffeb855773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00007fffeb855993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x00007fffeb8aa597 in std::__throw_logic_error(char const*) () from /lib64/libstdc++.so.6
    #7  0x00007ffff7bd4891 in std::string::_S_construct<char const*> (__beg=0x0, __end=0xffffffffffffffff <Address 0xffffffffffffffff out of bounds>, __a=...) at /usr/include/c++/4.8.2/bits/basic_string.tcc:133
    #8  0x00007fffeb8b66d8 in std::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&) () from /lib64/libstdc++.so.6
    #9  0x00007ffff2ef5606 in Opticks::initResource (this=0x626880) at /home/blyth/opticks/optickscore/Opticks.cc:711
    #10 0x00007ffff2efa63e in Opticks::configure (this=0x626880) at /home/blyth/opticks/optickscore/Opticks.cc:2028
    #11 0x00007ffff64e63cf in OpticksHub::configure (this=0x640840) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:307
    #12 0x00007ffff64e5cf2 in OpticksHub::init (this=0x640840) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:237
    #13 0x00007ffff64e5b3f in OpticksHub::OpticksHub (this=0x640840, ok=0x626880) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:217
    #14 0x00007ffff7bd59cf in OKMgr::OKMgr (this=0x7fffffffd810, argc=1, argv=0x7fffffffd988, argforced=0x0) at /home/blyth/opticks/ok/OKMgr.cc:54
    #15 0x0000000000402ead in main (argc=1, argv=0x7fffffffd988) at /home/blyth/opticks/ok/tests/OKTest.cc:32
    (gdb) 


Add OPTICKS_LEGACY_GEOMETRY_ENABLED envvar Opticks::IsLegacyGeometyEnabled() without which "--envkey" is default
--------------------------------------------------------------------------------------------------------------------

Without the envvar the key based geometry setup is on by default and there is no need 
for the "--envkey" option.  


::

   BOpticksResource::IsLegacyGeometyEnabled 
   Opticks::IsLegacyGeometyEnabled 
        envvar check  

   Opticks::envkey   
        require the key to be setup in non-legacy    

Doing these contortions to keep tests passing in legacy approach while being able 
to work to fix test fails in non-legacy approach.

With the envvar (legacy enabled) get 3/412 known fails without it get 10 fails (hmm was expecting more, not just +7)::

    FAILS:  10  / 412   :  Mon Sep 16 12:25:32 2019   

      13 /31  Test #13 : OpticksCoreTest.OpticksTwoTest                Child aborted***Exception:     0.08   
      11 /18  Test #11 : ExtG4Test.X4PhysicalVolumeTest                Child aborted***Exception:     0.15   
      12 /18  Test #12 : ExtG4Test.X4PhysicalVolume2Test               Child aborted***Exception:     0.15   

                        UNEXPECTED double setting of key 
                           BOpticksKey::SetKey (spec=0x4064c8 "CX4GDMLTest.X4PhysicalVolume.World0xc15cfc0_PV.27c39be4e46a36ea28a3c4da52522c9e") at /home/blyth/opticks/boostrap/BOpticksKey.cc:59

                        Change behaviour : 1st SetKey wins, subsequent are ignored with a warning 


      3  /3   Test #3  : AssimpRapTest.AssimpGGeoTest                  Child aborted***Exception:     0.09  
      3  /3   Test #3  : OpticksGeoTest.OpenMeshRapTest                Child aborted***Exception:     0.08   
                        
                        same issue : assert on NULL path from getDAEPath()  
                        EXPECTED  : AssimpGGeo and OpenMeshRap will be removed in non-legacy 


      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.92   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.20   

                        known OptiX 600 torus issue

      18 /24  Test #18 : OptiXRapTest.interpolationTest                ***Failed                      7.85   

                        huh : did not reproduce fail from commandline


      22 /34  Test #22 : CFG4Test.CGenstepCollectorTest                Child aborted***Exception:     1.17   

                        CGenstepCollectorTest: /home/blyth/opticks/npy/NLookup.cpp:186: void NLookup::close(const char*): Assertion `m_alabel && m_blabel' failed.    

      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      11.69 
     
                        EXPECTED analysis RC fail from scattering as WITH_LOGDOUBLE is commented


CGenstepCollectorTest
~~~~~~~~~~~~~~~~~~~~~~~~

::

    (gdb) bt
    #0  0x00007fffe4e7f207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe4e808f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe4e78026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe4e780d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffec89d18c in NLookup::close (this=0x6afb20, msg=0x407a40 "NLookup::close") at /home/blyth/opticks/npy/NLookup.cpp:186
    #5  0x000000000040469c in main (argc=1, argv=0x7fffffffda18) at /home/blyth/opticks/cfg4/tests/CGenstepCollectorTest.cc:65
    (gdb) f 5
    #5  0x000000000040469c in main (argc=1, argv=0x7fffffffda18) at /home/blyth/opticks/cfg4/tests/CGenstepCollectorTest.cc:65
    65      lookup->close(); 
    (gdb) f 4
    #4  0x00007fffec89d18c in NLookup::close (this=0x6afb20, msg=0x407a40 "NLookup::close") at /home/blyth/opticks/npy/NLookup.cpp:186
    186     assert(m_alabel && m_blabel) ; // have to setA and setB before close
    (gdb) p m_alabel
    $1 = 0x0
    (gdb) p m_blabel
    $2 = 0x5b0c270 "GGeo::setupLookup/m_bndlib"
    (gdb) 




readG4Environment
--------------------

* hmm readG4Environment, move to CFG4 ? as OKTest doesnt depend on Geant4

  * X4 is lowest level proj that depends on Geant4, but there is no convenient place there
  * made OpticksResource::SetupG4Environment static 
  * added --localg4 option to get it invoked from CG4::init

* hmm this assumes my way of recording/setting Geant4 envvars : but when using the encumbent 
  Geant4 can just skip this ?

::

    [blyth@localhost config]$ cat geant4.ini 
    G4LEVELGAMMADATA=/home/blyth/local/opticks/externals/share/Geant4-10.4.2/data/PhotonEvaporation5.2
    G4NEUTRONXSDATA=/home/blyth/local/opticks/externals/share/Geant4-10.4.2/data/G4NEUTRONXS1.4
    G4LEDATA=/home/blyth/local/opticks/externals/share/Geant4-10.4.2/data/G4EMLOW7.3
    G4NEUTRONHPDATA=/home/blyth/local/opticks/externals/share/Geant4-10.4.2/data/G4NDL4.5
    G4ENSDFSTATEDATA=/home/blyth/local/opticks/externals/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2
    G4RADIOACTIVEDATA=/home/blyth/local/opticks/externals/share/Geant4-10.4.2/data/RadioactiveDecay5.2
    G4ABLADATA=/home/blyth/local/opticks/externals/share/Geant4-10.4.2/data/G4ABLA3.1
    G4PIIDATA=/home/blyth/local/opticks/externals/share/Geant4-10.4.2/data/G4PII1.3
    G4SAIDXSDATA=/home/blyth/local/opticks/externals/share/Geant4-10.4.2/data/G4SAIDDATA1.1
    G4REALSURFACEDATA=/home/blyth/local/opticks/externals/share/Geant4-10.4.2/data/RealSurface2.1.1

    [blyth@localhost config]$ pwd
    /home/blyth/local/opticks/externals/config



hmm dbg not so useful without source ? switch to opt for binary dist ?
-------------------------------------------------------------------------


eradicate opticksdata, DAE loading, AssimpRap, OpenMeshRap, DCS, ...
------------------------------------------------------------------

On the chopping block::
   
   OpenMesh
   OpenMeshRap
   OpticksAssimp
   AssimpRap
   ImplicitMesher 
   DualContouringSample

* disable opticksdata setup ?

* before remove the legacy reading from DAE functionality 
  need to create a 2nd gen dayabay near GDML file, 
  so can still use that for debugging in direct workflow
  going forward  

* hmm : that will take a while : just use "OKTest --envkey" to get further


how to keep tests passing with opticksdata gone ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* change docs/installation scripts to get every installation to run a geocache-create on a gdml file
  populating the OPTICKS_SHARED_CACHE_PREFIX as a step after installation

* switch to "--envkey" as default, and remove the option

* arrange a default OPTICKS_KEY, as the last found in the geocache : might need to rationalize geocache
  layout to allow this 
 
* at first order this might get most tests to pass




Trying without OPTICKS_LEGACY_GEOMETRY_ENABLED makes CG4Test very slow
--------------------------------------------------------------------------

* this is because current OPTICKS_KEY is pointing at JUNO geometry... and voxeling takes ages
* need to configure plucking a lighter geometry like DYB near for test running
* also need a way to configure a default key when the envvar is not defined   

Connect to the test process::

   [blyth@localhost cfg4]$ gdb -p 167266

    (gdb) bt
    #0  0x00007ff8604acb6c in std::__uninitialized_copy<false>::__uninit_copy<std::move_iterator<HepGeom::Plane3D<double>*>, HepGeom::Plane3D<double>*> (__first=..., __last=..., __result=0xc0fc0d0) at /usr/include/c++/4.8.2/bits/stl_uninitialized.h:75
    #1  0x00007ff8604aca62 in std::uninitialized_copy<std::move_iterator<HepGeom::Plane3D<double>*>, HepGeom::Plane3D<double>*> (__first=..., __last=..., __result=0xc0fc0d0) at /usr/include/c++/4.8.2/bits/stl_uninitialized.h:117
    #2  0x00007ff8604ac80c in std::__uninitialized_copy_a<std::move_iterator<HepGeom::Plane3D<double>*>, HepGeom::Plane3D<double>*, HepGeom::Plane3D<double> > (__first=..., __last=..., __result=0xc0fc0d0) at /usr/include/c++/4.8.2/bits/stl_uninitialized.h:258
    #3  0x00007ff8604ac2e0 in std::__uninitialized_move_if_noexcept_a<HepGeom::Plane3D<double>*, HepGeom::Plane3D<double>*, std::allocator<HepGeom::Plane3D<double> > > (__first=0xc0fbf70, __last=0xc0fbff0, __result=0xc0fc0d0, __alloc=...)
        at /usr/include/c++/4.8.2/bits/stl_uninitialized.h:281
    #4  0x00007ff85a6ee366 in std::vector<HepGeom::Plane3D<double>, std::allocator<HepGeom::Plane3D<double> > >::_M_emplace_back_aux<HepGeom::Plane3D<double> >(HepGeom::Plane3D<double>&&) (this=0x7ffdafe8bd30) at /usr/include/c++/4.8.2/bits/vector.tcc:412
    #5  0x00007ff85a6ed7c9 in std::vector<HepGeom::Plane3D<double>, std::allocator<HepGeom::Plane3D<double> > >::emplace_back<HepGeom::Plane3D<double> >(HepGeom::Plane3D<double>&&) (this=0x7ffdafe8bd30) at /usr/include/c++/4.8.2/bits/vector.tcc:101
    #6  0x00007ff85a6ec658 in std::vector<HepGeom::Plane3D<double>, std::allocator<HepGeom::Plane3D<double> > >::push_back(HepGeom::Plane3D<double>&&) (this=0x7ffdafe8bd30, 
        __x=<unknown type in /home/blyth/local/opticks/externals/lib64/libG4geometry.so, CU 0x281cd2, DIE 0x29c783>) at /usr/include/c++/4.8.2/bits/stl_vector.h:920
    #7  0x00007ff85a6e678a in G4BoundingEnvelope::CreateListOfPlanes (this=0x7ffdafe8c110, baseA=std::vector of length 6, capacity 6 = {...}, baseB=std::vector of length 6, capacity 6 = {...}, pPlanes=std::vector of length 4, capacity 4 = {...})
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4BoundingEnvelope.cc:778
    #8  0x00007ff85a6e460d in G4BoundingEnvelope::CalculateExtent (this=0x7ffdafe8c110, pAxis=kYAxis, pVoxelLimits=..., pTransform3D=..., pMin=@0x7ffdafe8c3c8: 8.9999999999999999e+99, pMax=@0x7ffdafe8c3c0: -8.9999999999999999e+99)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4BoundingEnvelope.cc:547
    #9  0x00007ff85a80cde0 in G4Polycone::CalculateExtent (this=0x18e2b980, pAxis=kYAxis, pVoxelLimit=..., pTransform=..., pMin=@0x7ffdafe8c9d8: 8.9999999999999999e+99, pMax=@0x7ffdafe8c9d0: -8.9999999999999999e+99)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/solids/specific/src/G4Polycone.cc:695
    #10 0x00007ff85a70e61e in G4SmartVoxelHeader::BuildNodes (this=0xc0f8b60, pVolume=0x18e4d7c0, pLimits=..., pCandidates=0xc0e5040, pAxis=kYAxis) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:852
    #11 0x00007ff85a70d75f in G4SmartVoxelHeader::BuildVoxelsWithinLimits (this=0xc0f8b60, pVolume=0x18e4d7c0, pLimits=..., pCandidates=0xc0e5040) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:476
    #12 0x00007ff85a70c7fc in G4SmartVoxelHeader::G4SmartVoxelHeader (this=0xc0f8b60, pVolume=0x18e4d7c0, pLimits=..., pCandidates=0xc0e5040, pSlice=470) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:119
    #13 0x00007ff85a70f37c in G4SmartVoxelHeader::RefineNodes (this=0xc0bdbe0, pVolume=0x18e4d7c0, pLimits=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:1244
    #14 0x00007ff85a70dae7 in G4SmartVoxelHeader::BuildVoxelsWithinLimits (this=0xc0bdbe0, pVolume=0x18e4d7c0, pLimits=..., pCandidates=0xb5b5950) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:568
    #15 0x00007ff85a70c7fc in G4SmartVoxelHeader::G4SmartVoxelHeader (this=0xc0bdbe0, pVolume=0x18e4d7c0, pLimits=..., pCandidates=0xb5b5950, pSlice=61) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:119
    #16 0x00007ff85a70f37c in G4SmartVoxelHeader::RefineNodes (this=0xb25d240, pVolume=0x18e4d7c0, pLimits=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:1244
    #17 0x00007ff85a70dae7 in G4SmartVoxelHeader::BuildVoxelsWithinLimits (this=0xb25d240, pVolume=0x18e4d7c0, pLimits=..., pCandidates=0x7ffdafe8d420) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:568
    #18 0x00007ff85a70ccdb in G4SmartVoxelHeader::BuildVoxels (this=0xb25d240, pVolume=0x18e4d7c0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:258
    #19 0x00007ff85a70c70d in G4SmartVoxelHeader::G4SmartVoxelHeader (this=0xb25d240, pVolume=0x18e4d7c0, pSlice=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:82
    #20 0x00007ff85a6f9d2d in G4GeometryManager::BuildOptimisations (this=0xb252780, allOpts=true, verbose=false) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4GeometryManager.cc:200
    #21 0x00007ff85a6f9aa5 in G4GeometryManager::CloseGeometry (this=0xb252780, pOptimise=true, verbose=false, pVolume=0x0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4GeometryManager.cc:102
    #22 0x00007ff85e2e7589 in G4RunManagerKernel::ResetNavigator (this=0x1c071b0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManagerKernel.cc:757
    #23 0x00007ff85e2e73a6 in G4RunManagerKernel::RunInitialization (this=0x1c071b0, fakeRun=false) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManagerKernel.cc:699
    #24 0x00007ff85e2d7f69 in G4RunManager::RunInitialization (this=0x1c07220) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:313
    #25 0x00007ff85e2d7d0f in G4RunManager::BeamOn (this=0x1c07220, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:272
    #26 0x00007ff861a1a660 in CG4::propagate (this=0x70a63b0) at /home/blyth/opticks/cfg4/CG4.cc:369
    #27 0x00000000004047fa in main (argc=1, argv=0x7ffdafe90048) at /home/blyth/opticks/cfg4/tests/CG4Test.cc:71
    (gdb) c


Non-Legacy 6/412 FAILS, 3 expected, 2 can be igored : 1 CGenstepCollectorTest needs investigating
-----------------------------------------------------------------------------------------------------------

::

    unset OPTICKS_LEGACY_GEOMETRY_ENABLED  # in setup
    ini
    opticks-t 

::

    FAILS:  6   / 412   :  Mon Sep 16 13:55:37 2019   
      3  /3   Test #3  : AssimpRapTest.AssimpGGeoTest                  Child aborted***Exception:     0.08   
      3  /3   Test #3  : OpticksGeoTest.OpenMeshRapTest                Child aborted***Exception:     0.10   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.95   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.32   
      22 /34  Test #22 : CFG4Test.CGenstepCollectorTest                Child aborted***Exception:     1.14   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      11.86  

Also 2 tests that pass, but take far too long for unit tests. Need to get faster DYB geom working in non-legacy::

  7  /34  Test #7  : CFG4Test.CG4Test                              Passed                         1025.91 
  1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         1036.86 



Legacy running::

    unset OPTICKS_LEGACY_GEOMETRY_ENABLED  # in setup
    export OPTICKS_LEGACY_GEOMETRY_ENABLED 1
    ini
    opticks-t 



     7/34 Test  #7: CFG4Test.CG4Test ..........................   Passed   16.55 sec
     1/1 Test #1  : OKG4Test.OKG4Test ................   Passed   18.92 sec


    FAILS:  4   / 412   :  Mon Sep 16 14:19:08 2019   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.94   
      18 /24  Test #18 : OptiXRapTest.interpolationTest                ***Failed                      2.13   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.31   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      11.93  
    [blyth@localhost cfg4]$ 


Checking ctest.log::

   oxrap-bcd
   vi ctest.log

Failure from a python assert::

    116 2019-09-16 14:17:43.212 INFO  [278770] [interpolationTest::ana@178]  path /home/blyth/opticks/optixrap/tests/interpolationTest_interpol.py
    117 [2019-09-16 14:17:43,347] p278872 {legacy_init         :env.py    :185} WARNING  - ^[[33mlegacy_init : OPTICKS_KEY envvar deleted for legacy running, unset IDPATH to use direct_init^[[0m
    118 [2019-09-16 14:17:43,357] p278872 {__init__            :proplib.py:172} WARNING  - ^[[33mdirect data override^[[0m
    119 [2019-09-16 14:17:43,358] p278872 {<module>            :interpolationTest_interpol.py:59} INFO     - ^[[32m opath : Y : /home/blyth/local/opticks/tmp/interpolationTest/interpolationTest_interpol.npy ^[[0m
    120 [2019-09-16 14:17:43,358] p278872 {<module>            :interpolationTest_interpol.py:60} INFO     - ^[[32m cpath : Y : /home/blyth/local/opticks/tmp/interpolationTest/CInterpolationTest_interpol.npy ^[[0m
    121 Traceback (most recent call last):
    122   File "/home/blyth/opticks/optixrap/tests/interpolationTest_interpol.py", line 75, in <module>
    123     assert len(t) == len(c)
    124 AssertionError
    125 2019-09-16 14:17:43.393 INFO  [278770] [SSys::run@91] python /home/blyth/opticks/optixrap/tests/interpolationTest_interpol.py rc_raw : 256 rc : 1
    126 2019-09-16 14:17:43.393 ERROR [278770] [SSys::run@98] FAILED with  cmd python /home/blyth/opticks/optixrap/tests/interpolationTest_interpol.py RC 1
    127 2019-09-16 14:17:43.393 INFO  [278770] [interpolationTest::ana@180]  RC 1


* could an error due to flipping between geometries, which trips a consistency check ?

* running "interpolationTest" on commandline doesnt fail 

::

    unset OPTICKS_LEGACY_GEOMETRY_ENABLED
    interpolationTest      # fails again
    interpolationTest      # still fails 



Commented opticksdata hookup
------------------------------

Causes 89 fails... this was before switched to keyed setup as default. 

::

    FAILS:  89  / 412   :  Sun Sep 15 18:46:07 2019   
      2  /31  Test #2  : OpticksCoreTest.IndexerTest                   Child aborted***Exception:     0.09   
      8  /31  Test #8  : OpticksCoreTest.OpticksFlagsTest              Child aborted***Exception:     0.08   
      11 /31  Test #11 : OpticksCoreTest.OpticksCfg2Test               Child aborted***Exception:     0.07   
      12 /31  Test #12 : OpticksCoreTest.OpticksTest                   Child aborted***Exception:     0.08   
      13 /31  Test #13 : OpticksCoreTest.OpticksTwoTest                Child aborted***Exception:     0.08   
      14 /31  Test #14 : OpticksCoreTest.OpticksResourceTest           Child aborted***Exception:     0.08   
      19 /31  Test #19 : OpticksCoreTest.OK_PROFILE_Test               Child aborted***Exception:     0.08   
      20 /31  Test #20 : OpticksCoreTest.OpticksPrepareInstallCacheTest Child aborted***Exception:     0.08   
      21 /31  Test #21 : OpticksCoreTest.OpticksAnaTest                Child aborted***Exception:     0.07   
      22 /31  Test #22 : OpticksCoreTest.OpticksDbgTest                Child aborted***Exception:     0.07   
      24 /31  Test #24 : OpticksCoreTest.CompositionTest               Child aborted***Exception:     0.06   
      27 /31  Test #27 : OpticksCoreTest.EvtLoadTest                   Child aborted***Exception:     0.07   
      28 /31  Test #28 : OpticksCoreTest.OpticksEventAnaTest           Child aborted***Exception:     0.08   
      29 /31  Test #29 : OpticksCoreTest.OpticksEventCompareTest       Child aborted***Exception:     0.07   
      30 /31  Test #30 : OpticksCoreTest.OpticksEventDumpTest          Child aborted***Exception:     0.07   
      13 /53  Test #13 : GGeoTest.GScintillatorLibTest                 Child aborted***Exception:     0.08   
      15 /53  Test #15 : GGeoTest.GSourceLibTest                       Child aborted***Exception:     0.06   
      16 /53  Test #16 : GGeoTest.GBndLibTest                          Child aborted***Exception:     0.07   
      17 /53  Test #17 : GGeoTest.GBndLibInitTest                      Child aborted***Exception:     0.09   
      26 /53  Test #26 : GGeoTest.GItemIndex2Test                      Child aborted***Exception:     0.06   
      33 /53  Test #33 : GGeoTest.GPmtTest                             Child aborted***Exception:     0.06   
      34 /53  Test #34 : GGeoTest.BoundariesNPYTest                    Child aborted***Exception:     0.07   
      35 /53  Test #35 : GGeoTest.GAttrSeqTest                         Child aborted***Exception:     0.08   
      36 /53  Test #36 : GGeoTest.GBBoxMeshTest                        Child aborted***Exception:     0.08   
      38 /53  Test #38 : GGeoTest.GFlagsTest                           Child aborted***Exception:     0.06   
      39 /53  Test #39 : GGeoTest.GGeoLibTest                          Child aborted***Exception:     0.08   
      40 /53  Test #40 : GGeoTest.GGeoTest                             Child aborted***Exception:     0.08   
      41 /53  Test #41 : GGeoTest.GMakerTest                           Child aborted***Exception:     0.08   
      42 /53  Test #42 : GGeoTest.GMergedMeshTest                      Child aborted***Exception:     0.07   
      48 /53  Test #48 : GGeoTest.GSurfaceLibTest                      Child aborted***Exception:     0.07   
      50 /53  Test #50 : GGeoTest.NLookupTest                          Child aborted***Exception:     0.07   
      51 /53  Test #51 : GGeoTest.RecordsNPYTest                       Child aborted***Exception:     0.08   
      52 /53  Test #52 : GGeoTest.GSceneTest                           Child aborted***Exception:     0.08   
      1  /3   Test #1  : AssimpRapTest.AssimpRapTest                   Child aborted***Exception:     0.10   
      2  /3   Test #2  : AssimpRapTest.AssimpImporterTest              Child aborted***Exception:     0.08   
      3  /3   Test #3  : AssimpRapTest.AssimpGGeoTest                  Child aborted***Exception:     0.07   
      1  /3   Test #1  : OpticksGeoTest.OpticksGeoTest                 Child aborted***Exception:     0.09   
      2  /3   Test #2  : OpticksGeoTest.OpticksHubTest                 Child aborted***Exception:     0.09   
      3  /3   Test #3  : OpticksGeoTest.OpenMeshRapTest                Child aborted***Exception:     0.09   
      1  /17  Test #1  : ThrustRapTest.TCURANDTest                     Child aborted***Exception:     0.16   
      1  /24  Test #1  : OptiXRapTest.OContextCreateTest               Child aborted***Exception:     0.21   
      2  /24  Test #2  : OptiXRapTest.OScintillatorLibTest             Child aborted***Exception:     0.21   
      3  /24  Test #3  : OptiXRapTest.LTOOContextUploadDownloadTest    Child aborted***Exception:     0.21   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     0.26   
      5  /24  Test #5  : OptiXRapTest.bufferTest                       Child aborted***Exception:     0.20   
      6  /24  Test #6  : OptiXRapTest.textureTest                      Child aborted***Exception:     0.20   
      7  /24  Test #7  : OptiXRapTest.boundaryTest                     Child aborted***Exception:     0.19   
      8  /24  Test #8  : OptiXRapTest.boundaryLookupTest               Child aborted***Exception:     0.19   
      9  /24  Test #9  : OptiXRapTest.texTest                          Child aborted***Exception:     0.27   
      10 /24  Test #10 : OptiXRapTest.tex0Test                         Child aborted***Exception:     0.26   
      11 /24  Test #11 : OptiXRapTest.minimalTest                      Child aborted***Exception:     0.27   
      12 /24  Test #12 : OptiXRapTest.rayleighTest                     Child aborted***Exception:     0.19   
      13 /24  Test #13 : OptiXRapTest.writeBufferTest                  Child aborted***Exception:     0.20   
      14 /24  Test #14 : OptiXRapTest.writeBufferLowLevelTest          Child aborted***Exception:     0.27   
      15 /24  Test #15 : OptiXRapTest.redirectLogTest                  Child aborted***Exception:     0.28   
      16 /24  Test #16 : OptiXRapTest.downloadTest                     Child aborted***Exception:     0.19   
      17 /24  Test #17 : OptiXRapTest.eventTest                        Child aborted***Exception:     0.19   
      18 /24  Test #18 : OptiXRapTest.interpolationTest                Child aborted***Exception:     0.19   
      20 /24  Test #20 : OptiXRapTest.intersectAnalyticTest.iaDummyTest Child aborted***Exception:     0.25   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     0.10   
      22 /24  Test #22 : OptiXRapTest.intersectAnalyticTest.iaSphereTest Child aborted***Exception:     0.09   
      23 /24  Test #23 : OptiXRapTest.intersectAnalyticTest.iaConeTest Child aborted***Exception:     0.09   
      24 /24  Test #24 : OptiXRapTest.intersectAnalyticTest.iaConvexpolyhedronTest Child aborted***Exception:     0.09   
      1  /5   Test #1  : OKOPTest.OpIndexerTest                        Child aborted***Exception:     0.18   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     0.19   
      3  /5   Test #3  : OKOPTest.dirtyBufferTest                      Child aborted***Exception:     0.21   
      4  /5   Test #4  : OKOPTest.compactionTest                       Child aborted***Exception:     0.19   
      5  /5   Test #5  : OKOPTest.OpSnapTest                           Child aborted***Exception:     0.18   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     0.18   
      3  /5   Test #3  : OKTest.OTracerTest                            Child aborted***Exception:     0.20   
      5  /5   Test #5  : OKTest.TrivialTest                            Child aborted***Exception:     0.20   
      3  /18  Test #3  : ExtG4Test.X4SolidTest                         Child aborted***Exception:     0.14   
      10 /18  Test #10 : ExtG4Test.X4MaterialTableTest                 Child aborted***Exception:     0.16   
      16 /18  Test #16 : ExtG4Test.X4CSGTest                           Child aborted***Exception:     0.13   
      1  /34  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     0.26   
      2  /34  Test #2  : CFG4Test.CMaterialTest                        Child aborted***Exception:     0.26   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     0.26   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     0.25   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     0.26   
      7  /34  Test #7  : CFG4Test.CG4Test                              Child aborted***Exception:     0.26   
      22 /34  Test #22 : CFG4Test.CGenstepCollectorTest                Child aborted***Exception:     1.12   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     0.27   
      25 /34  Test #25 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     0.53   
      28 /34  Test #28 : CFG4Test.CPhotonTest                          Child aborted***Exception:     0.24   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    Child aborted***Exception:     0.26   
      32 /34  Test #32 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     0.24   
      33 /34  Test #33 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     0.27   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     0.30   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      3.52   
    [blyth@localhost opticks]$ 






OKTest --envkey with OPTICKS_SHARED_CACHE_PREFIX
------------------------------------------------------

Home directories always give permissions problems so try::

    [blyth@localhost .opticks]$ cp -r geocache /cvmfs/opticks.ihep.ac.cn/ok/shared/
    [blyth@localhost .opticks]$ cp -r rngcache /cvmfs/opticks.ihep.ac.cn/ok/shared/

And set::

    export OPTICKS_SHARED_CACHE_PREFIX=/cvmfs/opticks.ihep.ac.cn/ok/shared

::

    gdb --args OKTest --envkey

::

    2019-09-15 15:49:22.056 INFO  [359355] [OpticksGen::targetGenstep@328] setting frame 0 Id
    terminate called after throwing an instance of 'boost::filesystem::filesystem_error'
      what():  boost::filesystem::status: Permission denied: "/home/blyth/local/opticks/gl"

    Program received signal SIGABRT, Aborted.
    0x00007fffeaf48207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffeaf48207 in raise () from /lib64/libc.so.6
    #1  0x00007fffeaf498f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffeb8577d5 in __gnu_cxx::__verbose_terminate_handler() () from /lib64/libstdc++.so.6
    #3  0x00007fffeb855746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00007fffeb855773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00007fffeb855993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x00007ffff1f4d01f in boost::filesystem::detail::status(boost::filesystem::path const&, boost::system::error_code*) () from /lib64/libboost_filesystem-mt.so.1.53.0
    #7  0x00007ffff24518c8 in boost::filesystem::exists (p=...) at /usr/include/boost/filesystem/operations.hpp:289
    #8  0x00007ffff2498a7c in BFile::preparePath (dir_=0x5a40ca0 "/home/blyth/local/opticks/gl", name=0x7ffff7787a21 "dynamic.h", create=true) at /home/blyth/opticks/boostrap/BFile.cc:726
    #9  0x00007ffff24c94fd in BDynamicDefine::write (this=0x5a41fc0, dir=0x5a40ca0 "/home/blyth/local/opticks/gl", name=0x7ffff7787a21 "dynamic.h") at /home/blyth/opticks/boostrap/BDynamicDefine.cc:47
    #10 0x00007ffff7767fbe in Scene::write (this=0x5a44680, dd=0x5a41fc0) at /home/blyth/opticks/oglrap/Scene.cc:176
    #11 0x00007ffff7780cb4 in OpticksViz::prepareScene (this=0x5a40690, rendermode=0x0) at /home/blyth/opticks/oglrap/OpticksViz.cc:318
    #12 0x00007ffff7780238 in OpticksViz::init (this=0x5a40690) at /home/blyth/opticks/oglrap/OpticksViz.cc:176
    #13 0x00007ffff777fe01 in OpticksViz::OpticksViz (this=0x5a40690, hub=0x640c50, idx=0x5a40670, immediate=true) at /home/blyth/opticks/oglrap/OpticksViz.cc:133
    #14 0x00007ffff7bd5a8e in OKMgr::OKMgr (this=0x7fffffffd810, argc=2, argv=0x7fffffffd988, argforced=0x0) at /home/blyth/opticks/ok/OKMgr.cc:59
    #15 0x0000000000402ead in main (argc=2, argv=0x7fffffffd988) at /home/blyth/opticks/ok/tests/OKTest.cc:32
    (gdb) 

::

     174 void Scene::write(BDynamicDefine* dd)
     175 {
     176     dd->write( m_shader_dynamic_dir, "dynamic.h" );
     177 }
     178 

::

    [simon@localhost ~]$ Scene=ERROR OKTest --envkey
    ...
    2019-09-15 15:54:34.653 ERROR [367407] [OpticksGen::makeTorchstep@396]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0
    2019-09-15 15:54:34.653 INFO  [367407] [OpticksGen::targetGenstep@328] setting frame 0 Id
    2019-09-15 15:54:34.656 ERROR [367407] [Scene::init@149]  OGLRAP_INSTALL_PREFIX /home/blyth/local/opticks OGLRAP_SHADER_DIR /home/blyth/local/opticks/gl OGLRAP_SHADER_INCL_PATH /home/blyth/local/opticks/gl OGLRAP_SHADER_DYNAMIC_DIR /home/blyth/local/opticks/gl
    2019-09-15 15:54:34.657 ERROR [367407] [Scene::write@173] shader_dynamic_dir /home/blyth/local/opticks/gl
    terminate called after throwing an instance of 'boost::filesystem::filesystem_error'
      what():  boost::filesystem::status: Permission denied: "/home/blyth/local/opticks/gl"
    Aborted (core dumped)

::

     15 #[=[       
     16 Note that the OGLRap_Config.hh generated header is not installed,
     17 as it is just used internally by Scene.cc direct from inc in the build dir.
     18 This is how the shader sources are found at runtime.
     19 #]=]
     20 
     21 set(OGLRAP_GENERATED_HEADER OGLRap_Config.hh)
     22 set(OGLRAP_INSTALL_PREFIX     "${CMAKE_INSTALL_PREFIX}")
     23 set(OGLRAP_SHADER_DIR         "${CMAKE_INSTALL_PREFIX}/gl")
     24 set(OGLRAP_SHADER_DYNAMIC_DIR "${CMAKE_INSTALL_PREFIX}/gl")
     25 set(OGLRAP_SHADER_INCL_PATH   "${CMAKE_INSTALL_PREFIX}/gl")
     26 configure_file( ${OGLRAP_GENERATED_HEADER}.in inc/${OGLRAP_GENERATED_HEADER} )
     27 


* rearranged Shader::init to get shader dir from OpticksResource::ShaderDir() rather than the compiled
  in dir which is wrong for non-source running 

* TODO: investigate more, suspect only working due to uncontrolled write of the dynamic.h that happened to
  get collected into the distribution includes 



Hmm OpticksProfile::save trying to write into geocache
--------------------------------------------------------

::

    [simon@localhost ~]$ Scene=ERROR gdb --args OKTest --envkey
    ...
    2019-09-15 16:34:42.091 INFO  [445588] [OpEngine::propagate@157] ) propagator.launch 
    2019-09-15 16:34:42.130 INFO  [445588] [OpEngine::propagate@160] ]
    terminate called after throwing an instance of 'boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ini_parser::ini_parser_error> >'
      what():  /cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/OKTest/evt/g4live/torch/Time.ini: cannot open file
    
    Program received signal SIGABRT, Aborted.
    0x00007fffeaf48207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libX11-devel-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libdrm-2.4.91-3.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    ...
    #12 0x00007ffff2477f58 in BList<std::string, double>::save (li=0x6270d0, dir=0x6424c0 "/cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/OKTest/evt/g4live/torch", name=0x28e0ae28 "Time.ini")
        at /home/blyth/opticks/boostrap/BList.cc:52
    #13 0x00007ffff24daf6a in BTimes::save (this=0x6270d0, dir=0x6424c0 "/cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/OKTest/evt/g4live/torch") at /home/blyth/opticks/boostrap/BTimes.cc:122
    #14 0x00007ffff24dccde in BTimesTable::save (this=0x627040, dir=0x6424c0 "/cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/OKTest/evt/g4live/torch") at /home/blyth/opticks/boostrap/BTimesTable.cc:237
    #15 0x00007ffff2f20c23 in OpticksProfile::save (this=0x626f10, dir=0x6424c0 "/cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/OKTest/evt/g4live/torch")
        at /home/blyth/opticks/optickscore/OpticksProfile.cc:350
    #16 0x00007ffff2f20abd in OpticksProfile::save (this=0x626f10) at /home/blyth/opticks/optickscore/OpticksProfile.cc:334
    #17 0x00007ffff2ef4bf2 in Opticks::saveProfile (this=0x6268e0) at /home/blyth/opticks/optickscore/Opticks.cc:468
    #18 0x00007ffff2ef4c3c in Opticks::postpropagate (this=0x6268e0) at /home/blyth/opticks/optickscore/Opticks.cc:478
    #19 0x00007ffff7bd5ebb in OKMgr::propagate (this=0x7fffffffd800) at /home/blyth/opticks/ok/OKMgr.cc:123
    #20 0x0000000000402ebc in main (argc=2, argv=0x7fffffffd978) at /home/blyth/opticks/ok/tests/OKTest.cc:33
    (gdb) 



For non-restricted OKTest raise SIGINT at setDir::

    (gdb) bt
    #0  0x00007ffff0a7f49b in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff2f1fb9c in OpticksProfile::setDir (this=0x626e40, dir=0x5aa7750 "/home/blyth/.opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/OKTest/evt/g4live/torch") at /home/blyth/opticks/optickscore/OpticksProfile.cc:122
    #2  0x00007ffff2ef4c23 in Opticks::setProfileDir (this=0x626850, dir=0x5aa7750 "/home/blyth/.opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/OKTest/evt/g4live/torch") at /home/blyth/opticks/optickscore/Opticks.cc:464
    #3  0x00007ffff2efc34a in Opticks::postgeometry (this=0x626850) at /home/blyth/opticks/optickscore/Opticks.cc:2420
    #4  0x00007ffff2efbd84 in Opticks::setSpaceDomain (this=0x626850, x=0, y=0, z=0, w=60000) at /home/blyth/opticks/optickscore/Opticks.cc:2291
    #5  0x00007ffff2efb774 in Opticks::setSpaceDomain (this=0x626850, sd=...) at /home/blyth/opticks/optickscore/Opticks.cc:2269
    #6  0x00007ffff64e3de7 in OpticksAim::registerGeometry (this=0x653080, mm0=0x6db9d0) at /home/blyth/opticks/opticksgeo/OpticksAim.cc:60
    #7  0x00007ffff64e7ef4 in OpticksHub::registerGeometry (this=0x6409f0) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:639
    #8  0x00007ffff64e7640 in OpticksHub::loadGeometry (this=0x6409f0) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:559
    #9  0x00007ffff64e5e4e in OpticksHub::init (this=0x6409f0) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:253
    #10 0x00007ffff64e5b3f in OpticksHub::OpticksHub (this=0x6409f0, ok=0x626850) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:217
    #11 0x00007ffff7bd59cf in OKMgr::OKMgr (this=0x7fffffffd8b0, argc=2, argv=0x7fffffffda28, argforced=0x0) at /home/blyth/opticks/ok/OKMgr.cc:54
    #12 0x0000000000402ead in main (argc=2, argv=0x7fffffffda28) at /home/blyth/opticks/ok/tests/OKTest.cc:32
    (gdb) 


::

    2415 void Opticks::postgeometry()
    2416 {
    2417     configureDomains();
    2420     setProfileDir(getEventFold());
    2421 
    2422 }

    2604 const char* Opticks::getEventFold() const
    2605 {
    2606    return m_spec ? m_spec->getFold() : NULL ;
    2607 }


Changing OPTICKS_EVENT_BASE does not change this.



Flipping between users optix cache issue
------------------------------------------

::


    2019-09-15 17:08:19.276 INFO  [49933] [OContext::CheckDevices@204] 
    Device 0                      TITAN RTX ordinal 0 Compute Support: 7 5 Total Memory: 25364987904

    terminate called after throwing an instance of 'optix::Exception'
      what():  OptiX was unable to open the disk cache with sufficient privileges. Please make sure the database file is writeable by the current user.
    Aborted (core dumped)
    [blyth@localhost optickscore]$ l /var/tmp/
    total 0
    drwxrwxr--. 2 simon simon 62 Sep 15 16:31 OptixCache


::

    sudo rm -rf /var/tmp/OptixCache


    OPTIX_CACHE_PATH=/var/tmp/simon/OptiXCache OKTest



OpticksTest permissions checking existance of results_dir
-------------------------------------------------------------

::

    simon@localhost ~]$ gdb --args OpticksTest --envkey
    ...
                      geocache_dir :  Y :       /cvmfs/opticks.ihep.ac.cn/ok/shared/geocache
                      rngcache_dir :  Y :       /cvmfs/opticks.ihep.ac.cn/ok/shared/rngcache
                      runcache_dir :  N :                      /home/simon/.opticks/runcache
    terminate called after throwing an instance of 'boost::filesystem::filesystem_error'
      what():  boost::filesystem::status: Permission denied: "/home/blyth/local/opticks/results"
    
    Program received signal SIGABRT, Aborted.
    0x00007ffff44ed207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff44ed207 in raise () from /lib64/libc.so.6
    #1  0x00007ffff44ee8f8 in abort () from /lib64/libc.so.6
    #2  0x00007ffff4dfc7d5 in __gnu_cxx::__verbose_terminate_handler() () from /lib64/libstdc++.so.6
    #3  0x00007ffff4dfa746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00007ffff4dfa773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00007ffff4dfa993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x00007ffff628101f in boost::filesystem::detail::status(boost::filesystem::path const&, boost::system::error_code*) () from /lib64/libboost_filesystem-mt.so.1.53.0
    #7  0x00007ffff7086908 in boost::filesystem::exists (p=...) at /usr/include/boost/filesystem/operations.hpp:289
    #8  0x00007ffff70cb69f in BFile::ExistsDir (path=0x62dcc8 "/home/blyth/local/opticks/results", sub=0x0, name=0x0) at /home/blyth/opticks/boostrap/BFile.cc:347
    #9  0x00007ffff70f855e in BResource::dumpDirs (this=0x629540, msg=0x7ffff7b8c1ef "dumpDirs") at /home/blyth/opticks/boostrap/BResource.cc:223
    #10 0x00007ffff7b50f97 in OpticksResource::Summary (this=0x629850, msg=0x407200 "Opticks::Summary") at /home/blyth/opticks/optickscore/OpticksResource.cc:731
    #11 0x00007ffff7b30653 in Opticks::Summary (this=0x7fffffffd370, msg=0x407200 "Opticks::Summary") at /home/blyth/opticks/optickscore/Opticks.cc:2238
    #12 0x0000000000403df9 in main (argc=2, argv=0x7fffffffd978) at /home/blyth/opticks/optickscore/tests/OpticksTest.cc:181
    (gdb) 


Setting the below avoids the error, but where does the default come from ?::

    export OPTICKS_RESULTS_PREFIX=$HOME/results_prefix

Curiosly unset and no longer getting the error.



Investigate where event paths inside geocache are coming from
-----------------------------------------------------------------

* from BOpticksResource::setupViaKey

::

    [simon@localhost ~]$ OpticksEventSpec=FATAL BOpticksEvent=FATAL BFile=FATAL BOpticksResource=FATAL OpticksTest --envkey
    ...
    2019-09-15 18:11:25.495 FATAL [194736] [BOpticksResource::setupViaKey@775]  evtbase from idpath /cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1
    ...
    2019-09-15 18:11:25.502 INFO  [194736] [Opticks::Summary@2262] Opticks::SummaryDONE
    2019-09-15 18:11:25.502 INFO  [194736] [main@190] OpticksTest::main aft configure
    2019-09-15 18:11:25.502 FATAL [194736] [BOpticksEvent::replace@145]  pfx OpticksTest top g4live sub torch tag NULL
    2019-09-15 18:11:25.502 FATAL [194736] [BFile::ResolveKey@241] replacing $OPTICKS_EVENT_BASE   evalue /cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1 evtbase /cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1
    2019-09-15 18:11:25.502 FATAL [194736] [BOpticksEvent::directory@117]  base0 $OPTICKS_EVENT_BASE/$0/evt/$1/$2 anno NULL base $OPTICKS_EVENT_BASE/OpticksTest/evt/g4live/torch dir /cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/OpticksTest/evt/g4live/torch
    2019-09-15 18:11:25.502 FATAL [194736] [OpticksEventSpec::formFold@180]  pfx OpticksTest top g4live sub torch dir /cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/OpticksTest/evt/g4live/torch
    2019-09-15 18:11:25.502 INFO  [194736] [test_getEventFold@172] /cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/OpticksTest/evt/g4live/torch
    [simon@localhost ~]$ 





OKTest now runs from binary dist with fairly minimal environment
-------------------------------------------------------------------

::

    [simon@localhost ~]$ cat .opticks_setup_minimal 
    # ~/.opticks_setup_minimal
    
    export OPTICKS_INSTALL_PREFIX=/cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg
    export PATH=${OPTICKS_INSTALL_PREFIX}/lib:$PATH
    
    export OPTICKS_SHARED_CACHE_PREFIX=/cvmfs/opticks.ihep.ac.cn/ok/shared
    
    unset OPTICKS_KEY
    export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce   ## geocache-j1808-v5-export 
    
    export OPTICKS_DEFAULT_INTEROP_CVD=1
    # cvd setting pointing at the GPU that is connected to the monitor, need on multi-gpu machines
    
    unset OPTICKS_RESULTS_PREFIX
    export OPTICKS_RESULTS_PREFIX=$HOME/results_prefix
    
    unset OPTICKS_EVENT_BASE
    export OPTICKS_EVENT_BASE=$HOME/event_base
    





Getting the tests to run from binary dist 
------------------------------------------------

Need the CTestTestfile.cmake from the build tree to be installed::

    [blyth@localhost build]$ find . -name 'CTestTestfile.cmake' 
    ./g4csg/tests/CTestTestfile.cmake
    ./g4csg/CTestTestfile.cmake
    ./y4csg/tests/CTestTestfile.cmake
    ./y4csg/CTestTestfile.cmake
    ./okconf/tests/CTestTestfile.cmake
    ./okconf/CTestTestfile.cmake
    ./sysrap/tests/CTestTestfile.cmake
    ./sysrap/CTestTestfile.cmake
    ./npy/tests/CTestTestfile.cmake
    ./npy/CTestTestfile.cmake
    ...


First get tests to run from outside build tree, so the executables are being plucked from the PATH
----------------------------------------------------------------------------------------------------

* :google:`CMake install CTestTestfile.cmake`

Created bin/CTestTestfile.py to do this.::

   [blyth@localhost ~]$  CTestTestfile.py $(opticks-bdir) --dest /tmp/tests
    remove dest tree /tmp/tests 
    Copying CTestTestfile.cmake files from buildtree /home/blyth/local/opticks/build into a new destination tree /tmp/tests 
    write testfile to /tmp/tests/CTestTestfile.cmake 

    blyth@localhost tests]$ ctest.sh 
    Mon Sep 16 20:03:05 CST 2019
    Test project /tmp/tests
            Start   1: OKConfTest.OKConfTest
      1/411 Test   #1: OKConfTest.OKConfTest .......................................   Passed    0.01 sec
            Start   2: SysRapTest.SOKConfTest
      2/411 Test   #2: SysRapTest.SOKConfTest ......................................   Passed    0.01 sec
            Start   3: SysRapTest.SArTest
      3/411 Test   #3: SysRapTest.SArTest ..........................................   Passed    0.01 sec
            Start   4: SysRapTest.SArgsTest

    ...

    99% tests passed, 3 tests failed out of 411

    Total Test time (real) = 137.42 sec

    The following tests FAILED:
        323 - OptiXRapTest.Roots3And4Test (Child aborted)
        340 - OptiXRapTest.intersectAnalyticTest.iaTorusTest (Child aborted)
        411 - IntegrationTests.tboolean.box (Failed)
    Errors while running CTest
    Mon Sep 16 20:05:22 CST 2019

* 3 known fails


Integrate this with okdist--
----------------------------------

* okdist bundles up things from the $(opticks-dir) into the tarball, 
  so "install" the tests into $(opticks-dir)/tests with okdist-install-tests
  and add "tests" to okdist.py



Running the tests from exploded tarball with "simon"
-------------------------------------------------------

* lack the ctest.sh wrapper : need to install 
* running ctest from non-writable dir just causes warnings
* get 75/411 fails 

  * all that use Geant4 "Failed" to find libs presumably, they are excluded from externals/lib64 
  * ~13 from OptiXCache permissions, the lower level ones not already using OContext::Create

::


    [simon@localhost ~]$ cd /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests

    [simon@localhost tests]$ ctest 
    Test project /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests
    Cannot create directory /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/Testing/Temporary
    Cannot create log file: LastTest.log
            Start   1: OKConfTest.OKConfTest
      1/411 Test   #1: OKConfTest.OKConfTest .......................................   Passed    0.00 sec
            Start   2: SysRapTest.SOKConfTest
      2/411 Test   #2: SysRapTest.SOKConfTest ......................................   Passed    0.01 sec
            Start   3: SysRapTest.SArTest
      3/411 Test   #3: SysRapTest.SArTest ..........................................   Passed    0.01 sec
            Start   4: SysRapTest.SArgsTest
    ...
    Unable to find executable: tboolean.sh
    411/411 Test #411: IntegrationTests.tboolean.box ...............................***Not Run   0.00 sec

    82% tests passed, 75 tests failed out of 411

    Total Test time (real) = 135.29 sec

    The following tests FAILED:
    Cannot create directory /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/Testing/Temporary
    Cannot create log file: LastTestsFailed.log

        159 - NPYTest.NLoadTest (Child aborted)
                  fail to load gensteps from an opticksdata path 
                     /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/opticksdata/gensteps/dayabay/cerenkov/1.npy   

        205 - YoctoGLRapTest.YOGTFTest (Child aborted)
                   permissions writing /tmp/YOGTFTest.gltf 

        261 - GGeoTest.GItemIndex2Test (Child aborted)
                   permissions creating dir /cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/MeshIndex

        282 - GGeoTest.GPropertyTest (SEGFAULT)
                   fail to load $OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/glass/schott/F2.npy 

        291 - AssimpRapTest.AssimpGGeoTest (Child aborted)
        295 - OpticksGeoTest.OpenMeshRapTest (Child aborted)
                   getDAEPath assertion


        322 - OptiXRapTest.LTOOContextUploadDownloadTest (Child aborted)
                   fail to load /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/opticksdata/gensteps/juno/cerenkov/1.npy

        321 - OptiXRapTest.OScintillatorLibTest (Child aborted)
        323 - OptiXRapTest.Roots3And4Test (Child aborted)
        328 - OptiXRapTest.texTest (Child aborted)
        329 - OptiXRapTest.tex0Test (Child aborted)
        330 - OptiXRapTest.minimalTest (Child aborted)
        333 - OptiXRapTest.writeBufferLowLevelTest (Child aborted)
        334 - OptiXRapTest.redirectLogTest (Child aborted)
        337 - OptiXRapTest.interpolationTest (Failed)
        339 - OptiXRapTest.intersectAnalyticTest.iaDummyTest (Child aborted)
        340 - OptiXRapTest.intersectAnalyticTest.iaTorusTest (Child aborted)
        341 - OptiXRapTest.intersectAnalyticTest.iaSphereTest (Child aborted)
        342 - OptiXRapTest.intersectAnalyticTest.iaConeTest (Child aborted)
        343 - OptiXRapTest.intersectAnalyticTest.iaConvexpolyhedronTest (Child aborted)

            terminate called after throwing an instance of 'optix::Exception'
              what():  OptiX was unable to open the disk cache with sufficient privileges. Please make sure the database file is writeable by the current user.
             HMM THIS SHOULD HAVE BEEN FIXED



        356 - ExtG4Test.X4Test (Failed)
        357 - ExtG4Test.X4EntityTest (Failed)
        358 - ExtG4Test.X4SolidTest (Failed)
        359 - ExtG4Test.X4SolidLoadTest (Failed)
        360 - ExtG4Test.X4MeshTest (Failed)
        361 - ExtG4Test.X4SolidExtentTest (Failed)
        362 - ExtG4Test.X4SolidListTest (Failed)
        363 - ExtG4Test.X4PhysicsVectorTest (Failed)
        364 - ExtG4Test.X4MaterialTest (Failed)
        365 - ExtG4Test.X4MaterialTableTest (Failed)
        366 - ExtG4Test.X4PhysicalVolumeTest (Failed)
        367 - ExtG4Test.X4PhysicalVolume2Test (Failed)
        368 - ExtG4Test.X4Transform3DTest (Failed)
        369 - ExtG4Test.X4AffineTransformTest (Failed)
        370 - ExtG4Test.X4ThreeVectorTest (Failed)
        371 - ExtG4Test.X4CSGTest (Failed)
        372 - ExtG4Test.X4PolyconeTest (Failed)
        373 - ExtG4Test.X4GDMLParserTest (Failed)
        374 - CFG4Test.CMaterialLibTest (Failed)
        375 - CFG4Test.CMaterialTest (Failed)
        376 - CFG4Test.CTestDetectorTest (Failed)
        377 - CFG4Test.CGDMLTest (Failed)
        378 - CFG4Test.CGDMLDetectorTest (Failed)
        379 - CFG4Test.CGeometryTest (Failed)
        380 - CFG4Test.CG4Test (Failed)
        381 - CFG4Test.G4MaterialTest (Failed)
        382 - CFG4Test.G4StringTest (Failed)
        383 - CFG4Test.G4SphereTest (Failed)
        384 - CFG4Test.CSolidTest (Failed)
        385 - CFG4Test.G4PhysicsOrderedFreeVectorTest (Failed)
        386 - CFG4Test.CVecTest (Failed)
        387 - CFG4Test.G4MaterialPropertiesTableTest (Failed)
        388 - CFG4Test.CMPTTest (Failed)
        389 - CFG4Test.G4UniformRandTest (Failed)
        390 - CFG4Test.G4PhysicalConstantsTest (Failed)
        391 - CFG4Test.EngineTest (Failed)
        392 - CFG4Test.EngineMinimalTest (Failed)
        393 - CFG4Test.G4BoxTest (Failed)
        394 - CFG4Test.G4ThreeVectorTest (Failed)
        395 - CFG4Test.CGenstepCollectorTest (Failed)
        396 - CFG4Test.CInterpolationTest (Failed)
        397 - CFG4Test.OpRayleighTest (Failed)
        398 - CFG4Test.CGROUPVELTest (Failed)
        399 - CFG4Test.CMakerTest (Failed)
        400 - CFG4Test.CTreeJUNOTest (Failed)
        401 - CFG4Test.CPhotonTest (Failed)
        402 - CFG4Test.CRandomEngineTest (Failed)
        403 - CFG4Test.CAlignEngineTest (Failed)
        404 - CFG4Test.CMixMaxRngTest (Failed)
        405 - CFG4Test.CCerenkovGeneratorTest (Failed)
        406 - CFG4Test.CGenstepSourceTest (Failed)
        407 - CFG4Test.C4FPEDetectionTest (Failed)
        408 - OKG4Test.OKG4Test (Failed)
        409 - G4OKTest.G4OKTest (Failed)

        411 - IntegrationTests.tboolean.box (Not Run)
    Errors while running CTest



optixrap : some not using OContext got OptiXCache permissions problem : added envvar setup
------------------------------------------------------------------------------------------------


1. OPTIX_CACHE added envvar setup to the low level ones  


optixrap : OptiXTest users were looking for PTX in build tree : moved them to get PTX from install tree
--------------------------------------------------------------------------------------------------------------

Several cannot find their PTX::

    simon@localhost ~]$ minimalTest
    2019-09-16 21:22:08.824 ERROR [67982] [OContext::SetupOptiXCachePathEnvvar@284] envvar OPTIX_CACHE_PATHnot defined setting it internally to /var/tmp/simon/OptiXCache
    2019-09-16 21:22:09.003 FATAL [67982] [OptiXTest::OptiXTest@61] /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/build/optixrap/tests/minimalTest_generated_minimalTest.cu.ptx
    2019-09-16 21:22:09.003 INFO  [67982] [OptiXTest::init@67] OptiXTest::init cu minimalTest.cu ptxpath /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/build/optixrap/tests/minimalTest_generated_minimalTest.cu.ptx raygen minimal exception exception
    terminate called after throwing an instance of 'optix::Exception'
      what():  File not found (Details: Function "RTresult _rtProgramCreateFromPTXFile(RTcontext, const char*, const char*, RTprogram_api**)" caught exception: File not found - /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/build/optixrap/tests/minimalTest_generated_minimalTest.cu.ptx)
    Aborted (core dumped)
    [simon@localhost ~]$ 


::

    58% tests passed, 10 tests failed out of 24

    Total Test time (real) =  31.95 sec

    The following tests FAILED:
    Cannot create directory /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/optixrap/Testing/Temporary
    Cannot create log file: LastTestsFailed.log
          3 - OptiXRapTest.LTOOContextUploadDownloadTest (Child aborted)
         18 - OptiXRapTest.interpolationTest (Failed)

          4 - OptiXRapTest.Roots3And4Test (Child aborted)
         11 - OptiXRapTest.minimalTest (Child aborted)
         15 - OptiXRapTest.redirectLogTest (Child aborted)
         20 - OptiXRapTest.intersectAnalyticTest.iaDummyTest (Child aborted)
         21 - OptiXRapTest.intersectAnalyticTest.iaTorusTest (Child aborted)
         22 - OptiXRapTest.intersectAnalyticTest.iaSphereTest (Child aborted)
         23 - OptiXRapTest.intersectAnalyticTest.iaConeTest (Child aborted)
         24 - OptiXRapTest.intersectAnalyticTest.iaConvexpolyhedronTest (Child aborted)

                    this lot failing to find PTX

::

    [blyth@localhost tests]$ grep -l OptiXTest *.cc
    downloadTest.cc
    intersectAnalyticTest.cc
    LTOOContextUploadDownloadTest.cc
    minimalTest.cc
    redirectLogTest.cc
    Roots3And4Test.cc
    writeBufferTest.cc




After fix those : down to 4/24 fails : 2 are torus expected 
-------------------------------------------------------------

::

    83% tests passed, 4 tests failed out of 24

    Total Test time (real) =  35.00 sec

    The following tests FAILED:
    Cannot create directory /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/optixrap/tests/Testing/Temporary
    Cannot create log file: LastTestsFailed.log
          3 - OptiXRapTest.LTOOContextUploadDownloadTest (Child aborted)
                  fails to load gensteps  

         18 - OptiXRapTest.interpolationTest (Failed)
                  failing in python

          4 - OptiXRapTest.Roots3And4Test (Child aborted)
         21 - OptiXRapTest.intersectAnalyticTest.iaTorusTest (Child aborted)
                  expected  




interpolationTest failing in python, binary.simon running
--------------------------------------------------------------

::

    2019-09-16 22:12:14.934 INFO  [156472] [interpolationTest::launch@165]  save  base $TMP/interpolationTest name interpolationTest_interpol.npy
    2019-09-16 22:12:14.936 INFO  [156472] [interpolationTest::ana@178]  path /tmp/simon/opticks/optixrap/tests/interpolationTest_interpol.py
    python: can't open file '/tmp/simon/opticks/optixrap/tests/interpolationTest_interpol.py': [Errno 2] No such file or directory
    2019-09-16 22:12:15.106 INFO  [156472] [SSys::run@91] python /tmp/simon/opticks/optixrap/tests/interpolationTest_interpol.py rc_raw : 512 rc : 2
    2019-09-16 22:12:15.106 ERROR [156472] [SSys::run@98] FAILED with  cmd python /tmp/simon/opticks/optixrap/tests/interpolationTest_interpol.py RC 2
    2019-09-16 22:12:15.106 INFO  [156472] [interpolationTest::ana@180]  RC 2






Issue : test evt paths not being prefixed : FIXED by allowing OPTICKS_EVENT_BASE envvar
-------------------------------------------------------------------------------------------

::

    BOpticksEvent=ERROR LV=box tboolean.sh --generateoverride 10000


::

    2019-09-16 19:33:02.590 INFO  [322026] [Opticks::setupTimeDomain@2381]  cfg.getTimeMaxThumb [--timemaxthumb] 6 cfg.getAnimTimeMax [--animtimemax] -1 cfg.getAnimTimeMax [--animtimemax] -1 speed_of_light (mm/ns) 300 extent (mm) 450 rule_of_thumb_timemax (ns) 9 u_timemax 9 u_animtimemax 9
    2019-09-16 19:33:02.590 ERROR [322026] [BOpticksEvent::replace@145]  pfx tboolean-box top tboolean-box sub torch tag NULL
    2019-09-16 19:33:02.590 ERROR [322026] [BOpticksEvent::directory@117]  base0 $OPTICKS_EVENT_BASE/$0/evt/$1/$2 anno NULL base $OPTICKS_EVENT_BASE/tboolean-box/evt/tboolean-box/torch dir /tboolean-box/evt/tboolean-box/torch
    2019-09-16 19:33:02.590 FATAL [322026] [Opticks::setProfileDir@492]  dir /tboolean-box/evt/tboolean-box/torch
    2019-09-16 19:33:02.591 INFO  [322026] [OpticksHub::loadGeometry@565] ]


    [blyth@localhost tests]$ echo $OPTICKS_EVENT_BASE
    /home/blyth/local/opticks/tmp



Switching to not exclude libG4 in the dist tarball : get to 21/411 fails
----------------------------------------------------------------------------

::

    95% tests passed, 21 tests failed out of 411

    Total Test time (real) = 144.59 sec

    The following tests FAILED:
    Cannot create directory /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/Testing/Temporary
    Cannot create log file: LastTestsFailed.log
        159 - NPYTest.NLoadTest (Child aborted)
        205 - YoctoGLRapTest.YOGTFTest (Child aborted)
        261 - GGeoTest.GItemIndex2Test (Child aborted)
        282 - GGeoTest.GPropertyTest (SEGFAULT)

        291 - AssimpRapTest.AssimpGGeoTest (Child aborted)
        295 - OpticksGeoTest.OpenMeshRapTest (Child aborted)


        322 - OptiXRapTest.LTOOContextUploadDownloadTest (Child aborted)

        323 - OptiXRapTest.Roots3And4Test (Child aborted)
        337 - OptiXRapTest.interpolationTest (Failed)
        340 - OptiXRapTest.intersectAnalyticTest.iaTorusTest (Child aborted)

        358 - ExtG4Test.X4SolidTest (Child aborted)
               /tmp permissions : FIXED with $TMP

        373 - ExtG4Test.X4GDMLParserTest (Child aborted)


        376 - CFG4Test.CTestDetectorTest (SEGFAULT)
        378 - CFG4Test.CGDMLDetectorTest (Child aborted)
        379 - CFG4Test.CGeometryTest (Child aborted)
        380 - CFG4Test.CG4Test (SEGFAULT)
        395 - CFG4Test.CGenstepCollectorTest (Child aborted)
        396 - CFG4Test.CInterpolationTest (SEGFAULT)
        402 - CFG4Test.CRandomEngineTest (SEGFAULT)
        408 - OKG4Test.OKG4Test (SEGFAULT)
        411 - IntegrationTests.tboolean.box (Not Run)
    Errors while running CTest





17/411
----------

::

    96% tests passed, 17 tests failed out of 411

    Total Test time (real) = 138.00 sec

    The following tests FAILED:
    Cannot create directory /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/Testing/Temporary
    Cannot create log file: LastTestsFailed.log

        159 - NPYTest.NLoadTest (Child aborted)
        282 - GGeoTest.GPropertyTest (SEGFAULT)
        291 - AssimpRapTest.AssimpGGeoTest (Child aborted)
        295 - OpticksGeoTest.OpenMeshRapTest (Child aborted)

              All from missing opticksdata

        322 - OptiXRapTest.LTOOContextUploadDownloadTest (Child aborted)

              load failed for path [/cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/opticksdata/gensteps/juno/cerenkov/1.npy
              FIXED by switching to DummyGensteps 

        323 - OptiXRapTest.Roots3And4Test (Child aborted)
        340 - OptiXRapTest.intersectAnalyticTest.iaTorusTest (Child aborted)

              known 

        337 - OptiXRapTest.interpolationTest (Failed)

              assumes access to source tree for running python scripts  

        378 - CFG4Test.CGDMLDetectorTest (Child aborted)
        379 - CFG4Test.CGeometryTest (Child aborted)

              boost::filesystem::status: Permission denied: "/home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v5.gdml"

        376 - CFG4Test.CTestDetectorTest (SEGFAULT)
        380 - CFG4Test.CG4Test (SEGFAULT)
        396 - CFG4Test.CInterpolationTest (SEGFAULT)
        402 - CFG4Test.CRandomEngineTest (SEGFAULT)
        408 - OKG4Test.OKG4Test (SEGFAULT)

              G4 envvar 

        395 - CFG4Test.CGenstepCollectorTest (Child aborted)

              /home/blyth/opticks/npy/NLookup.cpp:186: void NLookup::close(const char): Assertion m_alabel && m_blabel failed. 

        411 - IntegrationTests.tboolean.box (Not Run)

              fails to find script tboolean.sh


    Errors while running CTest
    Tue Sep 17 11:15:49 CST 2019
    == okr-t : tdir /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests
    == okr-t : tlog /home/simon/okr-t.log



NLoadTest + GPropertyTest : opticksdata missing
-----------------------------------------------------

::

    load failed for path [/cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/opticksdata/gensteps/dayabay/cerenkov/1.npy]

    load FAILED for path $OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/glass/schott/F2.npy


interpolationTest + IntegrationTests.tboolean.box  : missing scripts
--------------------------------------------------------------------------

Python scripts need opticks/ana/... in PYTHONPATH::

    from opticks.ana.proplib import PropLib

* installing the python ana 




15/412
----------

::

    96% tests passed, 15 tests failed out of 412

    Total Test time (real) = 139.66 sec

    The following tests FAILED:
    Cannot create directory /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/Testing/Temporary
    Cannot create log file: LastTestsFailed.log

        159 - NPYTest.NLoadTest (SEGFAULT)
        283 - GGeoTest.GPropertyTest (SEGFAULT)

              opticksdata missing items

        292 - AssimpRapTest.AssimpGGeoTest (Child aborted)
        296 - OpticksGeoTest.OpenMeshRapTest (Child aborted)

              skipable : DAE access   

        324 - OptiXRapTest.Roots3And4Test (Child aborted)
        341 - OptiXRapTest.intersectAnalyticTest.iaTorusTest (Child aborted)

              known

        377 - CFG4Test.CTestDetectorTest (SEGFAULT)
        379 - CFG4Test.CGDMLDetectorTest (Child aborted)
        380 - CFG4Test.CGeometryTest (Child aborted)
        381 - CFG4Test.CG4Test (SEGFAULT)
        397 - CFG4Test.CInterpolationTest (SEGFAULT)
        403 - CFG4Test.CRandomEngineTest (SEGFAULT)

              fixed as shown below, by making the G4 data movable  

        396 - CFG4Test.CGenstepCollectorTest (Child aborted)

              lookup issue

        409 - OKG4Test.OKG4Test (SEGFAULT)

              now works, but too slowly like CG4Test 

        412 - IntegrationTests.tboolean.box (Not Run)

              need to install bash scripts as well as the python 


    Errors while running CTest
    == okr-t : tbeg Tue Sep 17 16:44:59 CST 2019
    == okr-t : tend Tue Sep 17 16:47:19 CST 2019
    == okr-t : tdir /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests
    == okr-t : tlog /home/simon/okr-t.log



hmm giving "simon" access to "blyth" kinda invalidates the test
-----------------------------------------------------------------

* for the test to be useful : everything must go via the tarball distribution, or be otherwise provided
* giving direct access to opticksdata from "blyth" breaks down the walls far too much 

::

    [blyth@localhost ~]$ chmod go+x .
    [blyth@localhost ~]$ chmod go-x .


tboolean.sh bash hookup : need to install these too
----------------------------------------------------------

* make ~/opticks/bin as propa proj with CMakeLists in order to install things like tboolean.sh 

* TODO: migrate ~/opticks/tests into ~/opticks/integration as om- machinery cannot handle tests/tests 


Geant4 failers : below fixes all except CGenstepCollectorTest : BUT CG4Test taking 17min for JUNO geometry
-------------------------------------------------------------------------------------------------------------

* add the G4 data files to okdist-- 
* with bin/envg4.py change externals/config/geant4.ini to use a token prefix instead of absolute path, so can work across installs
* add extra opticksdata path for the GDML to okdist.py
* modify Opticks::ExtractCacheMetaGDMLPath to determine the prefix and replace it with a token that 
  gets interpolated into the appropriate path for the current install  


::

    [simon@localhost x86_64-centos7-gcc48-geant4_10_04_p02-dbg]$ okr-t cfg4
    == okr-t : tdir /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/cfg4
    == okr-t : tlog /home/simon/okr-t-cfg4.log
    == okr-t : tbeg Tue Sep 17 20:43:51 CST 2019
    == okr-t : tlog /home/simon/okr-t-cfg4.log
    Test project /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/cfg4
    Cannot create directory /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/cfg4/Testing/Temporary
    Cannot create log file: LastTest.log
          Start  1: CFG4Test.CMaterialLibTest
     1/34 Test  #1: CFG4Test.CMaterialLibTest .................   Passed    0.57 sec
          Start  2: CFG4Test.CMaterialTest
     2/34 Test  #2: CFG4Test.CMaterialTest ....................   Passed    0.53 sec
          Start  3: CFG4Test.CTestDetectorTest
     ... 
     6/34 Test  #6: CFG4Test.CGeometryTest ....................   Passed   22.14 sec
          Start  7: CFG4Test.CG4Test
     7/34 Test  #7: CFG4Test.CG4Test ..........................   Passed  995.96 sec
          Start  8: CFG4Test.G4MaterialTest
     8/34 Test  #8: CFG4Test.G4MaterialTest ...................   Passed    0.08 sec
          Start  9: CFG4Test.G4StringTest
     ...
          Start 22: CFG4Test.CGenstepCollectorTest
    22/34 Test #22: CFG4Test.CGenstepCollectorTest ............Child aborted***Exception:   1.13 sec
     ...

    2019-09-17 21:01:36.773 INFO  [44460] [OpticksGen::targetGenstep@328] setting frame 0 Id
    CGenstepCollectorTest: /home/blyth/opticks/npy/NLookup.cpp:186: void NLookup::close(const char*): Assertion m_alabel && m_blabel failed.

          Start 23: CFG4Test.CInterpolationTest
    23/34 Test #23: CFG4Test.CInterpolationTest ...............   Passed   22.14 sec
          Start 24: CFG4Test.OpRayleighTest
    24/34 Test #24: CFG4Test.OpRayleighTest ...................   Passed    1.66 sec
          Start 25: CFG4Test.CGROUPVELTest
     
    97% tests passed, 1 tests failed out of 34

    Total Test time (real) = 1116.35 sec

    The following tests FAILED:
    Cannot create directory /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/cfg4/Testing/Temporary
    Cannot create log file: LastTestsFailed.log
         22 - CFG4Test.CGenstepCollectorTest (Child aborted)
    Errors while running CTest
    == okr-t : tbeg Tue Sep 17 20:43:51 CST 2019
    == okr-t : tend Tue Sep 17 21:02:27 CST 2019
    == okr-t : tdir /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/cfg4
    == okr-t : tlog /home/simon/okr-t-cfg4.log




JUNO gdml too big for quick testing
-------------------------------------------

::

    CG4Test
    OKG4Test 

How to export DYB into GDML ?::

    CGeometry::export_

There are options::

   --export
   --exportconfig $TMP   # the default   getExportConfig


::

    148 void CGeometry::export_()
    149 {
    150     bool expo = m_cfg->hasOpt("export");
    151     if(!expo) return ;
    152     //std::string expodir = m_cfg->getExportConfig();
    153 
    154     const char* expodir = "$TMP/CGeometry" ;
    155 
    156     if(BFile::ExistsDir(expodir))
    157     {
    158         BFile::RemoveDir(expodir);
    159         LOG(info) << "CGeometry::export_ removed " << expodir ;
    160     }
    161 
    162     BFile::CreateDir(expodir);
    163     m_detector->export_dae(expodir, "CGeometry.dae");
    164     m_detector->export_gdml(expodir, "CGeometry.gdml");
    165 }


::

    OKG4Test --export 
    ...
    2019-09-17 21:32:31.254 INFO  [96888] [CGDML::Export@65] export to /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml

::

    cd /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300
    cp /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml g4_00_CGeometry_export.gdml


::

    [blyth@localhost ~]$ opticksdata-dx
    /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export.gdml


::

    [blyth@localhost ~]$ geocache-dx-v0
    === o-cmdline-binary-match : --okx4
    === o-gdb-update : placeholder
    === o-main : /home/blyth/local/opticks/lib/OKX4Test --okx4 --g4codegen --deletegeocache --gdmlpath /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export.gdml -runfolder geocache-dx-v0 --runcomment export-dyb-near-for-regeneration ======= PWD /tmp/blyth/opticks/geocache-create- Tue Sep 17 21:55:08 CST 2019
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 0 /home/blyth/local/opticks/lib/OKX4Test
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 1 --okx4
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 2 --g4codegen
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 3 --deletegeocache
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 4 --gdmlpath
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 5 /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export.gdml
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 6 -runfolder
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 7 geocache-dx-v0
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 8 --runcomment
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 9 export-dyb-near-for-regeneration
    2019-09-17 21:55:08.928 INFO  [130758] [main@107]  csgskiplv NONE
    2019-09-17 21:55:08.928 INFO  [130758] [main@111]  parsing /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export.gdml
    G4GDML: Reading '/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export.gdml'...
    G4GDML: Reading definitions...

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : InvalidExpression
          issued by : G4GDMLEvaluator::DefineConstant()
    Redefinition of constant or variable: SCINTILLATIONYIELD
    *** Fatal Exception ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------


Indeed the below are duplicated::

   71     <constant name="SCINTILLATIONYIELD" value="10"/>
   72     <constant name="RESOLUTIONSCALE" value="1"/>
   73     <constant name="FASTTIMECONSTANT" value="3.6399998664856"/>
   74     <constant name="SLOWTIMECONSTANT" value="12.1999998092651"/>
   75     <constant name="YIELDRATIO" value="0.860000014305115"/>



Comment out and try again::

    opticksdata-dx-vi
    geocache-dx-v0

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : ReadError
          issued by : G4GDMLReadDefine::getMatrix()
    Matrix 'SCINTILLATIONYIELD' was not found!
    *** Fatal Exception ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------



TODO:

* contrast original GDML with the regenerated and exported
* check for property overrides in code



getting IntegrationTests.tboolean.box to run from install
------------------------------------------------------------

* rearrange bash hookup for relocatability
* make analytic and sysrap standard projs and install python modules from them


::

   ctest -R IntegrationTests.tboolean. --output-on-failure


This test runs commandline:: 

     LV=box tboolean.sh --generateoverride 10000


polyconfig not needed but other analytic ones like csg are::

    [simon@localhost ~]$ ini
    [simon@localhost ~]$ tboolean-
    [simon@localhost ~]$ tboolean-box-
    Traceback (most recent call last):
      File "<stdin>", line 4, in <module>
    ImportError: No module named analytic.polyconfig


Now it runs but fails to load test materials::

    opticksdata/refractiveindex/tmp/glass/schott/F2.npy
    opticksdata/refractiveindex/tmp/main/H2O/Hale.npy




8/412 but 2 are too slow
---------------------------

::

    ...
            Start 381: CFG4Test.CG4Test
    381/412 Test #381: CFG4Test.CG4Test ............................................   Passed  991.80 sec
    ...
            Start 409: OKG4Test.OKG4Test
    409/412 Test #409: OKG4Test.OKG4Test ...........................................   Passed  1019.75 sec


::

    98% tests passed, 8 tests failed out of 412

    Total Test time (real) = 2254.78 sec

    The following tests FAILED:
    Cannot create directory /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests/Testing/Temporary
    Cannot create log file: LastTestsFailed.log
        159 - NPYTest.NLoadTest (SEGFAULT)
        283 - GGeoTest.GPropertyTest (SEGFAULT)
        412 - IntegrationTests.tboolean.box (Failed)

              opticksdata  

        292 - AssimpRapTest.AssimpGGeoTest (Child aborted)
        296 - OpticksGeoTest.OpenMeshRapTest (Child aborted)

              will skip

        324 - OptiXRapTest.Roots3And4Test (Child aborted)
        341 - OptiXRapTest.intersectAnalyticTest.iaTorusTest (Child aborted)

              known 

        396 - CFG4Test.CGenstepCollectorTest (Child aborted)

              lookup? unknown problem


    Errors while running CTest
    == okr-t : tbeg Wed Sep 18 11:52:27 CST 2019
    == okr-t : tend Wed Sep 18 12:30:02 CST 2019
    == okr-t : tdir /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/tests
    == okr-t : tlog /home/simon/okr-t.log



Need opticksaux to take over from opticksdata
------------------------------------------------

* setup and populate git repo on bitbucket, opticksaux-
* opticksdata-migrate-to-opticksaux



Analysis issue with old numpy
--------------------------------

::

    In [1]: from opticks.ana.nbase import chi2, chi2_pvalue, ratio, count_unique_sorted

    In [2]: a = np.array( [], dtype=np.uint32 )

    In [3]: count_unique_sorted(a)
    Out[3]: array([], shape=(0, 2), dtype=uint64)

    In [4]: np.__version__
    Out[4]: '1.14.3'





::

    n [1]: a = np.array( [], dtype=np.uint32 )

    In [2]: a
    Out[2]: array([], dtype=uint32)

    In [4]: from opticks.ana.nbase import chi2, chi2_pvalue, ratio, count_unique_sorted

    In [7]: count_unique_sorted(a)
    ---------------------------------------------------------------------------
    IndexError                                Traceback (most recent call last)
    <ipython-input-7-9989b23863ee> in <module>()
    ----> 1 count_unique_sorted(a)

    /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/py/opticks/ana/nbase.py in count_unique_sorted(vals)
        127     vals = vals.astype(np.uint64)
        128     cu = count_unique(vals)
    --> 129     cu = cu[np.argsort(cu[:,1])[::-1]]  # descending frequency order
        130     return cu.astype(np.uint64)
        131 

    IndexError: index 0 is out of bounds for axis 0 with size 0
    > /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/py/opticks/ana/nbase.py(129)count_unique_sorted()
        128     cu = count_unique(vals)
    --> 129     cu = cu[np.argsort(cu[:,1])[::-1]]  # descending frequency order
        130     return cu.astype(np.uint64)

    ipdb> 

    In [8]: np.__version__
    Out[8]: '1.7.1'



