shakedown-running-from-binary-dist
=====================================

Context
----------

* :doc:`packaging-opticks-and-externals-for-use-on-gpu-cluster`
* :doc:`shakedown-running-from-expanded-binary-tarball` am earlier look at the same thing from April 2019


Workflow for testing binary dist
-----------------------------------

0. build Opticks, then create and explode the release binary distribution:: 

   okdist--

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



Commented opticksdata hookup
------------------------------

Causes 89 fails...

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




