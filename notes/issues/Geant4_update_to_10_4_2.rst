Geant4_update_to_10_4_2
=========================

* https://bitbucket.org/simoncblyth/opticks/commits/db6a93f072ed6ce5ef98556f5c2f1884d1de7c6f


DONE : Compilation
--------------------

* theParticleIterator
* needed to const_cast G4MaterialPropertiesTable

DONE : config : omitted opticks-buildtype : so had to rebuild
------------------------------------------------------------

Nuclear option as g4-configure omitted to setup Debug build.::

   g4-wipe
   g4--


FIXED : crash at used of X4PhysicsVector::Digest 
---------------------------------------------------

* X4PhysicsVector::Digest reproducibly SIGABRT on macOS, workaround get digest from converted GProperty 


FIXED : NULL dynamicparticle in ckm-
----------------------------------------

* getting NULL dynamicparticle : FIXED with a deep clean of all projs depending on G4 


WORKAROUND : polygonize crash : move from G4VisAtt to X4SolidExtent
---------------------------------------------------------------------

ckm- polygonize crash, and dont have debug symbols ?? for G4 

::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x48)
      * frame #0: 0x00000001029d02b9 libG4geometry.dylib`G4Box::GetExtent() const + 9
        frame #1: 0x000000010066ad2d libExtG4.dylib`X4Mesh::polygonize(this=0x00007ffeefbfc7c0) at X4Mesh.cc:128
        frame #2: 0x0000000100669fbf libExtG4.dylib`X4Mesh::init(this=0x00007ffeefbfc7c0) at X4Mesh.cc:93
        frame #3: 0x0000000100669f92 libExtG4.dylib`X4Mesh::X4Mesh(this=0x00007ffeefbfc7c0, solid=0x000000010e297bd0) at X4Mesh.cc:83
        frame #4: 0x0000000100669f0d libExtG4.dylib`X4Mesh::X4Mesh(this=0x00007ffeefbfc7c0, solid=0x000000010e297bd0) at X4Mesh.cc:82
        frame #5: 0x0000000100669ebc libExtG4.dylib`X4Mesh::Convert(solid=0x000000010e297bd0) at X4Mesh.cc:67

X4MeshTest has the same issue with GetExtent()::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x78)
      * frame #0: 0x0000000102491f09 libG4geometry.dylib`G4Sphere::GetExtent() const + 9
        frame #1: 0x000000010010dd2d libExtG4.dylib`X4Mesh::polygonize(this=0x0000000106814740) at X4Mesh.cc:128
        frame #2: 0x000000010010cfbf libExtG4.dylib`X4Mesh::init(this=0x0000000106814740) at X4Mesh.cc:93
        frame #3: 0x000000010010cf92 libExtG4.dylib`X4Mesh::X4Mesh(this=0x0000000106814740, solid=0x00000001068147d0) at X4Mesh.cc:83
        frame #4: 0x000000010010cf0d libExtG4.dylib`X4Mesh::X4Mesh(this=0x0000000106814740, solid=0x00000001068147d0) at X4Mesh.cc:82
        frame #5: 0x000000010000da58 X4MeshTest`main(argc=2, argv=0x00007ffeefbfea88) at X4MeshTest.cc:16
        frame #6: 0x00007fff533b2015 libdyld.dylib`start + 1
        frame #7: 0x00007fff533b2015 libdyld.dylib`start + 1
    (lldb) 


After replacing all use of G4VisExtent for X4SolidExtent all X4 tests pass::

    epsilon:extg4 blyth$ om-test
    === om-test-one : extg4           /Users/blyth/opticks/extg4                                   /usr/local/opticks/build/extg4                               
    Wed Aug 15 23:16:39 CST 2018
    Test project /usr/local/opticks/build/extg4
          Start  1: ExtG4Test.X4Test
     1/17 Test  #1: ExtG4Test.X4Test ..................   Passed    0.04 sec
          Start  2: ExtG4Test.X4EntityTest
     2/17 Test  #2: ExtG4Test.X4EntityTest ............   Passed    0.04 sec
          Start  3: ExtG4Test.X4SolidTest
     3/17 Test  #3: ExtG4Test.X4SolidTest .............   Passed    0.20 sec
          Start  4: ExtG4Test.X4SolidLoadTest
     4/17 Test  #4: ExtG4Test.X4SolidLoadTest .........   Passed    0.04 sec
          Start  5: ExtG4Test.X4MeshTest
     5/17 Test  #5: ExtG4Test.X4MeshTest ..............   Passed    0.04 sec
          Start  6: ExtG4Test.X4SolidExtentTest
     6/17 Test  #6: ExtG4Test.X4SolidExtentTest .......   Passed    0.04 sec
          Start  7: ExtG4Test.X4SolidListTest
     7/17 Test  #7: ExtG4Test.X4SolidListTest .........   Passed    0.04 sec
          Start  8: ExtG4Test.X4PhysicsVectorTest
     8/17 Test  #8: ExtG4Test.X4PhysicsVectorTest .....   Passed    0.04 sec
          Start  9: ExtG4Test.X4MaterialTest
     9/17 Test  #9: ExtG4Test.X4MaterialTest ..........   Passed    0.05 sec
          Start 10: ExtG4Test.X4MaterialTableTest
    10/17 Test #10: ExtG4Test.X4MaterialTableTest .....   Passed    0.05 sec
          Start 11: ExtG4Test.X4PhysicalVolumeTest
    11/17 Test #11: ExtG4Test.X4PhysicalVolumeTest ....   Passed    0.06 sec
          Start 12: ExtG4Test.X4PhysicalVolume2Test
    12/17 Test #12: ExtG4Test.X4PhysicalVolume2Test ...   Passed    0.06 sec
          Start 13: ExtG4Test.X4Transform3DTest
    13/17 Test #13: ExtG4Test.X4Transform3DTest .......   Passed    0.04 sec
          Start 14: ExtG4Test.X4AffineTransformTest
    14/17 Test #14: ExtG4Test.X4AffineTransformTest ...   Passed    0.04 sec
          Start 15: ExtG4Test.X4ThreeVectorTest
    15/17 Test #15: ExtG4Test.X4ThreeVectorTest .......   Passed    0.04 sec
          Start 16: ExtG4Test.X4CSGTest
    16/17 Test #16: ExtG4Test.X4CSGTest ...............   Passed    0.04 sec
          Start 17: ExtG4Test.X4PolyconeTest
    17/17 Test #17: ExtG4Test.X4PolyconeTest ..........   Passed    0.04 sec

    100% tests passed, 0 tests failed out of 17

    Total Test time (real) =   0.95 sec
    Wed Aug 15 23:16:40 CST 2018
    epsilon:extg4 blyth$ 



FIXED : CFG4 6 fails  : only one CInterpolationTest FAIL left, known from before G4 version hop
--------------------------------------------------------------------------------------------------

::

    74% tests passed, 6 tests failed out of 23

    Total Test time (real) =   7.46 sec

    The following tests FAILED:
          3 - CFG4Test.CTestDetectorTest (SEGFAULT)
          4 - CFG4Test.CGDMLDetectorTest (Child aborted)
          5 - CFG4Test.CGeometryTest (Child aborted)
          6 - CFG4Test.CG4Test (SEGFAULT)
         18 - CFG4Test.CInterpolationTest (SEGFAULT)
         23 - CFG4Test.CRandomEngineTest (SEGFAULT)
    Errors while running CTest
    Wed Aug 15 23:06:55 CST 2018
    epsilon:tests blyth$ 
    epsilon:tests blyth$ 


FIXED : ENSDFSTATE issue, had omitted to g4-export-ini to update the internal envvars
-------------------------------------------------------------------------------------------

The 4 SEGFAULT are all from the same cause::

    epsilon:tests blyth$ CRandomEngineTest
    2018-08-15 23:13:51.740 INFO  [1395722] [main@72] CRandomEngineTest
    2018-08-15 23:13:51.740 INFO  [1395722] [main@76]  pindex 0
      0 : CRandomEngineTest

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : PART70002
          issued by : G4NuclideTable
    5.609049e+17 is not valid indicator of G4Ions::G4FloatLevelBase. You may use a wrong version of ENSDFSTATE data. Please use G4ENSDFSTATE2.0 or later.
    *** Fatal Exception *** core dump ***
    Segmentation fault: 11
    epsilon:tests blyth$ 
    epsilon:tests blyth$ echo $ENSDFSTATE

    epsilon:tests blyth$ t g4-export
    g4-export is a function
    g4-export () 
    { 
        source $(g4-sh)
    }
    epsilon:tests blyth$ g4-sh
    /usr/local/opticks/externals/bin/geant4.sh
    epsilon:tests blyth$ vi /usr/local/opticks/externals/bin/geant4.sh


    epsilon:tests blyth$ env | grep G4   ## these external envvars get trumped by those from g4-ini

    G4LEVELGAMMADATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/PhotonEvaporation5.2
    G4NEUTRONXSDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4NEUTRONXS1.4
    G4LEDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4EMLOW7.3
    G4NEUTRONHPDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4NDL4.5
    G4ENSDFSTATEDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2
    G4RADIOACTIVEDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/RadioactiveDecay5.2
    G4ABLADATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4ABLA3.1
    G4PIIDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4PII1.3
    G4SAIDXSDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4SAIDDATA1.1
    G4REALSURFACEDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/RealSurface2.1.1

::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=EXC_I386_GPFLT)
      * frame #0: 0x00000001007e917c libG4vis_management.dylib`G4EventManager::GetTrackingManager(this=0x4e706574736e6547) const at G4EventManager.hh:165
        frame #1: 0x00000001007df339 libG4vis_management.dylib`G4RunManagerKernel::GetTrackingManager(this=0x000000010e7574d0) const at G4RunManagerKernel.hh:183
        frame #2: 0x0000000101e28ca2 libG4run.dylib`G4ExceptionHandler::DumpTrackInfo(this=0x000000010e7572f0) at G4ExceptionHandler.cc:151
        frame #3: 0x0000000101e283f4 libG4run.dylib`G4ExceptionHandler::Notify(this=0x000000010e7572f0, originOfException="G4NuclideTable", exceptionCode="PART70002", severity=FatalException, description="5.609049e+17 is not valid indicator of G4Ions::G4FloatLevelBase. You may use a wrong version of ENSDFSTATE data. Please use G4ENSDFSTATE2.0 or later.") at G4ExceptionHandler.cc:95
        frame #4: 0x0000000105b662fb libG4global.dylib`G4Exception(originOfException="G4NuclideTable", exceptionCode="PART70002", severity=FatalException, description="5.609049e+17 is not valid indicator of G4Ions::G4FloatLevelBase. You may use a wrong version of ENSDFSTATE data. Please use G4ENSDFSTATE2.0 or later.") at G4Exception.cc:52
        frame #5: 0x000000010516d4e3 libG4particles.dylib`G4NuclideTable::StripFloatLevelBase(this=0x0000000105246e70, sFLB=(std::__1::string = "5.609049e+17")) at G4NuclideTable.cc:395
        frame #6: 0x00000001051679e5 libG4particles.dylib`G4NuclideTable::GenerateNuclide(this=0x0000000105246e70) at G4NuclideTable.cc:228
        frame #7: 0x00000001051672f8 libG4particles.dylib`G4NuclideTable::G4NuclideTable(this=0x0000000105246e70) at G4NuclideTable.cc:74
        frame #8: 0x0000000105167085 libG4particles.dylib`G4NuclideTable::G4NuclideTable(this=0x0000000105246e70) at G4NuclideTable.cc:69
        frame #9: 0x0000000105167006 libG4particles.dylib`G4NuclideTable::GetInstance() at G4NuclideTable.cc:57
        frame #10: 0x000000010513d559 libG4particles.dylib`G4NuclideTable::GetNuclideTable() at G4NuclideTable.hh:73
        frame #11: 0x000000010513c484 libG4particles.dylib`G4IonTable::PrepareNuclideTable(this=0x000000010e757580) at G4IonTable.cc:1666
        frame #12: 0x000000010513c443 libG4particles.dylib`G4IonTable::G4IonTable(this=0x000000010e757580) at G4IonTable.cc:145
        frame #13: 0x000000010513c7e5 libG4particles.dylib`G4IonTable::G4IonTable(this=0x000000010e757580) at G4IonTable.cc:126
        frame #14: 0x0000000105181c79 libG4particles.dylib`G4ParticleTable::G4ParticleTable(this=0x0000000105246f60) at G4ParticleTable.cc:147
        frame #15: 0x00000001051812b5 libG4particles.dylib`G4ParticleTable::G4ParticleTable(this=0x0000000105246f60) at G4ParticleTable.cc:118
        frame #16: 0x00000001051811f6 libG4particles.dylib`G4ParticleTable::GetParticleTable() at G4ParticleTable.cc:99
        frame #17: 0x0000000101e662d6 libG4run.dylib`G4RunManagerKernel::G4RunManagerKernel(this=0x000000010e7574d0) at G4RunManagerKernel.cc:102
        frame #18: 0x0000000101e673e5 libG4run.dylib`G4RunManagerKernel::G4RunManagerKernel(this=0x000000010e7574d0) at G4RunManagerKernel.cc:88
        frame #19: 0x0000000101e44039 libG4run.dylib`G4RunManager::G4RunManager(this=0x000000010e757360) at G4RunManager.cc:105
        frame #20: 0x0000000101e449d5 libG4run.dylib`G4RunManager::G4RunManager(this=0x000000010e757360) at G4RunManager.cc:97
        frame #21: 0x000000010011c2ab libCFG4.dylib`CPhysics::CPhysics(this=0x000000010e757330, g4=0x00007ffeefbfe0f0) at CPhysics.cc:19
        frame #22: 0x000000010011c3ad libCFG4.dylib`CPhysics::CPhysics(this=0x000000010e757330, g4=0x00007ffeefbfe0f0) at CPhysics.cc:25
        frame #23: 0x00000001001e3945 libCFG4.dylib`CG4::CG4(this=0x00007ffeefbfe0f0, hub=0x00007ffeefbfe298) at CG4.cc:107
        frame #24: 0x00000001001e428d libCFG4.dylib`CG4::CG4(this=0x00007ffeefbfe0f0, hub=0x00007ffeefbfe298) at CG4.cc:128
        frame #25: 0x000000010000eea9 CInterpolationTest`main(argc=1, argv=0x00007ffeefbfea88) at CInterpolationTest.cc:57
        frame #26: 0x00007fff533b2015 libdyld.dylib`start + 1
        frame #27: 0x00007fff533b2015 libdyld.dylib`start + 1
    (lldb) 


::

    epsilon:extg4 blyth$ ll /usr/local/opticks/externals/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2/
    total 3504
    -rw-r--r--   1 blyth  staff      436 Sep 16  2016 README
    -rw-r--r--   1 blyth  staff  1785840 Sep  5  2017 ENSDFSTATE.dat
    -rw-r--r--   1 blyth  staff     1476 Sep 26  2017 History
    drwxr-xr-x  12 blyth  staff      384 Aug 15 18:40 ..
    drwxr-xr-x   5 blyth  staff      160 Aug 15 18:40 .
    epsilon:extg4 blyth$ ll /usr/local/opticks/externals/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2/ENSDFSTATE.dat 
    -rw-r--r--  1 blyth  staff  1785840 Sep  5  2017 /usr/local/opticks/externals/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2/ENSDFSTATE.dat
    epsilon:extg4 blyth$ 




FIXED : wrong version of internal g4env environment issue
----------------------------------------------------------------------

As I left office, recalled some internal envvar setup via ini::

    epsilon:opticks blyth$ g4-
    epsilon:opticks blyth$ t g4-export-ini
    g4-export-ini is a function
    g4-export-ini () 
    { 
        local msg="=== $FUNCNAME :";
        g4-export;
        local ini=$(g4-ini);
        local dir=$(dirname $ini);
        mkdir -p $dir;
        echo $msg writing G4 environment to $ini;
        env | grep G4 > $ini;
        cat $ini
    }

    epsilon:opticks blyth$ g4-ini
    /usr/local/opticks/externals/config/geant4.ini

    epsilon:opticks blyth$ opticks-find geant4.ini
    ./bin/oks.bash:    2016-07-07 13:48:50.187 WARN  [21116] [OpticksResource::readG4Environment@321] OpticksResource::readG4Environment MISSING FILE externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    ./bin/oks.bash:    /home/simonblyth/local/opticks/externals/config/geant4.ini
    ./bin/oks.bash:    === g4-export-ini : writing G4 environment to /home/simonblyth/local/opticks/externals/config/geant4.ini
    ./externals/g4.bash:	=== g4-export-ini : writing G4 environment to /home/blyth/local/opticks/externals/config/geant4.ini
    ./externals/g4.bash:g4-ini(){ echo $(opticks-prefix)/externals/config/geant4.ini ; }
    ./boostrap/tests/BFileTest.cc:    ss.push_back("$OPTICKS_INSTALL_PREFIX/externals/config/geant4.ini") ;
    ./boostrap/tests/BEnvTest.cc:    testIniLoad("$OPTICKS_INSTALL_PREFIX/externals/config/geant4.ini") ;
    ./boostrap/BOpticksResource.cc:const char* BOpticksResource::G4ENV_RELPATH = "externals/config/geant4.ini" ;
    epsilon:opticks blyth$ 

::

    096 const char* BOpticksResource::InstallPathG4ENV()
     97 {
     98     return InstallPath(G4ENV_RELPATH);
     99 }

    142 
    143     m_res->addPath("g4env_ini", InstallPathG4ENV() );
    144     m_res->addPath("okdata_ini", InstallPathOKDATA() );
    145 
    146 }


::

    epsilon:cfg4 blyth$ opticks-find g4env
    ./optickscore/OpticksResource.cc:       m_g4env(NULL),
    ./optickscore/OpticksResource.cc:    m_g4env = readIniEnvironment(inipath);
    ./optickscore/OpticksResource.cc:    if(m_g4env)
    ./optickscore/OpticksResource.cc:        m_g4env->setEnvironment();
    ./boostrap/BOpticksResource.cc:    m_res->addPath("g4env_ini", InstallPathG4ENV() );
    ./optickscore/OpticksResource.hh:       BEnv*          m_g4env ; 
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 


::

     214        BEnv*          m_g4env ;
     215        BEnv*          m_okenv ;


     488 void OpticksResource::readG4Environment()
     489 {
     490     // NB this relpath needs to match that in g4-;g4-export-ini
     491     //    it is relative to the install_prefix which 
     492     //    is canonically /usr/local/opticks
     493     //
     494     const char* inipath = InstallPathG4ENV();
     495 
     496     m_g4env = readIniEnvironment(inipath);
     497     if(m_g4env)
     498     {
     499         m_g4env->setEnvironment();
     500     }
     501     else
     502     {
     503         LOG(warning) << "OpticksResource::readG4Environment"
     504                      << " MISSING inipath " << inipath
     505                      << " (create it with bash functions: g4-;g4-export-ini ) "
     506                      ;
     507     }
     508 }


Dumping the internal environment, shows have omitted to update the geant4.ini::

    epsilon:boostrap blyth$ CTestDetectorTest --dumpenv
    2018-08-16 09:24:06.558 INFO  [1602290] [main@47] CTestDetectorTest
    ...
    2018-08-16 09:24:06.563 INFO  [1602290] [BEnv::dumpEnvironment@259] OPTICKSINSTALLPREFIX=/usr/local/opticks
    2018-08-16 09:24:06.563 INFO  [1602290] [BEnv::dumpEnvironment@259] G4ABLADATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4ABLA3.0
    2018-08-16 09:24:06.563 INFO  [1602290] [BEnv::dumpEnvironment@259] G4ENSDFSTATEDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4ENSDFSTATE1.2.1
    2018-08-16 09:24:06.563 INFO  [1602290] [BEnv::dumpEnvironment@259] G4LEDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4EMLOW6.48
    2018-08-16 09:24:06.563 INFO  [1602290] [BEnv::dumpEnvironment@259] G4LEVELGAMMADATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/PhotonEvaporation3.2
    2018-08-16 09:24:06.563 INFO  [1602290] [BEnv::dumpEnvironment@259] G4NEUTRONHPDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4NDL4.5
    2018-08-16 09:24:06.563 INFO  [1602290] [BEnv::dumpEnvironment@259] G4NEUTRONXSDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4NEUTRONXS1.4
    2018-08-16 09:24:06.563 INFO  [1602290] [BEnv::dumpEnvironment@259] G4PIIDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4PII1.3
    2018-08-16 09:24:06.563 INFO  [1602290] [BEnv::dumpEnvironment@259] G4RADIOACTIVEDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/RadioactiveDecay4.3.1
    2018-08-16 09:24:06.563 INFO  [1602290] [BEnv::dumpEnvironment@259] G4REALSURFACEDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/RealSurface1.0
    2018-08-16 09:24:06.563 INFO  [1602290] [BEnv::dumpEnvironment@259] G4SAIDXSDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4SAIDDATA1.1
    2018-08-16 09:24:06.563 INFO  [1602290] [OpticksHub::configure@240] OpticksHub::configure argc 2 argv[0] CTestDetectorTest m_gltf 0 is_tracer 0
    2018-08-16 09:24:06.563 ERROR [1602290] [OpticksHub::configure@272] ]


Update with::

    epsilon:issues blyth$ g4-export-ini  ## this is done by the standard g4--
    === g4-export-ini : writing G4 environment to /usr/local/opticks/externals/config/geant4.ini
    G4LEVELGAMMADATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/PhotonEvaporation5.2
    G4NEUTRONXSDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4NEUTRONXS1.4
    G4LEDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4EMLOW7.3
    G4NEUTRONHPDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4NDL4.5
    G4ENSDFSTATEDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2
    G4RADIOACTIVEDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/RadioactiveDecay5.2
    G4ABLADATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4ABLA3.1
    G4PIIDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4PII1.3
    G4SAIDXSDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4SAIDDATA1.1
    G4REALSURFACEDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/RealSurface2.1.1
    epsilon:issues blyth$ 




FIXED : 2 SIGABRT : from same assert related to skin surfaces : due to garbled GDML loaded LV names
------------------------------------------------------------------------------------------------------

::

    018-08-15 23:36:09.167 INFO  [1435854] [CDetector::attachSurfaces@277]  num_bs 0 num_sk 0
    2018-08-15 23:36:09.167 INFO  [1435854] [CDetector::attachSurfaces@289] [--dbgsurf] CDetector::attachSurfaces START
    2018-08-15 23:36:09.167 INFO  [1435854] [CSurfaceLib::convert@81] .
    2018-08-15 23:36:09.167 INFO  [1435854] [CSurfaceLib::convert@93] . num_surf 48
    Assertion failed: (lv), function makeSkinSurface, file /Users/blyth/opticks/cfg4/CSurfaceLib.cc, line 249.
    Process 92845 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff53502b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff53502b6e <+10>: jae    0x7fff53502b78            ; <+20>
        0x7fff53502b70 <+12>: movq   %rax, %rdi
        0x7fff53502b73 <+15>: jmp    0x7fff534f9b00            ; cerror_nocancel
        0x7fff53502b78 <+20>: retq   
    Target 0: (CGDMLDetectorTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff53502b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff536cd080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff5345e1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff534261ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001001ca1d6 libCFG4.dylib`CSurfaceLib::makeSkinSurface(this=0x000000010a732b40, surf=0x000000010a584c60, os=0x000000010ddd9c30) at CSurfaceLib.cc:249
        frame #5: 0x00000001001c8bbb libCFG4.dylib`CSurfaceLib::convert(this=0x000000010a732b40, detector=0x000000010a732a60, exclude_sensors=true) at CSurfaceLib.cc:124
        frame #6: 0x00000001001c149a libCFG4.dylib`CDetector::attachSurfaces(this=0x000000010a732a60) at CDetector.cc:292
        frame #7: 0x00000001001c5ef6 libCFG4.dylib`CGDMLDetector::init(this=0x000000010a732a60) at CGDMLDetector.cc:75
        frame #8: 0x00000001001c5bbb libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010a732a60, hub=0x00007ffeefbfe2e0, query=0x000000010b8105b0) at CGDMLDetector.cc:40
        frame #9: 0x00000001001c5f35 libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010a732a60, hub=0x00007ffeefbfe2e0, query=0x000000010b8105b0) at CGDMLDetector.cc:38
        frame #10: 0x000000010000f5a4 CGDMLDetectorTest`main(argc=1, argv=0x00007ffeefbfe6e0) at CGDMLDetectorTest.cc:51
        frame #11: 0x00007fff533b2015 libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 4
    frame #4: 0x00000001001ca1d6 libCFG4.dylib`CSurfaceLib::makeSkinSurface(this=0x000000010a732b40, surf=0x000000010a584c60, os=0x000000010ddd9c30) at CSurfaceLib.cc:249
       246 	              << " lv " << ( lv ? lv->GetName() : "NULL" )
       247 	              ;
       248 	
    -> 249 	    assert(lv) ;
       250 	
       251 	    G4LogicalSkinSurface* lss = new G4LogicalSkinSurface(name, const_cast<G4LogicalVolume*>(lv), os );
       252 	    return lss ;
    (lldb) p lvn
    (char *) $0 = 0x000000010ddd94d0 "/dd/Geometry/PoolDetails/lvNearTopCover0xc137060"
    (lldb) 
    (lldb) p sslv
    (std::__1::string) $1 = "__dd__Geometry__PoolDetails__lvNearTopCover0xc137060"
    (lldb) p name
    (const char *) $2 = 0x000000010a584c69 "NearPoolCoverSurface"
    (lldb) 

    (lldb) p m_detector->m_traverser->description()
    (std::__1::string) $5 = " numSelected 9068 bbox NBoundingBox low -23327.6914,-809820.6250,-12110.0000 high -9712.3086,-794399.3750,-2140.0000 ce -16520.0000,-802110.0000,-7125.0000,7710.6250 pvs.size 12230 lvs.size 12230"

    (lldb) p m_detector->m_traverser->m_lvm
    (std::__1::map<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, const G4LogicalVolume *, std::__1::less<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::allocator<std::__1::pair<const std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, const G4LogicalVolume *> > >) $6 = size=1012 {
      [0] = {
        first = ""
        second = 0x00000001128da2d0
      }
      [1] = {
        first = "\x02?
        second = 0x0000000111f01740
      }
      [2] = {
        first = "\x02?
        second = 0x0000000111f01740
      }
      [3] = {
        first = "\x04?
        second = 0x000000010a7fbc50
      }
      [4] = {
        first = "\x04?
        second = 0x000000010a7fbc50
      }
      [5] = {
        first = "\x06\x7f\n\x01\0\0\0A\0\0\0\0\0\0\0:\0\0\0\0\0\0\0\0O?\b%?\a\x02\0\0\0\0\0\0\0\0\0`\x84\x8cP\U0000007f\0\0?F?7?\x7f\0\0\x01ar, std\0\0\0\0\0\0\0\0\0J?7?\x7f\0\0"
        second = 0x000000010a7f0490
      }


Garbled names::

    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn 
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn 
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn ?
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn (?
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn 
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn 
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn 
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn 

Only LV::

    2018-08-16 09:38:56.027 INFO  [1617288] [CDetector::setTop@91] .
    2018-08-16 09:38:56.039 INFO  [1617288] [CTraverser::AncestorVisit@218]  pvn World0xc15cfc0_PV
    2018-08-16 09:38:56.039 INFO  [1617288] [CTraverser::AncestorVisit@219]  lvn ?c??
    2018-08-16 09:38:56.039 INFO  [1617288] [CTraverser::AncestorVisit@218]  pvn /dd/Structure/Sites/db-rock0xc15d358
    2018-08-16 09:38:56.039 INFO  [1617288] [CTraverser::AncestorVisit@219]  lvn ?c??
    2018-08-16 09:38:56.039 INFO  [1617288] [CTraverser::AncestorVisit@218]  pvn /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop0xbf89820
    2018-08-16 09:38:56.039 INFO  [1617288] [CTraverser::AncestorVisit@219]  lvn +@??
    2018-08-16 09:38:56.039 INFO  [1617288] [CTraverser::AncestorVisit@218]  pvn /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover0xc23f9b8



GDML read::

    30931     <volume name="World0xc15cfc0">
    30932       <materialref ref="/dd/Materials/Vacuum0xbf9fcc0"/>
    30933       <solidref ref="WorldBox0xc15cf40"/>
    30934       <physvol name="/dd/Structure/Sites/db-rock0xc15d358">
    30935         <volumeref ref="/dd/Geometry/Sites/lvNearSiteRock0xc030350"/>
    30936         <position name="/dd/Structure/Sites/db-rock0xc15d358_pos" unit="mm" x="-16519.9999999999" y="-802110" z="-2110"/>
    30937         <rotation name="/dd/Structure/Sites/db-rock0xc15d358_rot" unit="deg" x="0" y="0" z="-122.9"/>
    30938       </physvol>
    30939     </volume>

    // b G4GDMLReadStructure::VolumeRead(


    (lldb) b "G4GDMLReadStructure::VolumeRead"
    Breakpoint 1: where = libG4persistency.dylib`G4GDMLReadStructure::VolumeRead(xercesc_3_2::DOMElement const*) + 32 at G4GDMLReadStructure.cc:575, address = 0x000000000019b2a0
    (lldb) 

    (lldb) c
    Process 9126 resuming
    Process 9126 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x0000000100d17435 libG4persistency.dylib`G4GDMLReadStructure::VolumeRead(this=0x000000010f24d5e0, volumeElement=0x000000010f4b85e0) at G4GDMLReadStructure.cc:581
       578 	   
       579 	   XMLCh *name_attr = xercesc::XMLString::transcode("name");
       580 	   const G4String name = Transcode(volumeElement->getAttribute(name_attr));
    -> 581 	   xercesc::XMLString::release(&name_attr);
       582 	
       583 	   for (xercesc::DOMNode* iter = volumeElement->getFirstChild();
       584 	        iter != 0; iter = iter->getNextSibling())
    Target 0: (CTestDetectorTest) stopped.
    (lldb) p name
    (const G4String) $0 = (std::__1::string = "/dd/Geometry/PoolDetails/lvNearTopCover0xc137060")
    (lldb) 

    (lldb) c
    Process 9126 resuming
    Process 9126 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 4.1
        frame #0: 0x0000000100d17bd9 libG4persistency.dylib`G4GDMLReadStructure::VolumeRead(this=0x000000010f24d5e0, volumeElement=0x000000010f4b9028) at G4GDMLReadStructure.cc:609
       606 	   pMotherLogical = new G4LogicalVolume(solidPtr,materialPtr,
       607 	                                        GenerateName(name),0,0,0);
       608 	
    -> 609 	   if (!auxList.empty()) { auxMap[pMotherLogical] = auxList; }
       610 	
       611 	   Volume_contentRead(volumeElement);
       612 	}
    Target 0: (CTestDetectorTest) stopped.
    (lldb) p name
    (const G4String) $6 = (std::__1::string = "/dd/Geometry/RPC/lvRPCStrip0xc2213c0")
    (lldb) p pMotherLogical->GetName()
    (const G4String) $7 = (std::__1::string = "/dd/Geometry/RPC/lvRPCStrip0xc2213c0")
    (lldb) 





    572 void G4GDMLReadStructure::
    573 VolumeRead(const xercesc::DOMElement* const volumeElement)
    574 {
    575    G4VSolid* solidPtr = 0;
    576    G4Material* materialPtr = 0;
    577    G4GDMLAuxListType auxList;
    578 
    579    XMLCh *name_attr = xercesc::XMLString::transcode("name");
    580    const G4String name = Transcode(volumeElement->getAttribute(name_attr));
    581    xercesc::XMLString::release(&name_attr);
    582 


    289 void G4GDMLReadStructure::
    290 PhysvolRead(const xercesc::DOMElement* const physvolElement,
    291             G4AssemblyVolume* pAssembly)
    292 {
    293    G4String name;
    294    G4LogicalVolume* logvol = 0;
    295    G4AssemblyVolume* assembly = 0;
    296    G4ThreeVector position(0.0,0.0,0.0);
    297    G4ThreeVector rotation(0.0,0.0,0.0);
    298    G4ThreeVector scale(1.0,1.0,1.0);
    299    G4int copynumber = 0;
    300 
    301    const xercesc::DOMNamedNodeMap* const attributes
    302          = physvolElement->getAttributes();
    303    XMLSize_t attributeCount = attributes->getLength();
    304 
    305    for (XMLSize_t attribute_index=0;
    306         attribute_index<attributeCount; attribute_index++)
    307    { 
    308      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);
    309      
    310      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
    311        { continue; }
    312      
    313      const xercesc::DOMAttr* const attribute
    314            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);
    315      if (!attribute)
    316      { 
    317        G4Exception("G4GDMLReadStructure::PhysvolRead()",
    318                    "InvalidRead", FatalException, "No attribute found!");
    319        return;
    320      }
    321      const G4String attName = Transcode(attribute->getName());
    322      const G4String attValue = Transcode(attribute->getValue());
    323      
    324      if (attName=="name") { name = attValue; } 
    325      if (attName=="copynumber") { copynumber = eval.EvaluateInteger(attValue); }
    326    }
    327 




Getting rid of the PLOG.hh dangerous define of trace, fixes the mangled LV names.

::

    epsilon:cfg4 blyth$ om-test
    === om-test-one : cfg4            /Users/blyth/opticks/cfg4                                    /usr/local/opticks/build/cfg4                                
    Thu Aug 16 13:10:10 CST 2018
    Test project /usr/local/opticks/build/cfg4
          Start  1: CFG4Test.CMaterialLibTest
     1/23 Test  #1: CFG4Test.CMaterialLibTest .................   Passed    0.38 sec
          Start  2: CFG4Test.CMaterialTest
     2/23 Test  #2: CFG4Test.CMaterialTest ....................   Passed    0.30 sec
          Start  3: CFG4Test.CTestDetectorTest
     3/23 Test  #3: CFG4Test.CTestDetectorTest ................   Passed    1.69 sec
          Start  4: CFG4Test.CGDMLDetectorTest
     4/23 Test  #4: CFG4Test.CGDMLDetectorTest ................   Passed    1.52 sec
          Start  5: CFG4Test.CGeometryTest
     5/23 Test  #5: CFG4Test.CGeometryTest ....................   Passed    1.56 sec
          Start  6: CFG4Test.CG4Test
     6/23 Test  #6: CFG4Test.CG4Test ..........................   Passed   42.06 sec
          Start  7: CFG4Test.G4MaterialTest
     7/23 Test  #7: CFG4Test.G4MaterialTest ...................   Passed    0.06 sec
          Start  8: CFG4Test.G4StringTest
     8/23 Test  #8: CFG4Test.G4StringTest .....................   Passed    0.05 sec
          Start  9: CFG4Test.G4SphereTest
     9/23 Test  #9: CFG4Test.G4SphereTest .....................   Passed    0.05 sec
          Start 10: CFG4Test.CSolidTest
    10/23 Test #10: CFG4Test.CSolidTest .......................   Passed    0.05 sec
          Start 11: CFG4Test.G4PhysicsOrderedFreeVectorTest
    11/23 Test #11: CFG4Test.G4PhysicsOrderedFreeVectorTest ...   Passed    0.05 sec
          Start 12: CFG4Test.CVecTest
    12/23 Test #12: CFG4Test.CVecTest .........................   Passed    0.05 sec
          Start 13: CFG4Test.G4MaterialPropertiesTableTest
    13/23 Test #13: CFG4Test.G4MaterialPropertiesTableTest ....   Passed    0.05 sec
          Start 14: CFG4Test.G4UniformRandTest
    14/23 Test #14: CFG4Test.G4UniformRandTest ................   Passed    0.05 sec
          Start 15: CFG4Test.G4BoxTest
    15/23 Test #15: CFG4Test.G4BoxTest ........................   Passed    0.05 sec
          Start 16: CFG4Test.G4ThreeVectorTest
    16/23 Test #16: CFG4Test.G4ThreeVectorTest ................   Passed    0.05 sec
          Start 17: CFG4Test.CCollectorTest
    17/23 Test #17: CFG4Test.CCollectorTest ...................   Passed    1.80 sec
          Start 18: CFG4Test.CInterpolationTest
    18/23 Test #18: CFG4Test.CInterpolationTest ...............***Exception: Child aborted  1.86 sec
          Start 19: CFG4Test.OpRayleighTest
    19/23 Test #19: CFG4Test.OpRayleighTest ...................   Passed    1.34 sec
          Start 20: CFG4Test.CGROUPVELTest
    20/23 Test #20: CFG4Test.CGROUPVELTest ....................   Passed    0.34 sec
          Start 21: CFG4Test.CMakerTest
    21/23 Test #21: CFG4Test.CMakerTest .......................   Passed    0.05 sec
          Start 22: CFG4Test.CPhotonTest
    22/23 Test #22: CFG4Test.CPhotonTest ......................   Passed    0.05 sec
          Start 23: CFG4Test.CRandomEngineTest
    23/23 Test #23: CFG4Test.CRandomEngineTest ................   Passed    1.67 sec

    96% tests passed, 1 tests failed out of 23

    Total Test time (real) =  55.45 sec

    The following tests FAILED:
         18 - CFG4Test.CInterpolationTest (Child aborted)
    Errors while running CTest
    Thu Aug 16 13:11:06 CST 2018
    epsilon:cfg4 blyth$ 


That fail is a known problem with the default geocache::


    2018-08-16 13:15:09.635 INFO  [2386316] [main@190]    17( 5,-1,-1, 5)                                         IwsWater///IwsWater om         /dd/Materials/IwsWater im         /dd/Materials/IwsWater
    2018-08-16 13:15:09.635 INFO  [2386316] [main@141]  i  18 omat   5 osur   4 isur 4294967295 imat  36
    2018-08-16 13:15:09.635 FATAL [2386316] [*CMaterialBridge::getG4Material@190]  failed to find a G4Material with index 36 in all the indices 15 25 14 24 18 16 26 22 0 2 27 1 4 10 12 13 28 35 9 21 3 29 30 31 32 33 19 6 20 5 17 23 8 7 34 11 
    Assertion failed: (im), function main, file /Users/blyth/opticks/cfg4/tests/CInterpolationTest.cc, line 152.
    Abort trap: 6
    epsilon:cfg4 blyth$ 









::

    2018-08-15 23:38:23.321 INFO  [1438333] [CSurfaceLib::convert@93] . num_surf 48
    Assertion failed: (lv), function makeSkinSurface, file /Users/blyth/opticks/cfg4/CSurfaceLib.cc, line 249.
    Process 92901 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff53502b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff53502b6e <+10>: jae    0x7fff53502b78            ; <+20>
        0x7fff53502b70 <+12>: movq   %rax, %rdi
        0x7fff53502b73 <+15>: jmp    0x7fff534f9b00            ; cerror_nocancel
        0x7fff53502b78 <+20>: retq   
    Target 0: (CGeometryTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff53502b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff536cd080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff5345e1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff534261ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001001c71d6 libCFG4.dylib`CSurfaceLib::makeSkinSurface(this=0x000000010a79b860, surf=0x000000010a587980, os=0x0000000110a8acb0) at CSurfaceLib.cc:249
        frame #5: 0x00000001001c5bbb libCFG4.dylib`CSurfaceLib::convert(this=0x000000010a79b860, detector=0x000000010a79b6c0, exclude_sensors=true) at CSurfaceLib.cc:124
        frame #6: 0x00000001001be49a libCFG4.dylib`CDetector::attachSurfaces(this=0x000000010a79b6c0) at CDetector.cc:292
        frame #7: 0x00000001001c2ef6 libCFG4.dylib`CGDMLDetector::init(this=0x000000010a79b6c0) at CGDMLDetector.cc:75
        frame #8: 0x00000001001c2bbb libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010a79b6c0, hub=0x00007ffeefbfe690, query=0x000000010a512890) at CGDMLDetector.cc:40
        frame #9: 0x00000001001c2f35 libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010a79b6c0, hub=0x00007ffeefbfe690, query=0x000000010a512890) at CGDMLDetector.cc:38
        frame #10: 0x0000000100119a0a libCFG4.dylib`CGeometry::init(this=0x00007ffeefbfe650) at CGeometry.cc:66
        frame #11: 0x0000000100119730 libCFG4.dylib`CGeometry::CGeometry(this=0x00007ffeefbfe650, hub=0x00007ffeefbfe690) at CGeometry.cc:49
        frame #12: 0x0000000100119a9d libCFG4.dylib`CGeometry::CGeometry(this=0x00007ffeefbfe650, hub=0x00007ffeefbfe690) at CGeometry.cc:48
        frame #13: 0x000000010000f7a5 CGeometryTest`main(argc=1, argv=0x00007ffeefbfea98) at CGeometryTest.cc:45
        frame #14: 0x00007fff533b2015 libdyld.dylib`start + 1
        frame #15: 0x00007fff533b2015 libdyld.dylib`start + 1
    (lldb) 



ckm NULL track::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x0)
      * frame #0: 0x000000010002fedc CerenkovMinimal`G4Track::GetCurrentStepNumber(this=0x0000000000000000) const at G4Track.icc:235
        frame #1: 0x000000010002fe79 CerenkovMinimal`Ctx::setStep(this=0x0000000110a06fe0, step=0x0000000110cd09a0) at Ctx.cc:71
        frame #2: 0x000000010002d511 CerenkovMinimal`SteppingAction::UserSteppingAction(this=0x0000000110cf7a00, step=0x0000000110cd09a0) at SteppingAction.cc:15
        frame #3: 0x00000001023aef06 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000110cd0810) at G4SteppingManager.cc:243
        frame #4: 0x00000001023c586f libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000110cd07d0, apValueG4Track=0x0000000116675000) at G4TrackingManager.cc:126
        frame #5: 0x000000010228c71a libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000110cd0740, anEvent=0x0000000116653f20) at G4EventManager.cc:185
        frame #6: 0x000000010228dc2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x0000000110cd0740, anEvent=0x0000000116653f20) at G4EventManager.cc:338
        frame #7: 0x00000001021999f5 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000110a07050, i_event=0) at G4RunManager.cc:399
        frame #8: 0x0000000102199825 libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000110a07050, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
        frame #9: 0x0000000102197ce1 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000110a07050, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #10: 0x00000001000321cd CerenkovMinimal`G4::beamOn(this=0x00007ffeefbfe498, nev=1) at G4.cc:53
        frame #11: 0x0000000100032077 CerenkovMinimal`G4::G4(this=0x00007ffeefbfe498, nev=1) at G4.cc:48
        frame #12: 0x00000001000321fb CerenkovMinimal`G4::G4(this=0x00007ffeefbfe498, nev=1) at G4.cc:30
        frame #13: 0x0000000100011461 CerenkovMinimal`main(argc=1, argv=0x00007ffeefbfe578) at CerenkovMinimal.cc:7
        frame #14: 0x00007fff533b2015 libdyld.dylib`start + 1
        frame #15: 0x00007fff533b2015 libdyld.dylib`start + 1
    (lldb) f 1
    frame #1: 0x000000010002fe79 CerenkovMinimal`Ctx::setStep(this=0x0000000110a06fe0, step=0x0000000110cd09a0) at Ctx.cc:71
       68  	void Ctx::setStep(const G4Step* step)
       69  	{  
       70  	    _step = step ; 
    -> 71  	    _step_id = _track->GetCurrentStepNumber() - 1 ;
       72  	
       73  	    _track_step_count += 1 ;
       74  	    
    (lldb) 



FIXED : by deleting build dir and rebuilding : PostUserTrackingAction bad access
-----------------------------------------------------------------------------------

::

    (lldb) f 1
    frame #1: 0x00000001023c6937 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000110d85e80, apValueG4Track=0x0000000116e31e80) at G4TrackingManager.cc:140
       137 	
       138 	  // Post tracking user intervention process.
       139 	  if( fpUserTrackingAction != 0 ) {
    -> 140 	     fpUserTrackingAction->PostUserTrackingAction(fpTrack);
       141 	  }
       142 	
       143 	  // Destruct the trajectory if it was created
    (lldb) p fpTrack
    (G4Track *) $0 = 0x0000000116e31e80
    (lldb) f 0
    frame #0: 0x00007fff8b2e7058 libc++abi.dylib`vtable for __cxxabiv1::__si_class_type_info + 16
    libc++abi.dylib`vtable for __cxxabiv1::__si_class_type_info:
    ->  0x7fff8b2e7058 <+16>: popq   %rsi
        0x7fff8b2e7059 <+17>: cli    
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=2, address=0x7fff8b2e7058)
      * frame #0: 0x00007fff8b2e7058 libc++abi.dylib`vtable for __cxxabiv1::__si_class_type_info + 16
        frame #1: 0x00000001023c6937 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000110d85e80, apValueG4Track=0x0000000116e31e80) at G4TrackingManager.cc:140
        frame #2: 0x000000010228d71a libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000110d85df0, anEvent=0x0000000116e02600) at G4EventManager.cc:185
        frame #3: 0x000000010228ec2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x0000000110d85df0, anEvent=0x0000000116e02600) at G4EventManager.cc:338
        frame #4: 0x000000010219a9f5 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000110c5dbf0, i_event=0) at G4RunManager.cc:399
        frame #5: 0x000000010219a825 libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000110c5dbf0, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
        frame #6: 0x0000000102198ce1 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000110c5dbf0, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #7: 0x000000010003310d CerenkovMinimal`G4::beamOn(this=0x00007ffeefbfe498, nev=1) at G4.cc:53
        frame #8: 0x0000000100032fb7 CerenkovMinimal`G4::G4(this=0x00007ffeefbfe498, nev=1) at G4.cc:48
        frame #9: 0x000000010003313b CerenkovMinimal`G4::G4(this=0x00007ffeefbfe498, nev=1) at G4.cc:30
        frame #10: 0x0000000100011e91 CerenkovMinimal`main(argc=1, argv=0x00007ffeefbfe578) at CerenkovMinimal.cc:7
        frame #11: 0x00007fff533b2015 libdyld.dylib`start + 1
        frame #12: 0x00007fff533b2015 libdyld.dylib`start + 1
    (lldb) 

::

    epsilon:issues blyth$ opticks-deps | grep G4
    INFO:__main__:root /Users/blyth/opticks 
     10          OKCONF :               okconf :               OKConf : OpticksCUDA OptiX G4  
     65              X4 :                extg4 :                ExtG4 : G4 GGeo  
    170            CFG4 :                 cfg4 :                 CFG4 : G4 ExtG4 OpticksXercesC OpticksGeo  
    180            OKG4 :                 okg4 :                 OKG4 : OK CFG4  
    190            G4OK :                 g4ok :                 G4OK : CFG4 ExtG4 OKOP  


Clean build all proj depending on G4::

    opticks-g4-clean-build()
    {
        local arg="extg4:"

        om-subs $arg 
        om-clean $arg 

        type $FUNCNAME
        read -p "$FUNCNAME : enter YES to proceed to to clean and build : " ans

        [ "$ans" != "YES" ] && echo skip && return 

        om-clean $arg | sh 
        om-conf $arg 
        om-make $arg 
    }



