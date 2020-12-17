Hans-opticks-t-fails
=======================

::

    Hi simon 

    I used the head of the git repository 
    git clone http://bitbucket.org/simoncblyth/opticks

    building the externasl went fine 

    But to do opticks-full I had to change

     /data2/wenzel/gputest3/opticks/optickscore/CMakeLists.txt
     /data2/wenzel/gputest3/opticks/opticksgeo/CMakeLists.txt

    to include 

    include_directories($ENV{OPTICKS_HOME}/npy)

    then opticks-full  completed fine and 
    with this  opticks-t only reports 9 failures 

        FAILS:  9   / 431   :  Wed Dec 16 19:01:35 2020  

          22 /32  Test #22 : OptiXRapTest.interpolationTest                ***Failed                      0.51  
          3  /34  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: SegFault         0.18  
          5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     0.16  
          6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     0.16  
          7  /34  Test #7  : CFG4Test.CG4Test                              ***Exception: SegFault         0.18  
          1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: SegFault         0.20  
          23 /34  Test #23 : CFG4Test.CInterpolationTest                   ***Exception: SegFault         0.18  
          29 /34  Test #29 : CFG4Test.CRandomEngineTest                    ***Exception: SegFault         0.18  
          2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.00   


    below is the result of running these tests with gdb and backtracing. 

    hope that helps 

    cheers Hans 


All these are failing from the same cause, of being unable to extract the gdmlpath from 
the geocache metadata as was expecting "--gdmlpath path/to/geometry.gdml"
I have generalized the parsing of the argline in Opticks::ExtractCacheMetaGDMLPath 
to also just grab the first arg ending with ".gdml" so this will work in the 
G4OpticksTest case if you provide an absolute gdmlpath on commandline. 
You could use $PWD/name.gdml to get the shell to so the work.



gdmlpath assert trips up a few tests
---------------------------------------

::

    FAILS:  2   / 431   :  Fri Dec 18 00:02:49 2020   
      11 /19  Test #11 : ExtG4Test.X4PhysicalVolumeTest                Child aborted***Exception:     0.15   
      12 /19  Test #12 : ExtG4Test.X4PhysicalVolume2Test               Child aborted***Exception:     0.64   


Added "--nogdmlpath" to prevent the assert but message not getting thru::

    Opticks=INFO BCfg=INFO X4PhysicalVolumeTest

Found and fixed this problem due to sysrap/SArgs when argc=0 was ignoring the first forced argument.





interpolationTest : fails as the cached argline in geocache metadata does not have "--gdmlpath ..."  
------------------------------------------------------------------------------------------------------

Have handled this case returning None for gdmlpath.  As looks like the test doesnt use the gdmlpath
this probably fixes the fail.

::

    wenzel@aichi:/data2/wenzel/gputest3/opticks$ gdb --args /data2/wenzel/gputest2/local/opticks/build/optixrap/tests/interpolationTest --interactive-debug-mode 0 --output-on-failure
    GNU gdb (Ubuntu 9.2-0ubuntu1~20.04) 9.2
    Copyright (C) 2020 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.
    Type "show copying" and "show warranty" for details.
    This GDB was configured as "x86_64-linux-gnu".
    Type "show configuration" for configuration details.
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>.
    Find the GDB manual and other documentation resources online at:
        <http://www.gnu.org/software/gdb/documentation/>.

    For help, type "help".
    Type "apropos word" to search for commands related to "word"...
    Reading symbols from /data2/wenzel/gputest2/local/opticks/build/optixrap/tests/interpolationTest...
    (gdb) run
    Starting program: /data2/wenzel/gputest2/local/opticks/build/optixrap/tests/interpolationTest --interactive-debug-mode 0 --output-on-failure
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2020-12-16 19:04:14.442 INFO  [2410703] [BOpticksKey::SetKey@77]  spec G4OpticksTest.X4PhysicalVolume.World_PV.f2f063d9ea288eeab99e0b1617699755
    2020-12-16 19:04:14.443 INFO  [2410703] [Opticks::init@431] INTEROP_MODE hostname aichi
    2020-12-16 19:04:14.443 INFO  [2410703] [Opticks::init@440]  mandatory keyed access to geometry, opticksaux
    2020-12-16 19:04:14.443 INFO  [2410703] [Opticks::init@459] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_ANGULAR WITH_DEBUG_BUFFER WITH_WAY_BUFFER
    2020-12-16 19:04:14.443 ERROR [2410703] [BOpticksKey::SetKey@67] key is already set, ignoring update with spec (null)
    2020-12-16 19:04:14.443 INFO  [2410703] [BOpticksResource::setupViaKey@774]
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : G4OpticksTest.X4PhysicalVolume.World_PV.f2f063d9ea288eeab99e0b1617699755
                     exename  : G4OpticksTest
             current_exename  : interpolationTest
                       class  : X4PhysicalVolume
                     volname  : World_PV
                      digest  : f2f063d9ea288eeab99e0b1617699755
                      idname  : G4OpticksTest_World_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2020-12-16 19:04:14.445 INFO  [2410703] [Opticks::loadOriginCacheMeta@1877]  cachemetapath /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f2f063d9ea288eeab99e0b1617699755/1/cachemeta.json
    2020-12-16 19:04:14.445 INFO  [2410703] [BMeta::dump@199] Opticks::loadOriginCacheMeta
    {
        "GEOCACHE_CODE_VERSION": 9,
        "argline": "./G4OpticksTest G4Opticks_50000.gdml muon_noIO.mac ",
        "location": "Opticks::updateCacheMeta",
        "rundate": "20201216_133046",
        "runfolder": "G4OpticksTest",
        "runlabel": "R0_cvd_",
        "runstamp": 1608147046
    }
    2020-12-16 19:04:14.445 INFO  [2410703] [Opticks::loadOriginCacheMeta@1881] ExtractCacheMetaGDMLPath
    2020-12-16 19:04:14.445 INFO  [2410703] [Opticks::loadOriginCacheMeta@1909] (pass) GEOCACHE_CODE_VERSION 9
    2020-12-16 19:04:14.445 INFO  [2410703] [OpticksHub::loadGeometry@282] [ /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f2f063d9ea288eeab99e0b1617699755/1
    2020-12-16 19:04:14.460 INFO  [2410703] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2020-12-16 19:04:14.460 INFO  [2410703] [OpticksHub::loadGeometry@314] ]
    2020-12-16 19:04:14.461 FATAL [2410703] [Opticks::makeSimpleTorchStep@3459]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg
    2020-12-16 19:04:14.461 FATAL [2410703] [OpticksResource::getDefaultFrame@199]  PLACEHOLDER ZERO
    2020-12-16 19:04:14.461 INFO  [2410703] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2020-12-16 19:04:14.461 ERROR [2410703] [OpticksGen::makeTorchstep@441]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0 cmdline_target [--gensteptarget] : 0 gdmlaux_target : -1 active_target : 0
    2020-12-16 19:04:14.461 ERROR [2410703] [OpticksGen::makeTorchstep@468]  generateoverride 0 num_photons0 10000 num_photons 10000
    2020-12-16 19:04:14.474 INFO  [2410703] [OContext::InitRTX@312]  --rtx 0 setting  OFF
    [New Thread 0x7fffe6828700 (LWP 2410710)]
    2020-12-16 19:04:14.509 INFO  [2410703] [OContext::CheckDevices@196]
    Device 0               GeForce RTX 2070 ordinal 0 Compute Support: 7 5 Total Memory: 8366784512

    2020-12-16 19:04:14.520 INFO  [2410703] [CDevice::Dump@244] Visible devices[0:GeForce_RTX_2070]
    2020-12-16 19:04:14.520 INFO  [2410703] [CDevice::Dump@248] CDevice index 0 ordinal 0 name GeForce RTX 2070 major 7 minor 5 compute_capability 75 multiProcessorCount 36 totalGlobalMem 8366784512
    2020-12-16 19:04:14.520 INFO  [2410703] [CDevice::Dump@244] All devices[0:GeForce_RTX_2070]
    2020-12-16 19:04:14.520 INFO  [2410703] [CDevice::Dump@248] CDevice index 0 ordinal 0 name GeForce RTX 2070 major 7 minor 5 compute_capability 75 multiProcessorCount 36 totalGlobalMem 8366784512
    [New Thread 0x7fffe36a5700 (LWP 2410711)]
    [New Thread 0x7fffe2ce0700 (LWP 2410712)]
    2020-12-16 19:04:14.598 INFO  [2410703] [OGeo::init@240] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
    2020-12-16 19:04:14.598 INFO  [2410703] [GGeoLib::dump@359] OGeo::convert GGeoLib numMergedMesh 1 ptr 0x55555564b410
    mm index   0 geocode   A                  numVolumes          7 numFaces         504 numITransforms           1 numITransforms*numVolumes           7 GParts Y GPts Y
     num_remainder_volumes 7 num_instanced_volumes 0 num_remainder_volumes + num_instanced_volumes 7 num_total_faces 504 num_total_faces_woi 504 (woi:without instancing)
       0 pts Y  GPts.NumPt     7 lvIdx ( 2 1 0 0 0 0 0)
    2020-12-16 19:04:14.598 INFO  [2410703] [OGeo::convert@284] [ nmm 1
    2020-12-16 19:04:14.630 INFO  [2410703] [OGeo::convert@297] ] nmm 1
    2020-12-16 19:04:14.630 INFO  [2410703] [main@189]  ok
    2020-12-16 19:04:14.631 INFO  [2410703] [interpolationTest::init@115]  name interpolationTest_interpol.npy base $TMP/optixrap/interpolationTest script interpolationTest_interpol.py nb     7 nx   761 ny    56 progname              interpolationTest
    2020-12-16 19:04:14.631 INFO  [2410703] [OLaunchTest::init@69] OLaunchTest entry   0 width       1 height       1 ptx                               interpolationTest.cu prog                                  interpolationTest
    2020-12-16 19:04:14.631 INFO  [2410703] [OLaunchTest::launch@80] OLaunchTest entry   0 width     761 height       7 ptx                               interpolationTest.cu prog                                  interpolationTest
    2020-12-16 19:04:14.876 INFO  [2410703] [interpolationTest::launch@158] OLaunchTest entry   0 width     761 height       7 ptx                               interpolationTest.cu prog                                  interpolationTest
    2020-12-16 19:04:14.877 INFO  [2410703] [interpolationTest::launch@165]  save  base $TMP/optixrap/interpolationTest name interpolationTest_interpol.npy
    [Detaching after vfork from child process 2410713]
    [Detaching after vfork from child process 2410715]
    2020-12-16 19:04:14.880 INFO  [2410703] [SSys::RunPythonScript@521]  script interpolationTest_interpol.py script_path /data2/wenzel/gputest3/local/opticks/bin/interpolationTest_interpol.py python_executable /usr/bin/python
    [Detaching after vfork from child process 2410717]
    [{extract_argument_after:key.py    :113} INFO     - ppos -1
    Traceback (most recent call last):
      File "/data2/wenzel/gputest3/local/opticks/bin/interpolationTest_interpol.py", line 33, in <module>
        args = opticks_main()
      File "/data2/wenzel/gputest3/opticks/ana/main.py", line 398, in opticks_main
        opticks_environment(ok)
      File "/data2/wenzel/gputest3/opticks/ana/env.py", line 40, in opticks_environment
        env = OpticksEnv(ok)
      File "/data2/wenzel/gputest3/opticks/ana/env.py", line 130, in __init__
        self.direct_init()
      File "/data2/wenzel/gputest3/opticks/ana/env.py", line 158, in direct_init
        self.key = Key(os.environ["OPTICKS_KEY"])
      File "/data2/wenzel/gputest3/opticks/ana/key.py", line 102, in __init__
        self.gdmlpath = self.extract_argument_after(meta, "--gdmlpath")
      File "/data2/wenzel/gputest3/opticks/ana/key.py", line 122, in extract_argument_after
        return arg
    UnboundLocalError: local variable 'arg' referenced before assignment
    2020-12-16 19:04:14.934 INFO  [2410703] [SSys::run@100] /usr/bin/python /data2/wenzel/gputest3/local/opticks/bin/interpolationTest_interpol.py  rc_raw : 256 rc : 1
    2020-12-16 19:04:14.934 ERROR [2410703] [SSys::run@107] FAILED with  cmd /usr/bin/python /data2/wenzel/gputest3/local/opticks/bin/interpolationTest_interpol.py  RC 1
    2020-12-16 19:04:14.934 INFO  [2410703] [SSys::RunPythonScript@528]  RC 1
    [Thread 0x7fffe2ce0700 (LWP 2410712) exited]
    [Thread 0x7fffe6828700 (LWP 2410710) exited]
    [Thread 0x7ffff3a84f40 (LWP 2410703) exited]
    [Inferior 1 (process 2410703) exited with code 01]
    (gdb) bt
    No stack.



ana/env.py::

    156         assert not "IDPATH" in os.environ, "IDPATH envvar as input is forbidden"
    157         assert "OPTICKS_KEY" in os.environ, "OPTICKS_KEY envvar is required"
    158         self.key = Key(os.environ["OPTICKS_KEY"])

ana/key.py::

    088     def __init__(self, keyspec=None):
     89         if keyspec is None:
     90             keyspec = os.environ.get("OPTICKS_KEY",None)
     91         pass
     92         keydir = Key.Keydir(keyspec)
     93         exists = os.path.isdir(keydir)
     94         meta = json.load(open(os.path.join(keydir, "cachemeta.json")))
     95 
     96         self.keyspec = keyspec
     97         self.keydir = keydir
     98         self.exists = exists
     99         self.digest = keyspec.split(".")[-1]
    100         self.meta = meta
    101         self.version = int(meta["GEOCACHE_CODE_VERSION"])
    102         self.gdmlpath = self.extract_argument_after(meta, "--gdmlpath")
    103 


CTestDetectorTest : looks plain and simple due to lack of gdmlpath in argline 
-------------------------------------------------------------------------------

Mimic the failure by changing geocache metadata argline --gdmlpath to --hidden-gdmlpath::

    kcd
    vi cachemeta.json  

The fail is unavoidable, but make it fail sooner and make it more obvious what
is going wrong::

    1874 void Opticks::loadOriginCacheMeta()
    1875 {
    1876     const char* cachemetapath = getCacheMetaPath();
    1877     LOG(info) << " cachemetapath " << cachemetapath ; 
    1878     m_origin_cachemeta = BMeta::Load(cachemetapath); 
    1879     m_origin_cachemeta->dump("Opticks::loadOriginCacheMeta"); 
    1880     std::string gdmlpath = ExtractCacheMetaGDMLPath(m_origin_cachemeta);
    1881     LOG(info) << "ExtractCacheMetaGDMLPath " << gdmlpath ;
    1882     
    1883     m_origin_gdmlpath = gdmlpath.empty() ? NULL : strdup(gdmlpath.c_str());
    1884     
    1885     if(m_origin_gdmlpath == NULL)
    1886     {   
    1887         LOG(fatal) << "cachemetapath " << cachemetapath ; 
    1888         LOG(fatal) << "argline that creates cachemetapath must include \"--gdmlpath /path/to/geometry.gdml\" " ;
    1889     }
    1890     assert( m_origin_gdmlpath );
    1891     
    1892     

::

    wenzel@aichi:/data2/wenzel/gputest3/opticks$ gdb --args /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CTestDetectorTest --interactive-debug-mode 0 --output-on-failure
    GNU gdb (Ubuntu 9.2-0ubuntu1~20.04) 9.2
    Copyright (C) 2020 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.
    Type "show copying" and "show warranty" for details.
    This GDB was configured as "x86_64-linux-gnu".
    Type "show configuration" for configuration details.
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>.
    Find the GDB manual and other documentation resources online at:
        <http://www.gnu.org/software/gdb/documentation/>.

    For help, type "help".
    Type "apropos word" to search for commands related to "word"...
    Reading symbols from /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CTestDetectorTest...
    (gdb) run
    Starting program: /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CTestDetectorTest --interactive-debug-mode 0 --output-on-failure
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2020-12-16 19:05:27.071 INFO  [2413329] [main@44] /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CTestDetectorTest
    2020-12-16 19:05:27.071 INFO  [2413329] [BOpticksKey::SetKey@77]  spec G4OpticksTest.X4PhysicalVolume.World_PV.f2f063d9ea288eeab99e0b1617699755
    2020-12-16 19:05:27.072 INFO  [2413329] [Opticks::init@431] INTEROP_MODE hostname aichi
    2020-12-16 19:05:27.072 INFO  [2413329] [Opticks::init@440]  mandatory keyed access to geometry, opticksaux
    2020-12-16 19:05:27.072 INFO  [2413329] [Opticks::init@459] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_ANGULAR WITH_DEBUG_BUFFER WITH_WAY_BUFFER
    2020-12-16 19:05:27.072 ERROR [2413329] [OpticksResource::SetupG4Environment@212] inipath /data2/wenzel/gputest3/local/opticks/externals/config/geant4.ini
    2020-12-16 19:05:27.072 ERROR [2413329] [OpticksResource::SetupG4Environment@221]  MISSING inipath /data2/wenzel/gputest3/local/opticks/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini )
    2020-12-16 19:05:27.072 ERROR [2413329] [BOpticksKey::SetKey@67] key is already set, ignoring update with spec (null)
    2020-12-16 19:05:27.072 INFO  [2413329] [BOpticksResource::setupViaKey@774]
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : G4OpticksTest.X4PhysicalVolume.World_PV.f2f063d9ea288eeab99e0b1617699755
                     exename  : G4OpticksTest
             current_exename  : CTestDetectorTest
                       class  : X4PhysicalVolume
                     volname  : World_PV
                      digest  : f2f063d9ea288eeab99e0b1617699755
                      idname  : G4OpticksTest_World_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2020-12-16 19:05:27.073 INFO  [2413329] [Opticks::loadOriginCacheMeta@1877]  cachemetapath /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f2f063d9ea288eeab99e0b1617699755/1/cachemeta.json
    2020-12-16 19:05:27.073 INFO  [2413329] [BMeta::dump@199] Opticks::loadOriginCacheMeta
    {
        "GEOCACHE_CODE_VERSION": 9,
        "argline": "./G4OpticksTest G4Opticks_50000.gdml muon_noIO.mac ",
        "location": "Opticks::updateCacheMeta",
        "rundate": "20201216_133046",
        "runfolder": "G4OpticksTest",
        "runlabel": "R0_cvd_",
        "runstamp": 1608147046
    }
    2020-12-16 19:05:27.073 INFO  [2413329] [Opticks::loadOriginCacheMeta@1881] ExtractCacheMetaGDMLPath
    2020-12-16 19:05:27.074 INFO  [2413329] [Opticks::loadOriginCacheMeta@1909] (pass) GEOCACHE_CODE_VERSION 9
    2020-12-16 19:05:27.074 INFO  [2413329] [OpticksHub::loadGeometry@282] [ /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f2f063d9ea288eeab99e0b1617699755/1
    2020-12-16 19:05:27.089 INFO  [2413329] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2020-12-16 19:05:27.089 INFO  [2413329] [OpticksHub::loadGeometry@314] ]
    2020-12-16 19:05:27.089 FATAL [2413329] [Opticks::makeSimpleTorchStep@3459]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg
    2020-12-16 19:05:27.089 FATAL [2413329] [OpticksResource::getDefaultFrame@199]  PLACEHOLDER ZERO
    2020-12-16 19:05:27.089 INFO  [2413329] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2020-12-16 19:05:27.089 ERROR [2413329] [OpticksGen::makeTorchstep@441]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0 cmdline_target [--gensteptarget] : 0 gdmlaux_target : -1 active_target : 0
    2020-12-16 19:05:27.089 ERROR [2413329] [OpticksGen::makeTorchstep@468]  generateoverride 0 num_photons0 10000 num_photons 10000
    2020-12-16 19:05:27.090 INFO  [2413329] [BOpticksResource::IsGeant4EnvironmentDetected@291]  n 11 detect 1
    2020-12-16 19:05:27.090 ERROR [2413329] [CG4::preinit@136] External Geant4 environment is detected, not changing this.

    **************************************************************
     Geant4 version Name: geant4-10-06-patch-03 [MT]   (6-November-2020)
                           Copyright : Geant4 Collaboration
                          References : NIM A 506 (2003), 250-303
                                     : IEEE-TNS 53 (2006), 270-278
                                     : NIM A 835 (2016), 186-225
                                 WWW : http://geant4.org/
    **************************************************************

    2020-12-16 19:05:27.108 ERROR [2413329] [BFile::ExistsFile@485] BFile::ExistsFile BAD PATH path  sub NULL name NULL
    2020-12-16 19:05:27.108 ERROR [2413329] [CGDMLDetector::init@79] CGDMLDetector::init PATH DOES NOT EXIST  path
    2020-12-16 19:05:27.108 FATAL [2413329] [Opticks::setSpaceDomain@2609]  changing w 1000 -> 0
    2020-12-16 19:05:27.108 FATAL [2413329] [CTorchSource::configure@163] CTorchSource::configure _t 0.1 _radius 0 _pos 0.0000,0.0000,0.0000 _dir 0.0000,0.0000,1.0000 _zeaz 0.0000,1.0000,0.0000,1.0000 _pol 0.0000,0.0000,1.0000

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff45edf25 in G4RunManagerKernel::DefineWorldVolume(G4VPhysicalVolume*, bool) () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    (gdb) bt
    #0  0x00007ffff45edf25 in G4RunManagerKernel::DefineWorldVolume(G4VPhysicalVolume*, bool) () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    #1  0x00007ffff45cf3a5 in G4RunManager::InitializeGeometry() () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    #2  0x00007ffff45cf221 in G4RunManager::Initialize() () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    #3  0x00007ffff7f315ba in CG4::initialize (this=0x7fffffffc6b0) at /data2/wenzel/gputest3/opticks/cfg4/CG4.cc:218
    #4  0x00007ffff7f312aa in CG4::init (this=0x7fffffffc6b0) at /data2/wenzel/gputest3/opticks/cfg4/CG4.cc:190
    #5  0x00007ffff7f30fa2 in CG4::CG4 (this=0x7fffffffc6b0, hub=0x7fffffffc480) at /data2/wenzel/gputest3/opticks/cfg4/CG4.cc:181
    #6  0x00005555555590bd in main (argc=4, argv=0x7fffffffcb88) at /data2/wenzel/gputest2/opticks/cfg4/tests/CTestDetectorTest.cc:57
    (gdb) quit
    A debugging session is active.

    Inferior 1 [process 2413329] will be killed.

    Quit anyway? (y or n) y


CGDMLDetectorTest  : same issue as above CTestDetectorTest
-------------------------------------------------------------

::

    wenzel@aichi:/data2/wenzel/gputest3/opticks$ gdb --args /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CGDMLDetectorTest --interactive-debug-mode 0 --output-on-failure
    GNU gdb (Ubuntu 9.2-0ubuntu1~20.04) 9.2
    Copyright (C) 2020 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.
    Type "show copying" and "show warranty" for details.
    This GDB was configured as "x86_64-linux-gnu".
    Type "show configuration" for configuration details.
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>.
    Find the GDB manual and other documentation resources online at:
        <http://www.gnu.org/software/gdb/documentation/>.

    For help, type "help".
    Type "apropos word" to search for commands related to "word"...
    Reading symbols from /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CGDMLDetectorTest...
    (gdb) run
    Starting program: /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CGDMLDetectorTest --interactive-debug-mode 0 --output-on-failure
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2020-12-16 19:07:01.317 INFO  [2417454] [main@97] /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CGDMLDetectorTest
    2020-12-16 19:07:01.317 INFO  [2417454] [BOpticksKey::SetKey@77]  spec G4OpticksTest.X4PhysicalVolume.World_PV.f2f063d9ea288eeab99e0b1617699755
    2020-12-16 19:07:01.318 INFO  [2417454] [Opticks::init@431] INTEROP_MODE hostname aichi
    2020-12-16 19:07:01.318 INFO  [2417454] [Opticks::init@440]  mandatory keyed access to geometry, opticksaux
    2020-12-16 19:07:01.318 INFO  [2417454] [Opticks::init@459] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_ANGULAR WITH_DEBUG_BUFFER WITH_WAY_BUFFER
    2020-12-16 19:07:01.318 ERROR [2417454] [BOpticksKey::SetKey@67] key is already set, ignoring update with spec (null)
    2020-12-16 19:07:01.318 INFO  [2417454] [BOpticksResource::setupViaKey@774]
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : G4OpticksTest.X4PhysicalVolume.World_PV.f2f063d9ea288eeab99e0b1617699755
                     exename  : G4OpticksTest
             current_exename  : CGDMLDetectorTest
                       class  : X4PhysicalVolume
                     volname  : World_PV
                      digest  : f2f063d9ea288eeab99e0b1617699755
                      idname  : G4OpticksTest_World_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2020-12-16 19:07:01.320 INFO  [2417454] [Opticks::loadOriginCacheMeta@1877]  cachemetapath /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f2f063d9ea288eeab99e0b1617699755/1/cachemeta.json
    2020-12-16 19:07:01.320 INFO  [2417454] [BMeta::dump@199] Opticks::loadOriginCacheMeta
    {
        "GEOCACHE_CODE_VERSION": 9,
        "argline": "./G4OpticksTest G4Opticks_50000.gdml muon_noIO.mac ",
        "location": "Opticks::updateCacheMeta",
        "rundate": "20201216_133046",
        "runfolder": "G4OpticksTest",
        "runlabel": "R0_cvd_",
        "runstamp": 1608147046
    }
    2020-12-16 19:07:01.320 INFO  [2417454] [Opticks::loadOriginCacheMeta@1881] ExtractCacheMetaGDMLPath
    2020-12-16 19:07:01.320 INFO  [2417454] [Opticks::loadOriginCacheMeta@1909] (pass) GEOCACHE_CODE_VERSION 9
    2020-12-16 19:07:01.320 INFO  [2417454] [OpticksHub::loadGeometry@282] [ /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f2f063d9ea288eeab99e0b1617699755/1
    2020-12-16 19:07:01.335 INFO  [2417454] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2020-12-16 19:07:01.335 INFO  [2417454] [OpticksHub::loadGeometry@314] ]
    2020-12-16 19:07:01.335 FATAL [2417454] [Opticks::makeSimpleTorchStep@3459]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg
    2020-12-16 19:07:01.335 FATAL [2417454] [OpticksResource::getDefaultFrame@199]  PLACEHOLDER ZERO
    2020-12-16 19:07:01.335 INFO  [2417454] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2020-12-16 19:07:01.335 ERROR [2417454] [OpticksGen::makeTorchstep@441]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0 cmdline_target [--gensteptarget] : 0 gdmlaux_target : -1 active_target : 0
    2020-12-16 19:07:01.336 ERROR [2417454] [OpticksGen::makeTorchstep@468]  generateoverride 0 num_photons0 10000 num_photons 10000
    2020-12-16 19:07:01.336 ERROR [2417454] [main@102] //////////////////////////  AFTER OpticksHub instanciation /////////////////////////////////////
    2020-12-16 19:07:01.336 ERROR [2417454] [BFile::ExistsFile@485] BFile::ExistsFile BAD PATH path  sub NULL name NULL
    2020-12-16 19:07:01.336 ERROR [2417454] [CGDMLDetector::init@79] CGDMLDetector::init PATH DOES NOT EXIST  path
    2020-12-16 19:07:01.336 ERROR [2417454] [main@115] //////////////////////////  AFTER CGDMLDetector instanciation /////////////////////////////////////
    CGDMLDetectorTest: /data2/wenzel/gputest3/opticks/cfg4/CDetector.cc:153: void CDetector::saveBuffers(const char*, unsigned int): Assertion `m_traverser' failed.

    Program received signal SIGABRT, Aborted.
    __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
    50 ../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
    (gdb) bt
    #0  __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
    #1  0x00007ffff68b7859 in __GI_abort () at abort.c:79
    #2  0x00007ffff68b7729 in __assert_fail_base (fmt=0x7ffff6a4d588 "%s%s%s:%u: %s%sAssertion `%s' failed.\n%n", assertion=0x7ffff7f654c1 "m_traverser",
        file=0x7ffff7f65490 "/data2/wenzel/gputest3/opticks/cfg4/CDetector.cc", line=153, function=<optimized out>) at assert.c:92
    #3  0x00007ffff68c8f36 in __GI___assert_fail (assertion=0x7ffff7f654c1 "m_traverser", file=0x7ffff7f65490 "/data2/wenzel/gputest3/opticks/cfg4/CDetector.cc", line=153,
        function=0x7ffff7f654d0 "void CDetector::saveBuffers(const char*, unsigned int)") at assert.c:101
    #4  0x00007ffff7f0fd5e in CDetector::saveBuffers (this=0x5555557194b0, objname=0x7ffff7f6627d "CGDMLDetector", objindex=0) at /data2/wenzel/gputest3/opticks/cfg4/CDetector.cc:153
    #5  0x00007ffff7f13cdc in CGDMLDetector::saveBuffers (this=0x5555557194b0) at /data2/wenzel/gputest3/opticks/cfg4/CGDMLDetector.cc:146
    #6  0x000055555555c0a8 in main (argc=4, argv=0x7fffffffcb88) at /data2/wenzel/gputest2/opticks/cfg4/tests/CGDMLDetectorTest.cc:118
    (gdb) quit
    A debugging session is active.

    Inferior 1 [process 2417454] will be killed.

    Quit anyway? (y or n) y


CG4Test : same again
-----------------------

::

    wenzel@aichi:/data2/wenzel/gputest3/opticks$ gdb --args /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CG4Test --interactive-debug-mode 0 --output-on-failure
    GNU gdb (Ubuntu 9.2-0ubuntu1~20.04) 9.2
    Copyright (C) 2020 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.
    Type "show copying" and "show warranty" for details.
    This GDB was configured as "x86_64-linux-gnu".
    Type "show configuration" for configuration details.
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>.
    Find the GDB manual and other documentation resources online at:
        <http://www.gnu.org/software/gdb/documentation/>.

    For help, type "help".
    Type "apropos word" to search for commands related to "word"...
    Reading symbols from /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CG4Test...
    (gdb) run
    Starting program: /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CG4Test --interactive-debug-mode 0 --output-on-failure
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2020-12-16 19:08:31.157 INFO  [2421758] [main@38] /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CG4Test
    2020-12-16 19:08:31.157 INFO  [2421758] [BOpticksKey::SetKey@77]  spec G4OpticksTest.X4PhysicalVolume.World_PV.f2f063d9ea288eeab99e0b1617699755
    2020-12-16 19:08:31.157 INFO  [2421758] [Opticks::init@431] INTEROP_MODE hostname aichi
    2020-12-16 19:08:31.157 INFO  [2421758] [Opticks::init@440]  mandatory keyed access to geometry, opticksaux
    2020-12-16 19:08:31.158 INFO  [2421758] [Opticks::init@459] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_ANGULAR WITH_DEBUG_BUFFER WITH_WAY_BUFFER
    2020-12-16 19:08:31.158 ERROR [2421758] [BOpticksKey::SetKey@67] key is already set, ignoring update with spec (null)
    2020-12-16 19:08:31.158 INFO  [2421758] [BOpticksResource::setupViaKey@774]
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : G4OpticksTest.X4PhysicalVolume.World_PV.f2f063d9ea288eeab99e0b1617699755
                     exename  : G4OpticksTest
             current_exename  : CG4Test
                       class  : X4PhysicalVolume
                     volname  : World_PV
                      digest  : f2f063d9ea288eeab99e0b1617699755
                      idname  : G4OpticksTest_World_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2020-12-16 19:08:31.159 INFO  [2421758] [Opticks::loadOriginCacheMeta@1877]  cachemetapath /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f2f063d9ea288eeab99e0b1617699755/1/cachemeta.json
    2020-12-16 19:08:31.159 INFO  [2421758] [BMeta::dump@199] Opticks::loadOriginCacheMeta
    {
        "GEOCACHE_CODE_VERSION": 9,
        "argline": "./G4OpticksTest G4Opticks_50000.gdml muon_noIO.mac ",
        "location": "Opticks::updateCacheMeta",
        "rundate": "20201216_133046",
        "runfolder": "G4OpticksTest",
        "runlabel": "R0_cvd_",
        "runstamp": 1608147046
    }
    2020-12-16 19:08:31.159 INFO  [2421758] [Opticks::loadOriginCacheMeta@1881] ExtractCacheMetaGDMLPath
    2020-12-16 19:08:31.159 INFO  [2421758] [Opticks::loadOriginCacheMeta@1909] (pass) GEOCACHE_CODE_VERSION 9
    2020-12-16 19:08:31.159 INFO  [2421758] [OpticksHub::loadGeometry@282] [ /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f2f063d9ea288eeab99e0b1617699755/1
    2020-12-16 19:08:31.175 INFO  [2421758] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2020-12-16 19:08:31.175 INFO  [2421758] [OpticksHub::loadGeometry@314] ]
    2020-12-16 19:08:31.175 FATAL [2421758] [Opticks::makeSimpleTorchStep@3459]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg
    2020-12-16 19:08:31.175 FATAL [2421758] [OpticksResource::getDefaultFrame@199]  PLACEHOLDER ZERO
    2020-12-16 19:08:31.175 INFO  [2421758] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2020-12-16 19:08:31.175 ERROR [2421758] [OpticksGen::makeTorchstep@441]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0 cmdline_target [--gensteptarget] : 0 gdmlaux_target : -1 active_target : 0
    2020-12-16 19:08:31.175 ERROR [2421758] [OpticksGen::makeTorchstep@468]  generateoverride 0 num_photons0 10000 num_photons 10000
    2020-12-16 19:08:31.175 WARN  [2421758] [main@43]  post hub
    2020-12-16 19:08:31.175 WARN  [2421758] [main@46]  post run
    2020-12-16 19:08:31.176 INFO  [2421758] [BOpticksResource::IsGeant4EnvironmentDetected@291]  n 11 detect 1
    2020-12-16 19:08:31.176 ERROR [2421758] [CG4::preinit@136] External Geant4 environment is detected, not changing this.

    **************************************************************
     Geant4 version Name: geant4-10-06-patch-03 [MT]   (6-November-2020)
                           Copyright : Geant4 Collaboration
                          References : NIM A 506 (2003), 250-303
                                     : IEEE-TNS 53 (2006), 270-278
                                     : NIM A 835 (2016), 186-225
                                 WWW : http://geant4.org/
    **************************************************************

    2020-12-16 19:08:31.194 ERROR [2421758] [BFile::ExistsFile@485] BFile::ExistsFile BAD PATH path  sub NULL name NULL
    2020-12-16 19:08:31.194 ERROR [2421758] [CGDMLDetector::init@79] CGDMLDetector::init PATH DOES NOT EXIST  path
    2020-12-16 19:08:31.194 FATAL [2421758] [Opticks::setSpaceDomain@2609]  changing w 1000 -> 0
    2020-12-16 19:08:31.194 FATAL [2421758] [CTorchSource::configure@163] CTorchSource::configure _t 0.1 _radius 0 _pos 0.0000,0.0000,0.0000 _dir 0.0000,0.0000,1.0000 _zeaz 0.0000,1.0000,0.0000,1.0000 _pol 0.0000,0.0000,1.0000

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff45edf25 in G4RunManagerKernel::DefineWorldVolume(G4VPhysicalVolume*, bool) () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    (gdb) bt
    #0  0x00007ffff45edf25 in G4RunManagerKernel::DefineWorldVolume(G4VPhysicalVolume*, bool) () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    #1  0x00007ffff45cf3a5 in G4RunManager::InitializeGeometry() () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    #2  0x00007ffff45cf221 in G4RunManager::Initialize() () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    #3  0x00007ffff7f315ba in CG4::initialize (this=0x555555608690) at /data2/wenzel/gputest3/opticks/cfg4/CG4.cc:218
    #4  0x00007ffff7f312aa in CG4::init (this=0x555555608690) at /data2/wenzel/gputest3/opticks/cfg4/CG4.cc:190
    #5  0x00007ffff7f30fa2 in CG4::CG4 (this=0x555555608690, hub=0x7fffffffc660) at /data2/wenzel/gputest3/opticks/cfg4/CG4.cc:181
    #6  0x00005555555592e7 in main (argc=4, argv=0x7fffffffcb98) at /data2/wenzel/gputest2/opticks/cfg4/tests/CG4Test.cc:49
    (gdb) quit
    A debugging session is active.

    Inferior 1 [process 2421758] will be killed.

    Quit anyway? (y or n) y



CInterpolationTest : same again
--------------------------------

::

    wenzel@aichi:/data2/wenzel/gputest3/opticks$ gdb --args /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CInterpolationTest --interactive-debug-mode 0 --output-on-failure
    GNU gdb (Ubuntu 9.2-0ubuntu1~20.04) 9.2
    Copyright (C) 2020 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.
    Type "show copying" and "show warranty" for details.
    This GDB was configured as "x86_64-linux-gnu".
    Type "show configuration" for configuration details.
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>.
    Find the GDB manual and other documentation resources online at:
        <http://www.gnu.org/software/gdb/documentation/>.

    For help, type "help".
    Type "apropos word" to search for commands related to "word"...
    Reading symbols from /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CInterpolationTest...
    (gdb) run
    Starting program: /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CInterpolationTest --interactive-debug-mode 0 --output-on-failure
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2020-12-16 19:14:06.124 INFO  [2436073] [main@73] /data2/wenzel/gputest2/local/opticks/build/cfg4/tests/CInterpolationTest
    2020-12-16 19:14:06.124 INFO  [2436073] [BOpticksKey::SetKey@77]  spec G4OpticksTest.X4PhysicalVolume.World_PV.f2f063d9ea288eeab99e0b1617699755
    2020-12-16 19:14:06.124 INFO  [2436073] [Opticks::init@431] INTEROP_MODE hostname aichi
    2020-12-16 19:14:06.125 INFO  [2436073] [Opticks::init@440]  mandatory keyed access to geometry, opticksaux
    2020-12-16 19:14:06.125 INFO  [2436073] [Opticks::init@459] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_ANGULAR WITH_DEBUG_BUFFER WITH_WAY_BUFFER
    2020-12-16 19:14:06.125 ERROR [2436073] [BOpticksKey::SetKey@67] key is already set, ignoring update with spec (null)
    2020-12-16 19:14:06.125 INFO  [2436073] [BOpticksResource::setupViaKey@774]
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : G4OpticksTest.X4PhysicalVolume.World_PV.f2f063d9ea288eeab99e0b1617699755
                     exename  : G4OpticksTest
             current_exename  : CInterpolationTest
                       class  : X4PhysicalVolume
                     volname  : World_PV
                      digest  : f2f063d9ea288eeab99e0b1617699755
                      idname  : G4OpticksTest_World_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2020-12-16 19:14:06.126 INFO  [2436073] [Opticks::loadOriginCacheMeta@1877]  cachemetapath /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f2f063d9ea288eeab99e0b1617699755/1/cachemeta.json
    2020-12-16 19:14:06.126 INFO  [2436073] [BMeta::dump@199] Opticks::loadOriginCacheMeta
    {
        "GEOCACHE_CODE_VERSION": 9,
        "argline": "./G4OpticksTest G4Opticks_50000.gdml muon_noIO.mac ",
        "location": "Opticks::updateCacheMeta",
        "rundate": "20201216_133046",
        "runfolder": "G4OpticksTest",
        "runlabel": "R0_cvd_",
        "runstamp": 1608147046
    }
    2020-12-16 19:14:06.126 INFO  [2436073] [Opticks::loadOriginCacheMeta@1881] ExtractCacheMetaGDMLPath
    2020-12-16 19:14:06.126 INFO  [2436073] [Opticks::loadOriginCacheMeta@1909] (pass) GEOCACHE_CODE_VERSION 9
    2020-12-16 19:14:06.126 INFO  [2436073] [OpticksHub::loadGeometry@282] [ /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f2f063d9ea288eeab99e0b1617699755/1
    2020-12-16 19:14:06.142 INFO  [2436073] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2020-12-16 19:14:06.142 INFO  [2436073] [OpticksHub::loadGeometry@314] ]
    2020-12-16 19:14:06.142 FATAL [2436073] [Opticks::makeSimpleTorchStep@3459]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg
    2020-12-16 19:14:06.142 FATAL [2436073] [OpticksResource::getDefaultFrame@199]  PLACEHOLDER ZERO
    2020-12-16 19:14:06.142 INFO  [2436073] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2020-12-16 19:14:06.142 ERROR [2436073] [OpticksGen::makeTorchstep@441]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0 cmdline_target [--gensteptarget] : 0 gdmlaux_target : -1 active_target : 0
    2020-12-16 19:14:06.142 ERROR [2436073] [OpticksGen::makeTorchstep@468]  generateoverride 0 num_photons0 10000 num_photons 10000
    2020-12-16 19:14:06.142 INFO  [2436073] [BOpticksResource::IsGeant4EnvironmentDetected@291]  n 11 detect 1
    2020-12-16 19:14:06.142 ERROR [2436073] [CG4::preinit@136] External Geant4 environment is detected, not changing this.

    **************************************************************
     Geant4 version Name: geant4-10-06-patch-03 [MT]   (6-November-2020)
                           Copyright : Geant4 Collaboration
                          References : NIM A 506 (2003), 250-303
                                     : IEEE-TNS 53 (2006), 270-278
                                     : NIM A 835 (2016), 186-225
                                 WWW : http://geant4.org/
    **************************************************************

    2020-12-16 19:14:06.161 ERROR [2436073] [BFile::ExistsFile@485] BFile::ExistsFile BAD PATH path  sub NULL name NULL
    2020-12-16 19:14:06.161 ERROR [2436073] [CGDMLDetector::init@79] CGDMLDetector::init PATH DOES NOT EXIST  path
    2020-12-16 19:14:06.161 FATAL [2436073] [Opticks::setSpaceDomain@2609]  changing w 1000 -> 0
    2020-12-16 19:14:06.161 FATAL [2436073] [CTorchSource::configure@163] CTorchSource::configure _t 0.1 _radius 0 _pos 0.0000,0.0000,0.0000 _dir 0.0000,0.0000,1.0000 _zeaz 0.0000,1.0000,0.0000,1.0000 _pol 0.0000,0.0000,1.0000

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff45edf25 in G4RunManagerKernel::DefineWorldVolume(G4VPhysicalVolume*, bool) () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    (gdb) bt
    #0  0x00007ffff45edf25 in G4RunManagerKernel::DefineWorldVolume(G4VPhysicalVolume*, bool) () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    #1  0x00007ffff45cf3a5 in G4RunManager::InitializeGeometry() () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    #2  0x00007ffff45cf221 in G4RunManager::Initialize() () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    #3  0x00007ffff7f315ba in CG4::initialize (this=0x7fffffffc4a0) at /data2/wenzel/gputest3/opticks/cfg4/CG4.cc:218
    #4  0x00007ffff7f312aa in CG4::init (this=0x7fffffffc4a0) at /data2/wenzel/gputest3/opticks/cfg4/CG4.cc:190
    #5  0x00007ffff7f30fa2 in CG4::CG4 (this=0x7fffffffc4a0, hub=0x7fffffffc230) at /data2/wenzel/gputest3/opticks/cfg4/CG4.cc:181
    #6  0x000055555555ab06 in main (argc=4, argv=0x7fffffffcb88) at /data2/wenzel/gputest2/opticks/cfg4/tests/CInterpolationTest.cc:78



OKG4Test : same yet again
---------------------------

::

    wenzel@aichi:/data2/wenzel/gputest3/opticks$ gdb --args /data2/wenzel/gputest2/local/opticks/build/okg4/tests/OKG4Test  --interactive-debug-mode 0 --output-on-failure
    GNU gdb (Ubuntu 9.2-0ubuntu1~20.04) 9.2
    Copyright (C) 2020 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.
    Type "show copying" and "show warranty" for details.
    This GDB was configured as "x86_64-linux-gnu".
    Type "show configuration" for configuration details.
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>.
    Find the GDB manual and other documentation resources online at:
        <http://www.gnu.org/software/gdb/documentation/>.

    For help, type "help".
    Type "apropos word" to search for commands related to "word"...
    Reading symbols from /data2/wenzel/gputest2/local/opticks/build/okg4/tests/OKG4Test...
    (gdb) run
    Starting program: /data2/wenzel/gputest2/local/opticks/build/okg4/tests/OKG4Test --interactive-debug-mode 0 --output-on-failure
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2020-12-16 19:15:49.433 INFO  [2440526] [BOpticksKey::SetKey@77]  spec G4OpticksTest.X4PhysicalVolume.World_PV.f2f063d9ea288eeab99e0b1617699755
    2020-12-16 19:15:49.434 INFO  [2440526] [Opticks::init@431] INTEROP_MODE hostname aichi
    2020-12-16 19:15:49.434 INFO  [2440526] [Opticks::init@440]  mandatory keyed access to geometry, opticksaux
    2020-12-16 19:15:49.434 INFO  [2440526] [Opticks::init@459] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_ANGULAR WITH_DEBUG_BUFFER WITH_WAY_BUFFER
    2020-12-16 19:15:49.434 ERROR [2440526] [BOpticksKey::SetKey@67] key is already set, ignoring update with spec (null)
    2020-12-16 19:15:49.434 INFO  [2440526] [BOpticksResource::setupViaKey@774]
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : G4OpticksTest.X4PhysicalVolume.World_PV.f2f063d9ea288eeab99e0b1617699755
                     exename  : G4OpticksTest
             current_exename  : OKG4Test
                       class  : X4PhysicalVolume
                     volname  : World_PV
                      digest  : f2f063d9ea288eeab99e0b1617699755
                      idname  : G4OpticksTest_World_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2020-12-16 19:15:49.436 INFO  [2440526] [Opticks::loadOriginCacheMeta@1877]  cachemetapath /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f2f063d9ea288eeab99e0b1617699755/1/cachemeta.json
    2020-12-16 19:15:49.436 INFO  [2440526] [BMeta::dump@199] Opticks::loadOriginCacheMeta
    {
        "GEOCACHE_CODE_VERSION": 9,
        "argline": "./G4OpticksTest G4Opticks_50000.gdml muon_noIO.mac ",
        "location": "Opticks::updateCacheMeta",
        "rundate": "20201216_133046",
        "runfolder": "G4OpticksTest",
        "runlabel": "R0_cvd_",
        "runstamp": 1608147046
    }
    2020-12-16 19:15:49.436 INFO  [2440526] [Opticks::loadOriginCacheMeta@1881] ExtractCacheMetaGDMLPath
    2020-12-16 19:15:49.436 INFO  [2440526] [Opticks::loadOriginCacheMeta@1909] (pass) GEOCACHE_CODE_VERSION 9
    2020-12-16 19:15:49.436 INFO  [2440526] [OpticksHub::loadGeometry@282] [ /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f2f063d9ea288eeab99e0b1617699755/1
    2020-12-16 19:15:49.451 INFO  [2440526] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2020-12-16 19:15:49.451 INFO  [2440526] [OpticksHub::loadGeometry@314] ]
    2020-12-16 19:15:49.452 FATAL [2440526] [Opticks::makeSimpleTorchStep@3459]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg
    2020-12-16 19:15:49.452 FATAL [2440526] [OpticksResource::getDefaultFrame@199]  PLACEHOLDER ZERO
    2020-12-16 19:15:49.452 INFO  [2440526] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2020-12-16 19:15:49.452 ERROR [2440526] [OpticksGen::makeTorchstep@441]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0 cmdline_target [--gensteptarget] : 0 gdmlaux_target : -1 active_target : 0
    2020-12-16 19:15:49.452 ERROR [2440526] [OpticksGen::makeTorchstep@468]  generateoverride 0 num_photons0 10000 num_photons 10000
    2020-12-16 19:15:49.452 INFO  [2440526] [BOpticksResource::IsGeant4EnvironmentDetected@291]  n 11 detect 1
    2020-12-16 19:15:49.452 ERROR [2440526] [CG4::preinit@136] External Geant4 environment is detected, not changing this.

    **************************************************************
     Geant4 version Name: geant4-10-06-patch-03 [MT]   (6-November-2020)
                           Copyright : Geant4 Collaboration
                          References : NIM A 506 (2003), 250-303
                                     : IEEE-TNS 53 (2006), 270-278
                                     : NIM A 835 (2016), 186-225
                                 WWW : http://geant4.org/
    **************************************************************

    2020-12-16 19:15:49.471 ERROR [2440526] [BFile::ExistsFile@485] BFile::ExistsFile BAD PATH path  sub NULL name NULL
    2020-12-16 19:15:49.471 ERROR [2440526] [CGDMLDetector::init@79] CGDMLDetector::init PATH DOES NOT EXIST  path
    2020-12-16 19:15:49.471 FATAL [2440526] [Opticks::setSpaceDomain@2609]  changing w 1000 -> 0
    2020-12-16 19:15:49.471 FATAL [2440526] [CTorchSource::configure@163] CTorchSource::configure _t 0.1 _radius 0 _pos 0.0000,0.0000,0.0000 _dir 0.0000,0.0000,1.0000 _zeaz 0.0000,1.0000,0.0000,1.0000 _pol 0.0000,0.0000,1.0000

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff363af25 in G4RunManagerKernel::DefineWorldVolume(G4VPhysicalVolume*, bool) () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    (gdb) bt
    #0  0x00007ffff363af25 in G4RunManagerKernel::DefineWorldVolume(G4VPhysicalVolume*, bool) () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    #1  0x00007ffff361c3a5 in G4RunManager::InitializeGeometry() () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    #2  0x00007ffff361c221 in G4RunManager::Initialize() () from /home/wenzel/geant4.10.06.p03_clhep-install/lib/libG4run.so
    #3  0x00007ffff7a335ba in CG4::initialize (this=0x5555556301f0) at /data2/wenzel/gputest3/opticks/cfg4/CG4.cc:218
    #4  0x00007ffff7a332aa in CG4::init (this=0x5555556301f0) at /data2/wenzel/gputest3/opticks/cfg4/CG4.cc:190
    #5  0x00007ffff7a32fa2 in CG4::CG4 (this=0x5555556301f0, hub=0x555555652dc0) at /data2/wenzel/gputest3/opticks/cfg4/CG4.cc:181
    #6  0x00007ffff7f9918d in OKG4Mgr::OKG4Mgr (this=0x7fffffffc820, argc=4, argv=0x7fffffffcb98) at /data2/wenzel/gputest3/opticks/okg4/OKG4Mgr.cc:107
    #7  0x000055555555901e in main (argc=4, argv=0x7fffffffcb98) at /data2/wenzel/gputest2/opticks/okg4/tests/OKG4Test.cc:27






OSensorLibTest : probably fixed
-----------------------------------

Changed the fail to happen sooner, and found part of cause to be old path. 
SensorLib has moved from opticksgeo down to optickscore::
     
    -    const char* dir = "$TMP/opticksgeo/tests/MockSensorLibTest" ;
    +    const char* dir = "$TMP/optickscore/tests/MockSensorLibTest" ;
         SensorLib* senlib = SensorLib::Load(dir); 

::

    Starting program: /data2/wenzel/gputest2/local/opticks/build/optixrap/tests/OSensorLibTest --interactive-debug-mode 0 --output-on-failure
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2020-12-16 16:18:21.156 INFO  [1724091] [SensorLib::Load@14] $TMP/opticksgeo/tests/MockSensorLibTest
    2020-12-16 16:18:21.156 ERROR [1724091] [NPY<T>::load@954] NPY<T>::load failed for path [/tmp/wenzel/opticks/opticksgeo/tests/MockSensorLibTest/sensorData.npy] use debugload with NPYLoadTest to investigate (problems are usually from dtype mismatches)
    2020-12-16 16:18:21.156 ERROR [1724091] [NPY<T>::load@954] NPY<T>::load failed for path [/tmp/wenzel/opticks/opticksgeo/tests/MockSensorLibTest/angularEfficiency.npy] use debugload with NPYLoadTest to investigate (problems are usually from dtype mismatches)
    2020-12-16 16:18:21.156 INFO  [1724091] [SensorLib::dumpSensorData@76] OSensorLibTest modulo 0
    2020-12-16 16:18:21.156 INFO  [1724091] [SensorLib::dumpSensorData@77] SensorLib closed N loaded Y sensor_data N sensor_num 0 sensor_angular_efficiency N num_category 0
     sensorIndex : efficiency_1 : efficiency_2 :     category :   identifier
    2020-12-16 16:18:21.156 INFO  [1724091] [SensorLib::dumpAngularEfficiency@245] OSensorLibTest sensor_angular_efficiency NULL
    2020-12-16 16:18:21.156 ERROR [1724091] [SensorLib::close@362]  SKIP as m_sensor_num zero
    [New Thread 0x7fffe3e24700 (LWP 1724098)]
    [New Thread 0x7fffe3623700 (LWP 1724099)]
    [New Thread 0x7fffe2c5e700 (LWP 1724100)]

    Thread 1 "OSensorLibTest" received signal SIGSEGV, Segmentation fault.
    0x00007ffff79186c6 in std::vector<int, std::allocator<int> >::size (this=0x8) at /usr/include/c++/9/bits/stl_vector.h:916
    916      { return size_type(this->_M_impl._M_finish - this->_M_impl._M_start); }
    (gdb) bt
    #0  0x00007ffff79186c6 in std::vector<int, std::allocator<int> >::size (this=0x8) at /usr/include/c++/9/bits/stl_vector.h:916
    #1  0x00007ffff6e9f163 in NPYBase::getShape (this=0x0, n=-1) at /data2/wenzel/gputest2/opticks/npy/NPYBase.cpp:513
    #2  0x00007ffff7e349ff in OCtx::create_buffer (this=0x5555555b0a90, arr=0x0, key=0x7ffff7ebe736 "OSensorLib_sensor_data", type=73 'I', flag=32 ' ', item=-1,
        transpose=true) at /data2/wenzel/gputest2/opticks/optixrap/OCtx.cc:181
    #3  0x00007ffff7e3fa30 in OSensorLib::makeSensorDataBuffer (this=0x5555557d99e0) at /data2/wenzel/gputest2/opticks/optixrap/OSensorLib.cc:96
    #4  0x00007ffff7e3f87b in OSensorLib::convert (this=0x5555557d99e0) at /data2/wenzel/gputest2/opticks/optixrap/OSensorLib.cc:84
    #5  0x000055555555d133 in OSensorLibTest::OSensorLibTest (this=0x7fffffffcaa0, senlib=0x5555555b0220)
        at /data2/wenzel/gputest2/opticks/optixrap/tests/OSensorLibTest.cc:30
    #6  0x000055555555dd0d in main (argc=4, argv=0x7fffffffcdb8) at /data2/wenzel/gputest2/opticks/optixrap/tests/OSensorLibTest.cc:125





