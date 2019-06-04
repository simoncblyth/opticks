opticks-t-19-of-398-fails-2019-06-03  FIXED : NOW BACK TO THE USUAL 2 
=============================================================================


opticks-t::

    totals  19  / 398 

    FAILS:
      5  /36  Test #5  : BoostRapTest.BListTest                        Child aborted***Exception:     0.07    
      33 /119 Test #33 : NPYTest.NStateTest                            Child aborted***Exception:     0.08   
      ##  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ from a file not found planted assert in BTree

      12 /28  Test #12 : OpticksCoreTest.OpticksTest                   ***Exception: SegFault         0.09   
      24 /28  Test #24 : OpticksCoreTest.EvtLoadTest                   ***Exception: SegFault         0.08   
      25 /28  Test #25 : OpticksCoreTest.OpticksEventAnaTest           ***Exception: SegFault         0.08   
      26 /28  Test #26 : OpticksCoreTest.OpticksEventCompareTest       ***Exception: SegFault         0.08   
      27 /28  Test #27 : OpticksCoreTest.OpticksEventDumpTest          ***Exception: SegFault         0.08   
      32 /50  Test #32 : GGeoTest.GAttrSeqTest                         ***Exception: SegFault         0.08   
      48 /50  Test #48 : GGeoTest.RecordsNPYTest                       ***Exception: SegFault         0.11   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.96   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.47   
      5  /5   Test #5  : OKTest.TrivialTest                            ***Exception: SegFault         0.20   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     1.20   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     1.13   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     1.10   
      7  /34  Test #7  : CFG4Test.CG4Test                              Child aborted***Exception:     1.17   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     1.18   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    Child aborted***Exception:     1.18   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     1.26   


opticks-t1::

    totals  17  / 398

    FAILS:
      12 /28  Test #12 : OpticksCoreTest.OpticksTest                   ***Exception: SegFault         0.09   
      24 /28  Test #24 : OpticksCoreTest.EvtLoadTest                   ***Exception: SegFault         0.07   
      25 /28  Test #25 : OpticksCoreTest.OpticksEventAnaTest           ***Exception: SegFault         0.09   
      26 /28  Test #26 : OpticksCoreTest.OpticksEventCompareTest       ***Exception: SegFault         0.07   
      27 /28  Test #27 : OpticksCoreTest.OpticksEventDumpTest          ***Exception: SegFault         0.08   
       

      32 /50  Test #32 : GGeoTest.GAttrSeqTest                         ***Exception: SegFault         0.07   
      48 /50  Test #48 : GGeoTest.RecordsNPYTest                       ***Exception: SegFault         0.10   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.90   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.32   
      5  /5   Test #5  : OKTest.TrivialTest                            ***Exception: SegFault         0.18   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     1.15   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     1.09   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     1.09   
      7  /34  Test #7  : CFG4Test.CG4Test                              Child aborted***Exception:     1.25   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     1.16   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    Child aborted***Exception:     1.17   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     1.29   
    [blyth@localhost opticks]$ 



OpticksTest EvtLoadTest
-------------------------------

From the later OpticksEventSpec instanciation::

    2019-06-03 16:08:31.622 INFO  [449860] [main@152] OpticksTest::main aft configure

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff7b0b68e in OpticksEventSpec::getDet (this=0x0) at /home/blyth/opticks/optickscore/OpticksEventSpec.cc:110
    110     return m_det ; 
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff7b0b68e in OpticksEventSpec::getDet (this=0x0) at /home/blyth/opticks/optickscore/OpticksEventSpec.cc:110
    #1  0x00007ffff7b39c72 in Opticks::getDirectGenstepPath (this=0x7fffffffd470) at /home/blyth/opticks/optickscore/Opticks.cc:2374
    #2  0x0000000000405007 in OpticksTest::test_getDirectGenstepPath (this=0x7fffffffd600) at /home/blyth/opticks/optickscore/tests/OpticksTest.cc:51
    #3  0x0000000000403e24 in main (argc=1, argv=0x7fffffffda88) at /home/blyth/opticks/optickscore/tests/OpticksTest.cc:166
    (gdb) 


Moved the defineEventSpec earlier but modify OpticksEventSpec to lazily define the directories.
So they are correct when needed.

Brings to 9 FAILs::

    totals  9   / 398 


    FAILS:
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.87   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.23   
      ^^^^^^^^^^^^^^^^^^ "old" OptiX 6.0.0 torus issue 

      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     1.18   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     1.11   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     1.11   
      7  /34  Test #7  : CFG4Test.CG4Test                              Child aborted***Exception:     1.16   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     1.17   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    Child aborted***Exception:     1.18   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     1.28   
      ^^^^^^^^^^^^^^^^^^^^^ all these 7 from unexpected added materials  



CTestDetectorTest CGDMLDetectorTest CGeometryTest CG4Test CInterpolationTest CRandomEngineTest OKG4Test::

    G4GDML: Reading '/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml' done!
    2019-06-03 17:12:53.001 FATAL [131832] [CMaterialSort::sort@64]  sorting G4MaterialTable using order kv 38
    2019-06-03 17:12:53.002 INFO  [131832] [CDetector::setTop@94] .
    2019-06-03 17:12:53.260 INFO  [131832] [CTraverser::Summary@106] CDetector::traverse numMaterials 36 numMaterialsWithoutMPT 36
    2019-06-03 17:12:53.260 ERROR [131832] [CGDMLDetector::addMPTLegacyGDML@164]  ALL G4 MATERIALS LACK MPT  FIXING USING Opticks MATERIALS 
    2019-06-03 17:12:53.262 ERROR [131832] [CPropLib::addConstProperty@376]  OVERRIDE GdDopedLS.SCINTILLATIONYIELD from 11522 to 10
    2019-06-03 17:12:53.263 ERROR [131832] [CPropLib::addConstProperty@376]  OVERRIDE LiquidScintillator.SCINTILLATIONYIELD from 11522 to 10
    2019-06-03 17:12:53.263 ERROR [131832] [CPropLib::makeMaterialPropertiesTable@249]  name Bialkali adding EFFICIENCY : START GPropertyMap  type skinsurface name /dd/Geometry/PMT/lvHeadonPmtCathodeSensorSurface
    2019-06-03 17:12:53.266 INFO  [131832] [CGDMLDetector::addMPTLegacyGDML@202] CGDMLDetector::addMPT added MPT to 36 g4 materials 
    2019-06-03 17:12:53.266 INFO  [131832] [CGDMLDetector::standardizeGeant4MaterialProperties@218] [
    CTestDetectorTest: /home/blyth/opticks/extg4/X4MaterialLib.cc:64: void X4MaterialLib::init(): Assertion `num_materials == num_m4' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffe9d92207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe9d92207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe9d938f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe9d8b026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe9d8b0d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff77e7d90 in X4MaterialLib::init (this=0x7fffffffc0e0) at /home/blyth/opticks/extg4/X4MaterialLib.cc:64
    #5  0x00007ffff77e7d35 in X4MaterialLib::X4MaterialLib (this=0x7fffffffc0e0, mtab=0x7ffff06580c0 <G4Material::theMaterialTable>, mlib=0x6b72a0) at /home/blyth/opticks/extg4/X4MaterialLib.cc:56
    #6  0x00007ffff77e7cfb in X4MaterialLib::Standardize (mtab=0x7ffff06580c0 <G4Material::theMaterialTable>, mlib=0x6b72a0) at /home/blyth/opticks/extg4/X4MaterialLib.cc:47
    #7  0x00007ffff77e7cd1 in X4MaterialLib::Standardize () at /home/blyth/opticks/extg4/X4MaterialLib.cc:42
    #8  0x00007ffff7b35eff in CGDMLDetector::standardizeGeant4MaterialProperties (this=0x1b8c770) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:219
    #9  0x00007ffff7b3557b in CGDMLDetector::init (this=0x1b8c770) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:78
    #10 0x00007ffff7b351c4 in CGDMLDetector::CGDMLDetector (this=0x1b8c770, hub=0x7fffffffd020, query=0x6ab710, sd=0x1b8a110) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:43
    #11 0x00007ffff7adc7c1 in CGeometry::init (this=0x1b8c6c0) at /home/blyth/opticks/cfg4/CGeometry.cc:77
    #12 0x00007ffff7adc5cc in CGeometry::CGeometry (this=0x1b8c6c0, hub=0x7fffffffd020, sd=0x1b8a110) at /home/blyth/opticks/cfg4/CGeometry.cc:60
    #13 0x00007ffff7b4ca5b in CG4::CG4 (this=0x7fffffffd260, hub=0x7fffffffd020) at /home/blyth/opticks/cfg4/CG4.cc:121
    #14 0x000000000040369e in main (argc=1, argv=0x7fffffffda78) at /home/blyth/opticks/cfg4/tests/CTestDetectorTest.cc:58
    (gdb) f 4
    #4  0x00007ffff77e7d90 in X4MaterialLib::init (this=0x7fffffffc0e0) at /home/blyth/opticks/extg4/X4MaterialLib.cc:64
    64      assert( num_materials == num_m4 ); 
    (gdb) p num_materials
    $1 = 38
    (gdb) p num_m4
    $2 = 36
    (gdb) 


::

    2019-06-03 17:21:15.817 ERROR [146213] [CGDMLDetector::addMPTLegacyGDML@164]  ALL G4 MATERIALS LACK MPT  FIXING USING Opticks MATERIALS 
    2019-06-03 17:21:15.819 ERROR [146213] [CPropLib::addConstProperty@376]  OVERRIDE GdDopedLS.SCINTILLATIONYIELD from 11522 to 10
    2019-06-03 17:21:15.820 ERROR [146213] [CPropLib::addConstProperty@376]  OVERRIDE LiquidScintillator.SCINTILLATIONYIELD from 11522 to 10
    2019-06-03 17:21:15.820 ERROR [146213] [CPropLib::makeMaterialPropertiesTable@249]  name Bialkali adding EFFICIENCY : START GPropertyMap  type skinsurface name /dd/Geometry/PMT/lvHeadonPmtCathodeSensorSurface
    2019-06-03 17:21:15.823 INFO  [146213] [CGDMLDetector::addMPTLegacyGDML@202] CGDMLDetector::addMPT added MPT to 36 g4 materials 
    2019-06-03 17:21:15.823 INFO  [146213] [CGDMLDetector::standardizeGeant4MaterialProperties@218] [
    2019-06-03 17:21:15.823 FATAL [146213] [X4MaterialLib::init@67]  num_materials MISMATCH  G4Material::GetNumberOfMaterials 36 m_mlib->getNumMaterials 38
    OKG4Test: /home/blyth/opticks/extg4/X4MaterialLib.cc:73: void X4MaterialLib::init(): Assertion `match' failed.
    Aborted (core dumped)


::

    [blyth@localhost cfg4]$ opticks-f addTestMaterials
    ./extg4/X4PhysicalVolume.cc:    //m_mlib->addTestMaterials() ;
    ./ggeo/tests/GMaterialLibTest.cc:    // see GGeo::addTestMaterials
    ./ggeo/GGeo.cc:    mlib->addTestMaterials(); 
    ./ggeo/GMaterialLib.hh:       void addTestMaterials();
    ./ggeo/GGeoTest.cc:    m_mlib->addTestMaterials(); 
    ./ggeo/GMaterialLib.cc:void GMaterialLib::addTestMaterials()
    ./ggeo/GMaterialLib.cc:        LOG(info) << "GMaterialLib::addTestMaterials" 
    [blyth@localhost opticks]$ 


Two test materials::

    1043 void GMaterialLib::addTestMaterials()
    1044 {
    1045     typedef std::pair<std::string, std::string> SS ;
    1046     typedef std::vector<SS> VSS ;
    1047 
    1048     VSS rix ;
    1049 
    1050     rix.push_back(SS("GlassSchottF2", "$OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/glass/schott/F2.npy"));
    1051     rix.push_back(SS("MainH2OHale",   "$OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/main/H2O/Hale.npy"));
    1052 



Why now this issue, i didnt recreate the old geocache.  No but I did add abbreviations::

    [blyth@localhost GMaterialLib]$ jsn.py GPropertyLibMetadata.json | wc -l
    38
    [blyth@localhost GMaterialLib]$ pwd
    /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMaterialLib
    [blyth@localhost GMaterialLib]$ 

No thats not it, the old legacy geocache has 38 materials::

    [blyth@localhost GItemList]$ wc -l GMaterialLib.txt
    38 GMaterialLib.txt
    [blyth@localhost GItemList]$ l GMaterialLib.txt
    -rw-rw-r--. 1 blyth blyth 332 Oct 15  2018 GMaterialLib.txt


The question is why the X4 code is running on it ? The standardization is new for ckm matching::

    [blyth@localhost okg4]$ gdb OKG4Test 
    ...
    2019-06-03 20:29:47.813 INFO  [446288] [OpticksHub::loadGeometry@490] [ /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2019-06-03 20:29:47.813 ERROR [446288] [GGeo::init@433]  idpath /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae cache_exists 1 cache_requested 1 m_loaded 1 m_live 0
    2019-06-03 20:29:47.971 ERROR [446288] [GGeo::loadCacheMeta@759] /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/cachemeta.json
    ...
    2019-06-03 20:29:48.035 FATAL [446288] [CGeometry::init@75] G4 GDML geometry 
    2019-06-03 20:29:48.035 INFO  [446288] [CPropLib::init@68] [
    2019-06-03 20:29:48.035 INFO  [446288] [CPropLib::init@70] GSurfaceLib numSurfaces 48 this 0x74aa80 basis 0 isClosed 1 hasDomain 1
    2019-06-03 20:29:48.035 INFO  [446288] [CPropLib::init@93] ]
    2019-06-03 20:29:48.035 INFO  [446288] [CSurfaceLib::CSurfaceLib@37] .
    2019-06-03 20:29:48.035 INFO  [446288] [CDetector::init@84] .
    2019-06-03 20:29:48.035 INFO  [446288] [CGDMLDetector::CGDMLDetector@42] [
    2019-06-03 20:29:48.035 INFO  [446288] [CGDMLDetector::init@69] parse /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml
    G4GDML: Reading '/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml' done!
    2019-06-03 20:29:48.429 FATAL [446288] [CMaterialSort::sort@64]  sorting G4MaterialTable using order kv 38
    2019-06-03 20:29:48.429 INFO  [446288] [CDetector::setTop@94] .
    2019-06-03 20:29:48.636 INFO  [446288] [CTraverser::Summary@106] CDetector::traverse numMaterials 36 numMaterialsWithoutMPT 36
    2019-06-03 20:29:48.637 ERROR [446288] [CGDMLDetector::addMPTLegacyGDML@164]  ALL G4 MATERIALS LACK MPT  FIXING USING Opticks MATERIALS 
    2019-06-03 20:29:48.638 ERROR [446288] [CPropLib::addConstProperty@376]  OVERRIDE GdDopedLS.SCINTILLATIONYIELD from 11522 to 10
    2019-06-03 20:29:48.639 ERROR [446288] [CPropLib::addConstProperty@376]  OVERRIDE LiquidScintillator.SCINTILLATIONYIELD from 11522 to 10
    2019-06-03 20:29:48.639 ERROR [446288] [CPropLib::makeMaterialPropertiesTable@249]  name Bialkali adding EFFICIENCY : START GPropertyMap  type skinsurface name /dd/Geometry/PMT/lvHeadonPmtCathodeSensorSurface
    2019-06-03 20:29:48.642 INFO  [446288] [CGDMLDetector::addMPTLegacyGDML@202] CGDMLDetector::addMPT added MPT to 36 g4 materials 
    2019-06-03 20:29:48.642 INFO  [446288] [CGDMLDetector::standardizeGeant4MaterialProperties@218] [
    2019-06-03 20:29:48.642 FATAL [446288] [X4MaterialLib::init@67]  num_materials MISMATCH  G4Material::GetNumberOfMaterials 36 m_mlib->getNumMaterials 38
    OKG4Test: /home/blyth/opticks/extg4/X4MaterialLib.cc:73: void X4MaterialLib::init(): Assertion `match' failed.
    
    Program received signal SIGABRT, Aborted.
    ...
    (gdb) bt
    #0  0x00007fffe2031207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe20328f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe202a026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe202a0d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffefa86e92 in X4MaterialLib::init (this=0x7fffffffc7e0) at /home/blyth/opticks/extg4/X4MaterialLib.cc:73
    #5  0x00007fffefa86d35 in X4MaterialLib::X4MaterialLib (this=0x7fffffffc7e0, mtab=0x7fffe88f70c0 <G4Material::theMaterialTable>, mlib=0x6ce580) at /home/blyth/opticks/extg4/X4MaterialLib.cc:56
    #6  0x00007fffefa86cfb in X4MaterialLib::Standardize (mtab=0x7fffe88f70c0 <G4Material::theMaterialTable>, mlib=0x6ce580) at /home/blyth/opticks/extg4/X4MaterialLib.cc:47
    #7  0x00007fffefa86cd1 in X4MaterialLib::Standardize () at /home/blyth/opticks/extg4/X4MaterialLib.cc:42
    #8  0x00007fffefdd4eff in CGDMLDetector::standardizeGeant4MaterialProperties (this=0x1ba2c50) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:219
    #9  0x00007fffefdd457b in CGDMLDetector::init (this=0x1ba2c50) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:78
    #10 0x00007fffefdd41c4 in CGDMLDetector::CGDMLDetector (this=0x1ba2c50, hub=0x6b5df0, query=0x6c1610, sd=0x1ba05f0) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:43
    #11 0x00007fffefd7b7c1 in CGeometry::init (this=0x1ba2ba0) at /home/blyth/opticks/cfg4/CGeometry.cc:77
    #12 0x00007fffefd7b5cc in CGeometry::CGeometry (this=0x1ba2ba0, hub=0x6b5df0, sd=0x1ba05f0) at /home/blyth/opticks/cfg4/CGeometry.cc:60
    #13 0x00007fffefdeba5b in CG4::CG4 (this=0x19c02d0, hub=0x6b5df0) at /home/blyth/opticks/cfg4/CG4.cc:121
    #14 0x00007ffff7bd5256 in OKG4Mgr::OKG4Mgr (this=0x7fffffffd760, argc=1, argv=0x7fffffffda98) at /home/blyth/opticks/okg4/OKG4Mgr.cc:76
    #15 0x0000000000403998 in main (argc=1, argv=0x7fffffffda98) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    (gdb) f 14
    #14 0x00007ffff7bd5256 in OKG4Mgr::OKG4Mgr (this=0x7fffffffd760, argc=1, argv=0x7fffffffda98) at /home/blyth/opticks/okg4/OKG4Mgr.cc:76
    76      m_g4(m_load ? NULL : new CG4(m_hub)),   // configure and initialize immediately 
    (gdb) f 13
    #13 0x00007fffefdeba5b in CG4::CG4 (this=0x19c02d0, hub=0x6b5df0) at /home/blyth/opticks/cfg4/CG4.cc:121
    121     m_geometry(new CGeometry(m_hub, m_sd)),
    (gdb) f 12
    #12 0x00007fffefd7b5cc in CGeometry::CGeometry (this=0x1ba2ba0, hub=0x6b5df0, sd=0x1ba05f0) at /home/blyth/opticks/cfg4/CGeometry.cc:60
    60      init();
    (gdb) f 11
    #11 0x00007fffefd7b7c1 in CGeometry::init (this=0x1ba2ba0) at /home/blyth/opticks/cfg4/CGeometry.cc:77
    77          detector  = static_cast<CDetector*>(new CGDMLDetector(m_hub, query, m_sd)) ; 
    (gdb) f 10
    #10 0x00007fffefdd41c4 in CGDMLDetector::CGDMLDetector (this=0x1ba2c50, hub=0x6b5df0, query=0x6c1610, sd=0x1ba05f0) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:43
    warning: Source file is more recent than executable.
    43      init();
    (gdb) f 9
    #9  0x00007fffefdd457b in CGDMLDetector::init (this=0x1ba2c50) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:78
    78      standardizeGeant4MaterialProperties();
    (gdb) f 8
    #8  0x00007fffefdd4eff in CGDMLDetector::standardizeGeant4MaterialProperties (this=0x1ba2c50) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:219
    219     X4MaterialLib::Standardize() ;
    (gdb) f 7
    #7  0x00007fffefa86cd1 in X4MaterialLib::Standardize () at /home/blyth/opticks/extg4/X4MaterialLib.cc:42
    42      X4MaterialLib::Standardize( mtab, mlib ) ; 
    (gdb) f 6
    #6  0x00007fffefa86cfb in X4MaterialLib::Standardize (mtab=0x7fffe88f70c0 <G4Material::theMaterialTable>, mlib=0x6ce580) at /home/blyth/opticks/extg4/X4MaterialLib.cc:47
    47      X4MaterialLib xmlib(mtab, mlib) ;  
    (gdb) f 5
    #5  0x00007fffefa86d35 in X4MaterialLib::X4MaterialLib (this=0x7fffffffc7e0, mtab=0x7fffe88f70c0 <G4Material::theMaterialTable>, mlib=0x6ce580) at /home/blyth/opticks/extg4/X4MaterialLib.cc:56
    56      init();
    (gdb) f 4
    #4  0x00007fffefa86e92 in X4MaterialLib::init (this=0x7fffffffc7e0) at /home/blyth/opticks/extg4/X4MaterialLib.cc:73
    73      assert( match ); 
    (gdb) 



::

     52 void CGDMLDetector::init()
     53 {
     54     const char* path = m_ok->getCurrentGDMLPath() ;
     55 
     56     bool exists = BFile::ExistsFile(path);
     57     if( !exists )
     58     {
     59          LOG(error)
     60               << "CGDMLDetector::init"
     61               << " PATH DOES NOT EXIST "
     62               << " path " << path
     63               ;
     64 
     65          setValid(false);
     66          return ;
     67     }
     68 
     69     LOG(m_level) << "parse " << path ;
     70 
     71     G4VPhysicalVolume* world = parseGDML(path);
     72 
     73     sortMaterials();
     74 
     75     setTop(world);   // invokes *CDetector::traverse*
     76 
     77     addMPTLegacyGDML();
     78     standardizeGeant4MaterialProperties();
     79 
     80     attachSurfaces();
     81     // kludge_cathode_efficiency(); 
     82 
     83     hookupSD();
     84 
     85 }


The standarize is new::

    207 /**
    208 CGDMLDetector::standardizeGeant4MaterialProperties
    209 -----------------------------------------------------
    210 
    211 Duplicates G4Opticks::standardizeGeant4MaterialProperties
    212 
    213 **/
    214 
    215 
    216 void CGDMLDetector::standardizeGeant4MaterialProperties()
    217 {
    218     LOG(info) << "[" ;
    219     X4MaterialLib::Standardize() ;
    220     LOG(info) << "]" ;
    221 }
    222 
    223 



For legacy GDML this has some issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 2 extra OK materials (GlassSchottF2, MainH2OHale)  : the test glass comes after Air in the middle 
2. g4 material names are prefixed /dd/Materials/GdDopedLS




So for now just skip it for legacy::

     80     if(m_ok->isLegacy())
     81     {
     82         LOG(error) << " skip standardizeGeant4MaterialProperties in legacy running " ;
     83     }
     84     else
     85     {
     86         standardizeGeant4MaterialProperties();
     87     }
     88 
     89     attachSurfaces();
     90     // kludge_cathode_efficiency(); 
     91 
     92     hookupSD();
     93 
     94 }




After skipping the standardization for legacy have 5 fails, 2 expected
---------------------------------------------------------------------------

::

    totals  5   / 398 
    FAILS:
      1  /3   Test #1  : AssimpRapTest.AssimpRapTest                   ***Exception: Interrupt        0.81   
      3  /3   Test #3  : AssimpRapTest.AssimpGGeoTest                  ***Exception: Interrupt        0.77   
      3  /3   Test #3  : OpticksGeoTest.OpenMeshRapTest                ***Exception: Interrupt        0.75   

      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.90   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.24   


All three from same place, a planted std::raise(SIGINT) to find who calls GMaterialLib::addTestMaterials::

    2019-06-03 21:02:33.361 ERROR [56855] [AssimpGGeo::convertMaterials@451]  osnam                __dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib3Surface ostyp 0 osmod 1 osfin 3 osval 1
    2019-06-03 21:02:33.361 ERROR [56855] [AssimpGGeo::convertMaterials@451]  osnam                __dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib4Surface ostyp 0 osmod 1 osfin 3 osval 1
    2019-06-03 21:02:33.361 ERROR [56855] [AssimpGGeo::convertMaterials@451]  osnam                __dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib5Surface ostyp 0 osmod 1 osfin 3 osval 1
    2019-06-03 21:02:33.361 ERROR [56855] [AssimpGGeo::convertMaterials@451]  osnam                __dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib8Surface ostyp 0 osmod 1 osfin 3 osval 1
    2019-06-03 21:02:33.361 ERROR [56855] [AssimpGGeo::convertMaterials@451]  osnam                __dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib9Surface ostyp 0 osmod 1 osfin 3 osval 1
    2019-06-03 21:02:33.361 ERROR [56855] [AssimpGGeo::convertMaterials@451]  osnam              __dd__Geometry__PoolDetails__PoolSurfacesAll__VertiCableTraySurface ostyp 0 osmod 1 osfin 3 osval 1
    2019-06-03 21:02:33.362 ERROR [56855] [GMaterialLib::add@287]  MATERIAL WITH EFFICIENCY 
    2019-06-03 21:02:33.362 FATAL [56855] [GMaterialLib::setCathode@1096]  have already set that cathode GMaterial : __dd__Materials__Bialkali0xc2f2428

    Program received signal SIGINT, Interrupt.
    0x00007ffff2f87207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff2f87207 in raise () from /lib64/libc.so.6
    #1  0x00007ffff6be493d in GMaterialLib::addTestMaterials (this=0x636180) at /home/blyth/opticks/ggeo/GMaterialLib.cc:1046
    #2  0x00007ffff6c51855 in GGeo::prepareMaterialLib (this=0x635950) at /home/blyth/opticks/ggeo/GGeo.cc:1173
    #3  0x00007ffff6c50715 in GGeo::afterConvertMaterials (this=0x635950) at /home/blyth/opticks/ggeo/GGeo.cc:850
    #4  0x00007ffff7bc66f2 in AssimpGGeo::convert (this=0x7fffffffc710, ctrl=0x7ffff5bb4c53 "") at /home/blyth/opticks/assimprap/AssimpGGeo.cc:194
    #5  0x00007ffff7bc653f in AssimpGGeo::load (ggeo=0x635950) at /home/blyth/opticks/assimprap/AssimpGGeo.cc:181
    #6  0x00007ffff6c4ebfe in GGeo::loadFromG4DAE (this=0x635950) at /home/blyth/opticks/ggeo/GGeo.cc:625
    #7  0x0000000000403f67 in main (argc=1, argv=0x7fffffffda88) at /home/blyth/opticks/assimprap/tests/AssimpRapTest.cc:69
    (gdb) 

::

    (gdb) bt
    #0  0x00007ffff2f87207 in raise () from /lib64/libc.so.6
    #1  0x00007ffff6be493d in GMaterialLib::addTestMaterials (this=0x636180) at /home/blyth/opticks/ggeo/GMaterialLib.cc:1046
    #2  0x00007ffff6c51855 in GGeo::prepareMaterialLib (this=0x635950) at /home/blyth/opticks/ggeo/GGeo.cc:1173
    #3  0x00007ffff6c50715 in GGeo::afterConvertMaterials (this=0x635950) at /home/blyth/opticks/ggeo/GGeo.cc:850
    #4  0x00007ffff7bc66f2 in AssimpGGeo::convert (this=0x7fffffffc710, ctrl=0x7ffff5bb4c53 "") at /home/blyth/opticks/assimprap/AssimpGGeo.cc:194
    #5  0x00007ffff7bc653f in AssimpGGeo::load (ggeo=0x635950) at /home/blyth/opticks/assimprap/AssimpGGeo.cc:181
    #6  0x00007ffff6c4ebfe in GGeo::loadFromG4DAE (this=0x635950) at /home/blyth/opticks/ggeo/GGeo.cc:625
    #7  0x0000000000403f67 in main (argc=1, argv=0x7fffffffda88) at /home/blyth/opticks/assimprap/tests/AssimpRapTest.cc:69
    (gdb) f 7
    #7  0x0000000000403f67 in main (argc=1, argv=0x7fffffffda88) at /home/blyth/opticks/assimprap/tests/AssimpRapTest.cc:69
    69          m_ggeo->loadFromG4DAE();
    (gdb) f 6
    #6  0x00007ffff6c4ebfe in GGeo::loadFromG4DAE (this=0x635950) at /home/blyth/opticks/ggeo/GGeo.cc:625
    625         int rc = (*m_loader_imp)(this);   //  imp set in OpticksGeometry::loadGeometryBase, m_ggeo->setLoaderImp(&AssimpGGeo::load); 
    (gdb) f 5
    #5  0x00007ffff7bc653f in AssimpGGeo::load (ggeo=0x635950) at /home/blyth/opticks/assimprap/AssimpGGeo.cc:181
    181         int rc = agg.convert(ctrl);
    (gdb) f 4
    #4  0x00007ffff7bc66f2 in AssimpGGeo::convert (this=0x7fffffffc710, ctrl=0x7ffff5bb4c53 "") at /home/blyth/opticks/assimprap/AssimpGGeo.cc:194
    194         m_ggeo->afterConvertMaterials(); 
    (gdb) f 3
    #3  0x00007ffff6c50715 in GGeo::afterConvertMaterials (this=0x635950) at /home/blyth/opticks/ggeo/GGeo.cc:850
    850         prepareMaterialLib(); 
    (gdb) f 2
    #2  0x00007ffff6c51855 in GGeo::prepareMaterialLib (this=0x635950) at /home/blyth/opticks/ggeo/GGeo.cc:1173
    1173        mlib->addTestMaterials(); 
    (gdb) f 1
    #1  0x00007ffff6be493d in GMaterialLib::addTestMaterials (this=0x636180) at /home/blyth/opticks/ggeo/GMaterialLib.cc:1046
    1046        std::raise(SIGINT); 
    (gdb) 



Removing the plant down to 2
----------------------------------

::

    FAILS:
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.82   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.25   



