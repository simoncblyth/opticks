bialkali-material-with-no-associated-sensor-surface
=====================================================



macOS has the SensorSurface::

    2018-07-16 21:51:34.728 ERROR [362308] [GGeo::loadFromCache@679] GGeo::loadFromCache DONE
    2018-07-16 21:51:34.736 INFO  [362308] [GGeo::loadGeometry@571] GGeo::loadGeometry DONE
    2018-07-16 21:51:34.736 INFO  [362308] [GPropertyLib::close@418] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2018-07-16 21:51:34.737 INFO  [362308] [GPropertyLib::close@418] GPropertyLib::close type GSurfaceLib buf 48,2,39,4
    2018-07-16 21:51:34.738 INFO  [362308] [CPropLib::init@66] CPropLib::init
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 0 name NearPoolCoverSurface pos 7 iss 0
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 1 name NearDeadLinerSurface pos 7 iss 0
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 2 name NearOWSLinerSurface pos 6 iss 0
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 3 name NearIWSCurtainSurface pos 8 iss 0
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 4 name SSTWaterSurfaceNear1 pos 7 iss 0
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 5 name SSTOilSurface pos 0 iss 0
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 6 name lvPmtHemiCathodeSensorSurface pos 16 iss 1
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 7 name lvHeadonPmtCathodeSensorSurface pos 18 iss 1
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 8 name RSOilSurface pos -1 iss 0
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 9 name ESRAirSurfaceTop pos 3 iss 0
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 10 name ESRAirSurfaceBot pos 3 iss 0
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 11 name AdCableTraySurface pos 5 iss 0
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 12 name SSTWaterSurfaceNear2 pos 7 iss 0
    2018-07-16 21:51:34.738 ERROR [362308] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 13 name PmtMtTopRingSurface pos 6 iss 0

but Linux misses em::

    2018-07-16 21:49:00.356 INFO  [45931] [GPropertyLib::close@418] GPropertyLib::close type GSurfaceLib buf 46,2,39,4
    2018-07-16 21:49:00.356 INFO  [45931] [CPropLib::init@66] CPropLib::init
    2018-07-16 21:49:00.356 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 0 name NearPoolCoverSurface pos 7 iss 0
    2018-07-16 21:49:00.356 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 1 name NearDeadLinerSurface pos 7 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 2 name NearOWSLinerSurface pos 6 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 3 name NearIWSCurtainSurface pos 8 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 4 name SSTWaterSurfaceNear1 pos 7 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 5 name SSTOilSurface pos 0 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 6 name RSOilSurface pos -1 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 7 name ESRAirSurfaceTop pos 3 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 8 name ESRAirSurfaceBot pos 3 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 9 name AdCableTraySurface pos 5 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 10 name SSTWaterSurfaceNear2 pos 7 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 11 name PmtMtTopRingSurface pos 6 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 12 name PmtMtBaseRingSurface pos 7 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 13 name PmtMtRib1Surface pos 3 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 14 name PmtMtRib2Surface pos 3 iss 0
    2018-07-16 21:49:00.357 ERROR [45931] [GSurfaceLib::isSensorSurface@977] GSurfaceLib::isSensorSurface surface 15 name PmtMtRib3Surface pos 3 iss 0



Linux 2018/7/16 monday 10/337 fails, mainly in cfg4
----------------------------------------------------


::

    totals  10  / 337 

    FAILS:
      1  /22  Test #1  : CFG4Test.CMaterialLibTest                     ***Exception: Child aborted    0.41   
      2  /22  Test #2  : CFG4Test.CMaterialTest                        ***Exception: Child aborted    0.40   
      3  /22  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    0.98   
      4  /22  Test #4  : CFG4Test.CGDMLDetectorTest                    ***Exception: Child aborted    0.89   
      5  /22  Test #5  : CFG4Test.CGeometryTest                        ***Exception: Child aborted    0.87   
      6  /22  Test #6  : CFG4Test.CG4Test                              ***Exception: Child aborted    0.92   
      17 /22  Test #17 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    0.93   
      19 /22  Test #19 : CFG4Test.CGROUPVELTest                        ***Exception: Child aborted    0.40   
      22 /22  Test #22 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    0.89   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    1.06   


All look to be from same cause of a material named Bialkali with no associated sensor surface.::

    (gdb) bt
    #0  0x00007fffeb8bd277 in raise () from /lib64/libc.so.6
    #1  0x00007fffeb8be968 in abort () from /lib64/libc.so.6
    #2  0x00007fffeb8b6096 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffeb8b6142 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7aff142 in CPropLib::makeMaterialPropertiesTable (this=0x189c820, ggmat=0x6af880) at /home/blyth/opticks/cfg4/CPropLib.cc:228
    #5  0x00007ffff7b0d38b in CMaterialLib::convertMaterial (this=0x189c820, kmat=0x6af880) at /home/blyth/opticks/cfg4/CMaterialLib.cc:198
    #6  0x00007ffff7b0ca34 in CMaterialLib::convert (this=0x189c820) at /home/blyth/opticks/cfg4/CMaterialLib.cc:110
    #7  0x0000000000403514 in main (argc=1, argv=0x7fffffffe118) at /home/blyth/opticks/cfg4/tests/CMaterialTest.cc:48
    (gdb) f 7
    #7  0x0000000000403514 in main (argc=1, argv=0x7fffffffe118) at /home/blyth/opticks/cfg4/tests/CMaterialTest.cc:48
    48      clib->convert();
    (gdb) f 6
    #6  0x00007ffff7b0ca34 in CMaterialLib::convert (this=0x189c820) at /home/blyth/opticks/cfg4/CMaterialLib.cc:110
    110         const G4Material* g4mat = convertMaterial(ggmat);
    (gdb) f 5
    #5  0x00007ffff7b0d38b in CMaterialLib::convertMaterial (this=0x189c820, kmat=0x6af880) at /home/blyth/opticks/cfg4/CMaterialLib.cc:198
    198     G4MaterialPropertiesTable* mpt = makeMaterialPropertiesTable(kmat);
    (gdb) f 4
    #4  0x00007ffff7aff142 in CPropLib::makeMaterialPropertiesTable (this=0x189c820, ggmat=0x6af880) at /home/blyth/opticks/cfg4/CPropLib.cc:228
    228         assert(surf);
    (gdb) 



::

    2018-07-16 21:30:51.245 FATAL [16156] [CPropLib::makeMaterialPropertiesTable@222] m_sensor_surface is obtained from slib at CPropLib::init  when Bialkai material is in the mlib  it is required for a sensor surface (with EFFICIENCY/detect) property  to be in the slib 
    CMaterialTest: /home/blyth/opticks/cfg4/CPropLib.cc:228: G4MaterialPropertiesTable* CPropLib::makeMaterialPropertiesTable(const GMaterial*): Assertion `surf' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffeb8bd277 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glibc-2.17-222.el7.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-19.el7.x86_64 libcom_err-1.42.9-12.el7_5.x86_64 libgcc-4.8.5-28.el7_5.1.x86_64 libicu-50.1.2-15.el7.x86_64 libselinux-2.5-12.el7.x86_64 libstdc++-4.8.5-28.el7_5.1.x86_64 openssl-libs-1.0.2k-12.el7.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-8.el7_2.x86_64 zlib-1.2.7-17.el7.x86_64
    (gdb) bt
    #0  0x00007fffeb8bd277 in raise () from /lib64/libc.so.6
    #1  0x00007fffeb8be968 in abort () from /lib64/libc.so.6
    #2  0x00007fffeb8b6096 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffeb8b6142 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7aff142 in CPropLib::makeMaterialPropertiesTable (this=0x189c820, ggmat=0x6af880) at /home/blyth/opticks/cfg4/CPropLib.cc:228
    #5  0x00007ffff7b0d38b in CMaterialLib::convertMaterial (this=0x189c820, kmat=0x6af880) at /home/blyth/opticks/cfg4/CMaterialLib.cc:198
    #6  0x00007ffff7b0ca34 in CMaterialLib::convert (this=0x189c820) at /home/blyth/opticks/cfg4/CMaterialLib.cc:110
    #7  0x0000000000403514 in main (argc=1, argv=0x7fffffffe118) at /home/blyth/opticks/cfg4/tests/CMaterialTest.cc:48
        (gdb) 
    
    

::

    201 G4MaterialPropertiesTable* CPropLib::makeMaterialPropertiesTable(const GMaterial* ggmat)
    202 {
    203     const char* name = ggmat->getShortName();
    204     GMaterial* _ggmat = const_cast<GMaterial*>(ggmat) ; // wont change it, i promise 
    205 
    206     LOG(trace) << " name " << name ;
    207 
    208 
    209     G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    210     addProperties(mpt, _ggmat, "RINDEX,ABSLENGTH,RAYLEIGH,REEMISSIONPROB,GROUPVEL");
    211 
    212     if(strcmp(name, SENSOR_MATERIAL)==0)
    213     {
    214         GPropertyMap<float>* surf = m_sensor_surface ;
    215 
    216         if(!surf)
    217         {
    218             LOG(fatal) << "CPropLib::makeMaterialPropertiesTable"
    219                        << " material with SENSOR_MATERIAL name " << name
    220                        << " but no sensor_surface "
    221                        ;
    222             LOG(fatal) << "m_sensor_surface is obtained from slib at CPropLib::init "
    223                        << " when Bialkai material is in the mlib "
    224                        << " it is required for a sensor surface (with EFFICIENCY/detect) property "
    225                        << " to be in the slib "
    226                        ;
    227         }
    228         assert(surf);
    229         addProperties(mpt, surf, "EFFICIENCY");
    230 
    231         // REFLECTIVITY ?
    232     }




