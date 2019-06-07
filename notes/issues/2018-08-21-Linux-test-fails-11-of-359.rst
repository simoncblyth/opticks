2018-08-21-Linux-test-fails-11-of-359
=========================================

::

     totals  11  / 359 

    FAILS:
      1  /1   Test #1  : OpticksGLTest.OOAxisAppCheck                  ***Failed                      0.23   
      1  /24  Test #1  : CFG4Test.CMaterialLibTest                     ***Exception: Child aborted    0.44   
      2  /24  Test #2  : CFG4Test.CMaterialTest                        ***Exception: Child aborted    0.43   
      3  /24  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    1.13   
      5  /24  Test #5  : CFG4Test.CGDMLDetectorTest                    ***Exception: Child aborted    1.05   
      6  /24  Test #6  : CFG4Test.CGeometryTest                        ***Exception: Child aborted    1.05   
      7  /24  Test #7  : CFG4Test.CG4Test                              ***Exception: Child aborted    1.14   
      19 /24  Test #19 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    1.14   
      21 /24  Test #21 : CFG4Test.CGROUPVELTest                        ***Exception: Child aborted    0.48   
      24 /24  Test #24 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    1.11   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    1.99   



Other than OOAxisAppCheck 10 of the fails are from the same problem
----------------------------------------------------------------------

::

    2018-08-21 22:32:15.261 FATAL [410047] [CPropLib::makeMaterialPropertiesTable@222] CPropLib::makeMaterialPropertiesTable material with SENSOR_MATERIAL name Bialkali but no sensor_surface 
    2018-08-21 22:32:15.261 FATAL [410047] [CPropLib::makeMaterialPropertiesTable@226] m_sensor_surface is obtained from slib at CPropLib::init  when Bialkai material is in the mlib  it is required for a sensor surface (with EFFICIENCY/detect) property  to be in the slib 
    CMaterialLibTest: /home/blyth/opticks/cfg4/CPropLib.cc:232: G4MaterialPropertiesTable* CPropLib::makeMaterialPropertiesTable(const GMaterial*): Assertion `surf' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffe9e25277 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glibc-2.17-222.el7.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-19.el7.x86_64 libcom_err-1.42.9-12.el7_5.x86_64 libgcc-4.8.5-28.el7_5.1.x86_64 libicu-50.1.2-15.el7.x86_64 libselinux-2.5-12.el7.x86_64 libstdc++-4.8.5-28.el7_5.1.x86_64 openssl-libs-1.0.2k-12.el7.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-8.el7_2.x86_64 zlib-1.2.7-17.el7.x86_64
    (gdb) bt
    #0  0x00007fffe9e25277 in raise () from /lib64/libc.so.6
    #1  0x00007fffe9e26968 in abort () from /lib64/libc.so.6
    #2  0x00007fffe9e1e096 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe9e1e142 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7af4d9c in CPropLib::makeMaterialPropertiesTable (this=0x18f3880, ggmat=0x6c3630) at /home/blyth/opticks/cfg4/CPropLib.cc:232
    #5  0x00007ffff7b05d51 in CMaterialLib::convertMaterial (this=0x18f3880, kmat=0x6c3630) at /home/blyth/opticks/cfg4/CMaterialLib.cc:198
    #6  0x00007ffff7b053fa in CMaterialLib::convert (this=0x18f3880) at /home/blyth/opticks/cfg4/CMaterialLib.cc:110
    #7  0x0000000000404a22 in main (argc=1, argv=0x7fffffffde28) at /home/blyth/opticks/cfg4/tests/CMaterialLibTest.cc:122
    (gdb) 



