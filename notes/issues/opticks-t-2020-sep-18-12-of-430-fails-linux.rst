opticks-t-2020-sep-18-12-of-430-fails-linux
==================================================

::

    opticks-t

    FAILS:  12  / 430   :  Fri Sep 18 22:31:35 2020   
      32 /32  Test #32 : OpticksCoreTest.IntersectSDFTest              ***Exception: SegFault         0.06   

            DONE : prevent this failing for non-existing inputs 

      30 /53  Test #30 : GGeoTest.GPtsTest                             Child aborted***Exception:     0.58   

            Failing on first mm 0  

            2020-09-18 23:51:16.192 INFO  [237539] [Opticks::loadOriginCacheMeta@1853]  gdmlpath 
            2020-09-18 23:51:16.473 INFO  [237539] [main@141]  geolib.nmm 10
            GPtsTest: /home/blyth/opticks/ggeo/tests/GPtsTest.cc:84: void testGPts::init(): Assertion `parts' failed.
            Aborted (core dumped)
           
            #3  0x00007ffff3bf30d2 in __assert_fail () from /lib64/libc.so.6
            #4  0x0000000000405378 in testGPts::init (this=0x7fffffffab00) at /home/blyth/opticks/ggeo/tests/GPtsTest.cc:84
            #5  0x0000000000405307 in testGPts::testGPts (this=0x7fffffffab00, meshlib_=0x636ae0, bndlib_=0xb729d0, mm_=0xcb9930) at /home/blyth/opticks/ggeo/tests/GPtsTest.cc:77
            #6  0x0000000000404032 in main (argc=1, argv=0x7fffffffb1a8) at /home/blyth/opticks/ggeo/tests/GPtsTest.cc:152
            (gdb) 

            GGeoLib::loadConstituents should be loading and associating these     

            GGeoLib=INFO GPtsTest 

            Suspect can no longer do this comparison as the GParts has been dropped ?



      21 /28  Test #21 : OptiXRapTest.interpolationTest                ***Failed                      10.43  

           fails for lack of numpy in the python (juno) picked off PATH
           easy to kludge eg using python3, but what is the definitive solution ?  

           * added SSys::RunPythonScript and SSys:ResolvePython to fix this kind of problem definitively (hopefully)
             by making sensitive to OPTICKS_PYTHON envvar to pick the python

           opticks-c python


      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     9.92   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: SegFault         1.10   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     1.13   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     1.10   
      7  /34  Test #7  : CFG4Test.CG4Test                              ***Exception: SegFault         1.16   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   ***Exception: SegFault         1.13   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    ***Exception: SegFault         1.10   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: SegFault         1.49   



      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.34   
    [blyth@localhost opticks]$ date
    Fri Sep 18 22:39:03 CST 2020

