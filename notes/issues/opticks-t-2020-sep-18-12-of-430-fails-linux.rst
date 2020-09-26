opticks-t-2020-sep-18-12-of-430-fails-linux
==================================================





Adding OPTICKS_PYTHON to pick the python with numpy reduces to 10::

    FAILS:  10  / 430   :  Sat Sep 26 23:03:56 2020   
      30 /53  Test #30 : GGeoTest.GPtsTest                             ***Failed                      0.37   

            cannot compare : suspect deferred GParts as standard makes this test useless 
            for now switch off the fail, and see if this is correct

      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     10.04  

        2020-09-26 23:11:19.777 ERROR [146867] [G4StepNPY::checkGencodes@272]  i 0 unexpected label 4096
        2020-09-26 23:11:19.777 FATAL [146867] [G4StepNPY::checkGencodes@283] G4StepNPY::checklabel FAIL numStep 1 mismatch 1
        OKTest: /home/blyth/opticks/npy/G4StepNPY.cpp:288: void G4StepNPY::checkGencodes(): Assertion `mismatch == 0' failed.

        2020-09-26 23:26:26.079 ERROR [172407] [G4StepNPY::checkGencodes@281]  i 0 unexpected gencode label 4096 allowed gencodes 5,
        2020-09-26 23:26:26.079 FATAL [172407] [G4StepNPY::checkGencodes@293] G4StepNPY::checklabel FAIL numStep 1 mismatch 1
        OKTest: /home/blyth/opticks/npy/G4StepNPY.cpp:298: void G4StepNPY::checkGencodes(): Assertion `mismatch == 0' failed.



        #3  0x00007fffeacb40d2 in __assert_fail () from /lib64/libc.so.6
        #4  0x00007ffff29645a4 in G4StepNPY::checkGencodes (this=0x225ad8c0) at /home/blyth/opticks/npy/G4StepNPY.cpp:288
        #5  0x00007ffff2e7b1bf in OpticksRun::importGenstepData (this=0x678a60, gs=0x57684e0, oac_label=0x0) at /home/blyth/opticks/optickscore/OpticksRun.cc:423
        #6  0x00007ffff2e7a396 in OpticksRun::importGensteps (this=0x678a60) at /home/blyth/opticks/optickscore/OpticksRun.cc:253
        #7  0x00007ffff2e7a290 in OpticksRun::setGensteps (this=0x678a60, gensteps=0x57684e0) at /home/blyth/opticks/optickscore/OpticksRun.cc:225
        #8  0x00007ffff7bd524e in OKMgr::propagate (this=0x7fffffffad70) at /home/blyth/opticks/ok/OKMgr.cc:123
        #9  0x0000000000402f0c in main (argc=1, argv=0x7fffffffaee8) at /home/blyth/opticks/ok/tests/OKTest.cc:32
        (gdb) 
        (gdb) f 8
        #8  0x00007ffff7bd524e in OKMgr::propagate (this=0x7fffffffad70) at /home/blyth/opticks/ok/OKMgr.cc:123
        123             m_run->setGensteps(m_gen->getInputGensteps()); 
        (gdb) f 7
        #7  0x00007ffff2e7a290 in OpticksRun::setGensteps (this=0x678a60, gensteps=0x57684e0) at /home/blyth/opticks/optickscore/OpticksRun.cc:225
        225     importGensteps();
        (gdb) f 6
        #6  0x00007ffff2e7a396 in OpticksRun::importGensteps (this=0x678a60) at /home/blyth/opticks/optickscore/OpticksRun.cc:253
        253     m_g4step = importGenstepData(m_gensteps, oac_label) ;
        (gdb) p m_gensteps
        $1 = (NPY<float> *) 0x57684e0
        (gdb) p m_gensteps->getShapeString()
        Too few arguments in function call.
        (gdb) p m_gensteps->getShapeString(0)
        $2 = "1,6,4"
        (gdb) 

        (gdb) f 5
        #5  0x00007ffff2e7b1bf in OpticksRun::importGenstepData (this=0x678a60, gs=0x57684e0, oac_label=0x0) at /home/blyth/opticks/optickscore/OpticksRun.cc:423
        423     g4step->checkGencodes();
        (gdb) f 4
        #4  0x00007ffff29645a4 in G4StepNPY::checkGencodes (this=0x225ad8c0) at /home/blyth/opticks/npy/G4StepNPY.cpp:288
        288     assert(mismatch == 0 );
        (gdb) l
        283          LOG(fatal)<<"G4StepNPY::checklabel FAIL" 
        284                    << " numStep " << numStep
        285                    << " mismatch " << mismatch ; 
        286                    ;
        287     }
        288     assert(mismatch == 0 );
        289 }
        290 

        Probably old gensteps not adhering to the new enum codes   


        blyth@localhost optickscore]$ OpticksGenstepTest 
        2020-09-26 23:39:37.477 INFO  [196742] [main@32] OpticksGenstep::Dump()
        2020-09-26 23:39:37.478 INFO  [196742] [main@33] 
                 0 : INVALID
                 1 : G4Cerenkov_1042
                 2 : G4Scintillation_1042
                 3 : DsG4Cerenkov_r3971
                 4 : DsG4Scintillation_r3971
                 5 : torch
                 6 : fabricated
                 7 : emitsource
                 8 : natural
                 9 : machinery
                10 : g4gun
                11 : primarysource
                12 : genstepsource






      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: SegFault         1.09   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     1.04   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     1.06   
      7  /34  Test #7  : CFG4Test.CG4Test                              ***Exception: SegFault         1.13   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   ***Exception: SegFault         1.16   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    ***Exception: SegFault         1.09   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: SegFault         1.20   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      1.15   









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

