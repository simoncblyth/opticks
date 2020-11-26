G4StepNPY_checkGencodes_mismatch_assert
=========================================

Issue
----------------------------------------

* :doc:`OSensorLib_canonical.rst`

After making the 1-based sensorIndex change an opticks-t run
shows some gencode related fails.

Huh this problem looks unrelated to sensorIndex change.  Must be coming from an earlier change.

The issue looks the same as one from Sept

* :doc:`G4StepNPY_gencode_assert.rst`



macOS fails
--------------

opticks-t::

    CTestLog :             analytic :      0/     1 : 2020-11-26 09:28:17.596622 : /usr/local/opticks/build/analytic/ctest.log 
    CTestLog :                  bin :      0/     1 : 2020-11-26 09:28:17.846581 : /usr/local/opticks/build/bin/ctest.log 

    SLOW: tests taking longer that 15 seconds
      2  /2   Test #2  : IntegrationTests.tboolean.box                 Passed                         15.91  

    FAILS:  6   / 452   :  Thu Nov 26 09:28:17 2020   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     8.04   
      7  /34  Test #7  : CFG4Test.CG4Test                              Child aborted***Exception:     4.75   
      32 /34  Test #32 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     4.25   
      33 /34  Test #33 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     4.18   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     9.34   
      1  /2   Test #1  : G4OKTest.G4OKTest                             Child aborted***Exception:     7.62   
    epsilon:opticks blyth$ 

::

    lldb_ OKTest 

    2020-11-26 09:33:52.619 ERROR [15602251] [G4StepNPY::checkGencodes@281]  i 0 unexpected gencode label 5 allowed gencodes 1,2,3,4,7,
    2020-11-26 09:33:52.619 FATAL [15602251] [G4StepNPY::checkGencodes@293] G4StepNPY::checklabel FAIL numStep 1 mismatch 1
    Assertion failed: (mismatch == 0), function checkGencodes, file /Users/blyth/opticks/npy/G4StepNPY.cpp, line 298.
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
    (lldb) bt
        frame #3: 0x00007fff57ede1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010411ce27 libNPY.dylib`G4StepNPY::checkGencodes(this=0x00000001244ef2f0) at G4StepNPY.cpp:298
        frame #5: 0x0000000103c4aea1 libOpticksCore.dylib`OpticksRun::importGenstepData(this=0x0000000108718810, gs=0x00000001129528b0, oac_label=0x0000000000000000) at OpticksRun.cc:434
        frame #6: 0x0000000103c49fd9 libOpticksCore.dylib`OpticksRun::importGensteps(this=0x0000000108718810) at OpticksRun.cc:254
        frame #7: 0x0000000103c4994d libOpticksCore.dylib`OpticksRun::setGensteps(this=0x0000000108718810, gensteps=0x00000001129528b0) at OpticksRun.cc:225
        frame #8: 0x00000001000d5ae8 libOK.dylib`OKMgr::propagate(this=0x00007ffeefbfe8e8) at OKMgr.cc:123
        frame #9: 0x000000010000b997 OKTest`main(argc=1, argv=0x00007ffeefbfe9a8) at OKTest.cc:32
        frame #10: 0x00007fff57e6a015 libdyld.dylib`start + 1
        frame #11: 0x00007fff57e6a015 libdyld.dylib`start + 1
    (lldb) 


Linux fails
-------------

Check Linux before doing the sensorIndex change, 5 of 6 same fails::

    FAILS:  5   / 452   :  Thu Nov 26 19:53:17 2020   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     9.70   
      7  /34  Test #7  : CFG4Test.CG4Test                              Child aborted***Exception:     5.97   
      32 /34  Test #32 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     5.42   
      33 /34  Test #33 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     5.32   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     10.26  
    [blyth@localhost opticks]$ 



Darwin tries
-------------

Huh running CG4Test doesnt fail.::


    lldb_ CG4Test 

    2020-11-26 12:01:32.170 WARN  [79642] [main@51]  post CG4 
    2020-11-26 12:01:32.170 WARN  [79642] [main@55]   post CG4::interactive
    2020-11-26 12:01:32.174 ERROR [79642] [main@63]  setting gensteps 0x1153bc1f0
    2020-11-26 12:01:32.179 INFO  [79642] [*CG4::propagate@395]  calling BeamOn numG4Evt 1
    2020-11-26 12:01:41.035 INFO  [79642] [CTorchSource::GeneratePrimaryVertex@290]  event_gencode 5 : BAD_FLAG
    2020-11-26 12:01:41.763 INFO  [79642] [*CG4::propagate@401]  calling BeamOn numG4Evt 1 DONE 
    2020-11-26 12:01:41.763 INFO  [79642] [CG4::postpropagate@422] [ (0) ctx CG4Ctx::desc_stats dump_count 0 event_total 1 event_track_count 10000


Also OKTest doesnt fail when run interactively.::

    lldb_ OKTest 


Hmm.  Note that the opticks-t run was the first after a reboot (for Darwin)
Need to check the loading of gensteps. 


Grab commandlines from earlier incident

* :doc:`G4StepNPY_gencode_assert.rst`

::

    OpticksRun=INFO OpticksGen=INFO lldb_ OKTest -- --dbggsimport

    2020-11-26 12:15:16.773 INFO  [86972] [OpticksRun::annotateEvent@158]  testcsgpath - geotestconfig -
    2020-11-26 12:15:16.773 INFO  [86972] [OpticksRun::setGensteps@221] gensteps 1,6,4
    2020-11-26 12:15:16.773 FATAL [86972] [*OpticksRun::importGenstepData@373] (--dbggsimport) saving gs to $TMP/OpticksRun_importGenstepData/dbggsimport.npy
    2020-11-26 12:15:16.774 INFO  [86972] [*OpticksRun::importGenstepData@396] Run evt Evt /tmp/blyth/opticks/OKTest/evt/g4live/torch/1 20201126_121516 /usr/local/opticks/lib/OKTest g4evt Evt /tmp/blyth/opticks/OKTest/evt/g4live/torch/-1 20201126_121516 /usr/local/opticks/lib/OKTest shape 1,6,4 oac : GS_TORCH 
    2020-11-26 12:15:16.774 INFO  [86972] [*OpticksRun::importGenstepData@417]  checklabel of torch steps  oac : GS_TORCH 
    2020-11-26 12:15:16.774 INFO  [86972] [*OpticksRun::importGenstepData@438]  Keys  OpticksGenstep_TORCH: 5 OpticksGenstep_G4Cerenkov_1042: 1 OpticksGenstep_G4Scintillation_1042: 2 OpticksGenstep_DsG4Cerenkov_r3971: 3 OpticksGenstep_DsG4Scintillation_r3971: 4 OpticksGenstep_G4GUN: 10
    2020-11-26 12:15:16.774 INFO  [86972] [*OpticksRun::importGenstepData@448]  counts  [          5     10000 ]  [      total     10000 ] 
    2020-11-26 12:15:16.774 NONE  [86972] [OpticksViz::uploadEvent@406] [ (0)
    2020-11-26 12:15:16.797 NONE  [86972] [OpticksViz::uploadEvent@413] ] (0)



Add debug to G4StepNPY::

    OpticksRun=INFO OpticksGen=INFO G4StepNPY=INFO  lldb_ OKTest -- --dbggsimport

    2020-11-26 12:33:05.140 INFO  [99983] [G4StepNPY::G4StepNPY@45]  npy 1,6,4
    2020-11-26 12:33:05.140 INFO  [99983] [*OpticksRun::importGenstepData@396] Run evt Evt /tmp/blyth/opticks/OKTest/evt/g4live/torch/1 20201126_123305 /usr/local/opticks/lib/OKTest g4evt Evt /tmp/blyth/opticks/OKTest/evt/g4live/torch/-1 20201126_123305 /usr/local/opticks/lib/OKTest shape 1,6,4 oac : GS_TORCH 
    2020-11-26 12:33:05.140 INFO  [99983] [*OpticksRun::importGenstepData@417]  checklabel of torch steps  oac : GS_TORCH 
    2020-11-26 12:33:05.140 INFO  [99983] [G4StepNPY::checkGencodes@281]   numStep 1 allowedGencodes: 5,
    2020-11-26 12:33:05.140 INFO  [99983] [*OpticksRun::importGenstepData@438]  Keys  OpticksGenstep_TORCH: 5 OpticksGenstep_G4Cerenkov_1042: 1 OpticksGenstep_G4Scintillation_1042: 2 OpticksGenstep_DsG4Cerenkov_r3971: 3 OpticksGenstep_DsG4Scintillation_r3971: 4 OpticksGenstep_G4GUN: 10
    2020-11-26 12:33:05.140 INFO  [99983] [*OpticksRun::importGenstepData@448]  counts  [          5     10000 ]  [      total     10000 ] 
    2020-11-26 12:33:05.140 NONE  [99983] [OpticksViz::uploadEvent@406] [ (0)
    2020-11-26 12:33:05.161 NONE  [99983] [OpticksViz::uploadEvent@413] ] (0)



::

    epsilon:npy blyth$ np.py -v -i $TMP/OpticksRun_importGenstepData/dbggsimport.npy 
    a : /tmp/blyth/opticks/OpticksRun_importGenstepData/dbggsimport.npy :            (1, 6, 4) : aac496cac9ae32326ac9a0168f523b22 : 20201126-1215 
    (1, 6, 4)
    f32
    [[[[  0.    0.    0.    0. ]
       [  0.    0.    0.    0.1]
       [  0.    0.    1.    1. ]
       [  0.    0.    1.  430. ]
       [  0.    1.    0.    1. ]
       [  0.    0.    0.    0. ]]]]
    (1, 6, 4)
    i32
    [[[[         5          0         95      10000]
       [         0          0          0 1036831949]
       [         0          0 1065353216 1065353216]
       [         0          0 1065353216 1138163712]
       [         0 1065353216          0 1065353216]
       [         0          0          0          1]]]]
    epsilon:npy blyth$ 


::


    [blyth@localhost ~]$ oe
    [blyth@localhost ~]$ OpticksRun=INFO OpticksGen=INFO G4StepNPY=INFO  gdb $(which OKTest) 

    2020-11-26 20:41:52.546 INFO  [27879] [OpticksRun::annotateEvent@158]  testcsgpath - geotestconfig -
    2020-11-26 20:41:52.546 INFO  [27879] [OpticksRun::setGensteps@221] gensteps 1,6,4
    2020-11-26 20:41:52.547 INFO  [27879] [G4StepNPY::G4StepNPY@45]  npy 1,6,4
    2020-11-26 20:41:52.547 INFO  [27879] [OpticksRun::importGenstepData@396] Run evt Evt /tmp/blyth/opticks/OKTest/evt/g4live/torch/1 20201126_204152 /home/blyth/local/opticks/lib/OKTest g4evt Evt /tmp/blyth/opticks/OKTest/evt/g4live/torch/-1 20201126_204152 /home/blyth/local/opticks/lib/OKTest shape 1,6,4 oac : 
    2020-11-26 20:41:52.547 INFO  [27879] [OpticksRun::importGenstepData@429]  checklabel of non-legacy (collected direct) gensteps  oac : 
    2020-11-26 20:41:52.547 INFO  [27879] [G4StepNPY::checkGencodes@281]   numStep 1 allowedGencodes: 1,2,3,4,7,
    2020-11-26 20:41:52.547 ERROR [27879] [G4StepNPY::checkGencodes@294]  i 0 unexpected gencode label 5 allowed gencodes 1,2,3,4,7,
    2020-11-26 20:41:52.547 FATAL [27879] [G4StepNPY::checkGencodes@306] G4StepNPY::checklabel FAIL numStep 1 mismatch 1
    OKTest: /home/blyth/opticks/npy/G4StepNPY.cpp:311: void G4StepNPY::checkGencodes(): Assertion `mismatch == 0' failed.



