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



Dec 2020 : tripping this again with OpticksRunTest using a torchstep from OpticksGenstep::Candle
--------------------------------------------------------------------------------------------------

::

    2020-12-13 16:14:58.562 INFO  [7283277] [Opticks::loadOriginCacheMeta@2018] (pass) GEOCACHE_CODE_VERSION 9
    2020-12-13 16:14:58.562 INFO  [7283277] [test_OpticksRun_reset@20] 0
    2020-12-13 16:14:58.564 ERROR [7283277] [G4StepNPY::checkGencodes@294]  i 0 unexpected gencode label 5 allowed gencodes 1,2,3,4,7,
    2020-12-13 16:14:58.564 FATAL [7283277] [G4StepNPY::checkGencodes@306] G4StepNPY::checklabel FAIL numStep 1 mismatch 1
    Assertion failed: (mismatch == 0), function checkGencodes, file /Users/blyth/opticks/npy/G4StepNPY.cpp, line 311.


What should be allowed ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gencodes that oxrap/cu/generate.cu understands, namely::

    epsilon:optixrap blyth$ grep gencode\ == cu/generate.cu 
        if(gencode == OpticksGenstep_G4Cerenkov_1042 ) 
        else if(gencode == OpticksGenstep_DsG4Scintillation_r3971 )
        else if(gencode == OpticksGenstep_G4Scintillation_1042 )
        else if(gencode == OpticksGenstep_TORCH)
        else if(gencode == OpticksGenstep_EMITSOURCE)
    epsilon:optixrap blyth$     

::

    474     if(gencode == OpticksGenstep_G4Cerenkov_1042 )
    475     {
    476         CerenkovStep cs ;
    477         csload(cs, genstep_buffer, genstep_offset, genstep_id);
    481         generate_cerenkov_photon(p, cs, rng );
    482         s.flag = CERENKOV ;
    483     }
    484     else if(gencode == OpticksGenstep_DsG4Scintillation_r3971 )
    485     {
    486         ScintillationStep ss ;
    487         ssload(ss, genstep_buffer, genstep_offset, genstep_id);
    491         generate_scintillation_photon(p, ss, rng );  // maybe split on gencode ?
    492         s.flag = SCINTILLATION ;
    493     }
    494     else if(gencode == OpticksGenstep_G4Scintillation_1042 )
    495     {
    496         Genstep_G4Scintillation_1042 ss ;
    497         ss.load( genstep_buffer, genstep_offset, genstep_id);
    501         ss.generate_photon(p, rng );
    502         s.flag = SCINTILLATION ;
    503     }
    504     else if(gencode == OpticksGenstep_TORCH)
    505     {
    506         TorchStep ts ;
    507         tsload(ts, genstep_buffer, genstep_offset, genstep_id);
    511         generate_torch_photon(p, ts, rng );
    512         s.flag = TORCH ;
    513     }
    514     else if(gencode == OpticksGenstep_EMITSOURCE)
    515     {
    516         // source_buffer is input only, photon_buffer output only, 
    517         // photon_offset is same for both these buffers
    518         pload(p, source_buffer, photon_offset );
    519         s.flag = TORCH ;
    523     }


okc/OpticksGenstep.h::

     19 enum
     20 {
     21     OpticksGenstep_INVALID                  = 0,   // Allowed?
     22     OpticksGenstep_G4Cerenkov_1042          = 1,   // Y
     23     OpticksGenstep_G4Scintillation_1042     = 2,   // Y
     24     OpticksGenstep_DsG4Cerenkov_r3971       = 3,   // Y  <<<< OOPS : LISTED AS ALLOWED BUT NOT COVERED : PROBABLY ASSUMING SAME AS G4Cerenkov_1042 ?
     25     OpticksGenstep_DsG4Scintillation_r3971  = 4,   // Y
     26     OpticksGenstep_TORCH                    = 5,   // N  <<<<  OOPS : COVERED BUT NOT LISTED AS ALLOWED
     27     OpticksGenstep_FABRICATED               = 6,   // N 
     28     OpticksGenstep_EMITSOURCE               = 7,   // Y
     29     OpticksGenstep_NATURAL                  = 8,   // N




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


    [blyth@localhost ~]$ oe;OpticksRun=INFO OpticksGen=INFO G4StepNPY=INFO  gdb $(which OKTest) 

    2020-11-26 20:41:52.546 INFO  [27879] [OpticksRun::annotateEvent@158]  testcsgpath - geotestconfig -
    2020-11-26 20:41:52.546 INFO  [27879] [OpticksRun::setGensteps@221] gensteps 1,6,4
    2020-11-26 20:41:52.547 INFO  [27879] [G4StepNPY::G4StepNPY@45]  npy 1,6,4
    2020-11-26 20:41:52.547 INFO  [27879] [OpticksRun::importGenstepData@396] Run evt Evt /tmp/blyth/opticks/OKTest/evt/g4live/torch/1 20201126_204152 /home/blyth/local/opticks/lib/OKTest g4evt Evt /tmp/blyth/opticks/OKTest/evt/g4live/torch/-1 20201126_204152 /home/blyth/local/opticks/lib/OKTest shape 1,6,4 oac : 
    2020-11-26 20:41:52.547 INFO  [27879] [OpticksRun::importGenstepData@429]  checklabel of non-legacy (collected direct) gensteps  oac : 
    2020-11-26 20:41:52.547 INFO  [27879] [G4StepNPY::checkGencodes@281]   numStep 1 allowedGencodes: 1,2,3,4,7,
    2020-11-26 20:41:52.547 ERROR [27879] [G4StepNPY::checkGencodes@294]  i 0 unexpected gencode label 5 allowed gencodes 1,2,3,4,7,
    2020-11-26 20:41:52.547 FATAL [27879] [G4StepNPY::checkGencodes@306] G4StepNPY::checklabel FAIL numStep 1 mismatch 1
    OKTest: /home/blyth/opticks/npy/G4StepNPY.cpp:311: void G4StepNPY::checkGencodes(): Assertion `mismatch == 0' failed.




::

    [blyth@localhost opticks]$ oe;OpticksRun=INFO OpticksGen=INFO G4StepNPY=INFO  gdb $(which CGenstepSourceTest) 
    GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-114.el7
    ...
    020-11-26 23:21:30.065 INFO  [287760] [Opticks::loadOriginCacheMeta@1944] (pass) GEOCACHE_CODE_VERSION 8
    2020-11-26 23:21:30.065 INFO  [287760] [OpticksHub::loadGeometry@280] [ /home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1
    2020-11-26 23:21:34.727 INFO  [287760] [OpticksHub::loadGeometry@312] ]
    2020-11-26 23:21:34.728 INFO  [287760] [OpticksGen::init@129] 
    2020-11-26 23:21:34.728 INFO  [287760] [OpticksGen::initFromDirectGensteps@183] /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy
    2020-11-26 23:21:34.732 ERROR [287760] [main@63] --------------------------------
    2020-11-26 23:21:34.732 INFO  [287760] [CGenstepSource::generatePhotonsFromOneGenstep@141]  gencode 5 OpticksFlags::Flag(gencode) BAD_FLAG
    2020-11-26 23:21:34.732 FATAL [287760] [CGenstepSource::generatePhotonsFromOneGenstep@156]  failed to generate for  gencode 5 flag BAD_FLAG
    CGenstepSourceTest: /home/blyth/opticks/cfg4/CGenstepSource.cc:162: G4VParticleChange* CGenstepSource::generatePhotonsFromOneGenstep(): Assertion `pc' failed.

    Program received signal SIGABRT, Aborted.
    (gdb) bt
    #4  0x00007ffff7b4da02 in CGenstepSource::generatePhotonsFromOneGenstep (this=0x73306f0) at /home/blyth/opticks/cfg4/CGenstepSource.cc:162
    #5  0x00007ffff7b4d529 in CGenstepSource::GeneratePrimaryVertex (this=0x73306f0, event=0x73319c0) at /home/blyth/opticks/cfg4/CGenstepSource.cc:98
    #6  0x00000000004046ed in main (argc=1, argv=0x7fffffffabf8) at /home/blyth/opticks/cfg4/tests/CGenstepSourceTest.cc:82
    (gdb) 


::

    epsilon:issues blyth$ oe;OpticksRun=INFO OpticksGen=INFO G4StepNPY=INFO  lldb_ $(which CGenstepSourceTest) 
    2020-11-26 15:23:59.330 INFO  [309736] [Opticks::loadOriginCacheMeta@1916] ExtractCacheMetaGDMLPath /usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml
    2020-11-26 15:23:59.330 INFO  [309736] [Opticks::loadOriginCacheMeta@1944] (pass) GEOCACHE_CODE_VERSION 8
    2020-11-26 15:23:59.330 INFO  [309736] [OpticksHub::loadGeometry@280] [ /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1
    2020-11-26 15:24:03.158 INFO  [309736] [OpticksHub::loadGeometry@312] ]
    2020-11-26 15:24:03.158 INFO  [309736] [OpticksGen::init@129] 
    2020-11-26 15:24:03.158 INFO  [309736] [OpticksGen::initFromLegacyGensteps@189] 
    2020-11-26 15:24:03.158 INFO  [309736] [OpticksGen::initFromLegacyGensteps@199]  code 5 type torch
    2020-11-26 15:24:03.158 INFO  [309736] [*OpticksGen::makeLegacyGensteps@227]  code 5 srctype torch
    2020-11-26 15:24:03.158 FATAL [309736] [*Opticks::makeSimpleTorchStep@3572]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg 
    2020-11-26 15:24:03.159 ERROR [309736] [*OpticksGen::makeTorchstep@429]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0 genstepTarget --gensteptarget : 0
    2020-11-26 15:24:03.159 INFO  [309736] [OpticksGen::targetGenstep@355] setting frame 0 Id
    2020-11-26 15:24:03.159 ERROR [309736] [*OpticksGen::makeTorchstep@455]  generateoverride 0 num_photons0 10000 num_photons 10000
    2020-11-26 15:24:03.168 ERROR [309736] [main@63] --------------------------------
    Process 20280 exited with status = 0 (0x00000000) 
    (lldb) 


Getting different behaviour due to existance of direct gensteps in Linux case::

    [blyth@localhost ~]$ l /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy
    -rw-rw-r--. 1 blyth blyth 176 Nov 25 00:51 /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy
    [blyth@localhost ~]$ date
    Thu Nov 26 23:27:43 CST 2020

    [blyth@localhost ~]$ python3 ~/opticks/bin/np.py -v -i  /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy
    a :                 /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy :            (1, 6, 4) : b1c03673018cd1e81a7f5080cdaf31e8 : 20201125-0051 
    (1, 6, 4)
    f32
    [[[[      0.          0.          0.          0.   ]
       [ -18079.453 -799699.44    -6605.          0.1  ]
       [      0.          0.          1.          1.   ]
       [      0.          0.          1.        430.   ]
       [      0.          1.          0.          1.   ]
       [      0.          0.          0.          0.   ]]]]
    (1, 6, 4)
    i32
    [[[[         5          0          0       5000]
       [-963821848 -918340297 -976328704 1036831949]
       [         0          0 1065353216 1065353216]
       [         0          0 1065353216 1138163712]
       [         0 1065353216          0 1065353216]
       [         0          0          0          1]]]]
    [blyth@localhost ~]$ 



::

     59 OpticksGen::OpticksGen(OpticksHub* hub)
     60     :
     61     m_hub(hub),
     62     m_gun(new OpticksGun(hub)),
     63     m_ok(hub->getOpticks()),
     64     m_cfg(m_ok->getCfg()),
     65     m_ggeo(hub->getGGeo()),
     66     m_ggb(hub->getGGeoBase()),
     67     m_blib(m_ggb->getBndLib()),
     68     m_lookup(hub->getLookup()),
     69     m_torchstep(NULL),
     70     m_fabstep(NULL),
     71     m_csg_emit(hub->findEmitter()),
     72     m_dbgemit(m_ok->isDbgEmit()),     // --dbgemit
     73     m_emitter(m_csg_emit ? new NEmitPhotonsNPY(m_csg_emit, OpticksGenstep_EMITSOURCE, m_ok->getSeed(), m_dbgemit, m_ok->getMaskBuffer(), m_ok->getGenerateOverride() ) : NULL ),
     74     m_input_photons(NULL),
     75     m_tagoffset(0),
     76     m_direct_gensteps(m_ok->findGensteps(m_tagoffset)),
     77     m_legacy_gensteps(NULL),
     78     m_source_code(initSourceCode())
     79 {
     80     init() ;
     81 }

::

    3425 NPY<float>* Opticks::findGensteps( unsigned tagoffset ) const
    3426 {
    3427     LOG(LEVEL) << "[ tagoffset " ;
    3428 
    3429     NPY<float>* gs = NULL ;
    3430     if( hasKey() && !isTest() )
    3431     {
    3432         if( isDbgGSLoad() && existsDebugGenstepPath(tagoffset) )
    3433         {
    3434             gs = loadDebugGenstep(tagoffset) ;
    3435         }
    3436         else if( existsDirectGenstepPath(tagoffset) )
    3437         {
    3438             gs = loadDirectGenstep(tagoffset) ;
    3439         }  
    3440     }  
    3441     LOG(LEVEL) << "] gs " << gs ;
    3442     return gs ;
    3443 }

::

    116 /**
    117 OpticksGen::init
    118 ------------------
    119 
    120 Upshot is that one of the below gets set
    121 
    122 * m_direct_gensteps 
    123 * m_legacy_gensteps : for emitter as well as legacy gensteps
    124 
    125 **/
    126 
    127 void OpticksGen::init()
    128 {
    129     LOG(LEVEL);
    130     if(m_direct_gensteps)
    131     {
    132         initFromDirectGensteps();
    133     }
    134     else if(m_emitter)
    135     {
    136         initFromEmitterGensteps();
    137     }
    138     else
    139     {
    140         initFromLegacyGensteps();
    141     }
    142 }




::

    oe;OpticksRun=INFO OpticksGen=INFO G4StepNPY=INFO Opticks=INFO gdb $(which CGenstepSourceTest) 

    2020-11-26 23:34:16.353 INFO  [306938] [OpticksHub::loadGeometry@312] ]
    2020-11-26 23:34:16.353 INFO  [306938] [Opticks::findGensteps@3427] [ tagoffset 
    2020-11-26 23:34:16.354 INFO  [306938] [Opticks::existsDirectGenstepPath@3393]  path /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy exists 1
    2020-11-26 23:34:16.354 INFO  [306938] [Opticks::findGensteps@3441] ] gs 0x728a160
    2020-11-26 23:34:16.354 INFO  [306938] [OpticksGen::init@129] 
    2020-11-26 23:34:16.354 INFO  [306938] [OpticksGen::initFromDirectGensteps@183] /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy
    2020-11-26 23:34:16.358 ERROR [306938] [main@63] --------------------------------
    2020-11-26 23:34:16.358 INFO  [306938] [Opticks::findGensteps@3427] [ tagoffset 
    2020-11-26 23:34:16.358 INFO  [306938] [Opticks::existsDirectGenstepPath@3393]  path /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy exists 1
    2020-11-26 23:34:16.358 INFO  [306938] [Opticks::findGensteps@3441] ] gs 0x7330ed0
    2020-11-26 23:34:16.359 INFO  [306938] [CGenstepSource::generatePhotonsFromOneGenstep@141]  gencode 5 OpticksFlags::Flag(gencode) BAD_FLAG
    2020-11-26 23:34:16.359 FATAL [306938] [CGenstepSource::generatePhotonsFromOneGenstep@156]  failed to generate for  gencode 5 flag BAD_FLAG
    CGenstepSourceTest: /home/blyth/opticks/cfg4/CGenstepSource.cc:162: G4VParticleChange* CGenstepSource::generatePhotonsFromOneGenstep(): Assertion `pc' failed.

    (gdb) bt
    #4  0x00007ffff7b4da02 in CGenstepSource::generatePhotonsFromOneGenstep (this=0x7330a50) at /home/blyth/opticks/cfg4/CGenstepSource.cc:162
    #5  0x00007ffff7b4d529 in CGenstepSource::GeneratePrimaryVertex (this=0x7330a50, event=0x7331ea0) at /home/blyth/opticks/cfg4/CGenstepSource.cc:98
    #6  0x00000000004046ed in main (argc=1, argv=0x7fffffffab08) at /home/blyth/opticks/cfg4/tests/CGenstepSourceTest.cc:82
    (gdb) 

    oe;OpticksRun=INFO OpticksGen=INFO G4StepNPY=INFO Opticks=INFO lldb_ $(which CGenstepSourceTest) 

    epsilon:optickscore blyth$ oe;OpticksRun=INFO OpticksGen=INFO G4StepNPY=INFO Opticks=INFO lldb_ $(which CGenstepSourceTest) 
    (lldb) target create "/usr/local/opticks/lib/CGenstepSourceTest"

    2020-11-26 15:37:20.331 INFO  [318152] [OpticksHub::loadGeometry@312] ]
    2020-11-26 15:37:20.331 INFO  [318152] [*Opticks::findGensteps@3427] [ tagoffset 
    2020-11-26 15:37:20.331 INFO  [318152] [Opticks::existsDirectGenstepPath@3393]  path /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy exists 0
    2020-11-26 15:37:20.331 INFO  [318152] [*Opticks::findGensteps@3441] ] gs 0x0
    2020-11-26 15:37:20.331 INFO  [318152] [OpticksGen::init@129] 
    2020-11-26 15:37:20.331 INFO  [318152] [OpticksGen::initFromLegacyGensteps@189] 
    2020-11-26 15:37:20.331 INFO  [318152] [OpticksGen::initFromLegacyGensteps@199]  code 5 type torch
    2020-11-26 15:37:20.331 INFO  [318152] [*OpticksGen::makeLegacyGensteps@227]  code 5 srctype torch
    2020-11-26 15:37:20.331 FATAL [318152] [*Opticks::makeSimpleTorchStep@3572]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg 
    2020-11-26 15:37:20.331 ERROR [318152] [*OpticksGen::makeTorchstep@429]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0 genstepTarget --gensteptarget : 0
    2020-11-26 15:37:20.332 INFO  [318152] [OpticksGen::targetGenstep@355] setting frame 0 Id
    2020-11-26 15:37:20.332 ERROR [318152] [*OpticksGen::makeTorchstep@455]  generateoverride 0 num_photons0 10000 num_photons 10000
    2020-11-26 15:37:20.340 ERROR [318152] [main@63] --------------------------------
    2020-11-26 15:37:20.340 INFO  [318152] [*Opticks::findGensteps@3427] [ tagoffset 
    2020-11-26 15:37:20.341 INFO  [318152] [Opticks::existsDirectGenstepPath@3393]  path /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy exists 0
    2020-11-26 15:37:20.341 INFO  [318152] [*Opticks::findGensteps@3441] ] gs 0x0
    Process 20966 exited with status = 0 (0x00000000) 


Unhealthy for the existance of that file to change behavior::

    [blyth@localhost ~]$ find /tmp/blyth/opticks/evt/g4live/
    /tmp/blyth/opticks/evt/g4live/
    /tmp/blyth/opticks/evt/g4live/natural
    /tmp/blyth/opticks/evt/g4live/natural/1
    /tmp/blyth/opticks/evt/g4live/natural/1/gs.json
    /tmp/blyth/opticks/evt/g4live/natural/1/gs.npy
    /tmp/blyth/opticks/evt/g4live/torch
    /tmp/blyth/opticks/evt/g4live/torch/1
    /tmp/blyth/opticks/evt/g4live/torch/1/gs.json
    /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy
    [blyth@localhost ~]$ 
    [blyth@localhost ~]$ python3 ~/opticks/bin/js.py /tmp/blyth/opticks/evt/g4live/torch/1/gs.json
    {'ArrayContentIndex': 0, 'ArrayContentVersion': 1042}
    [blyth@localhost ~]$ python3 ~/opticks/bin/js.py /tmp/blyth/opticks/evt/g4live/natural/1/gs.json
    {'ArrayContentIndex': 0, 'ArrayContentVersion': 1042}
    [blyth@localhost ~]$ 

    epsilon:optickscore blyth$ find /tmp/blyth/opticks/evt/g4live/
    find: /tmp/blyth/opticks/evt/g4live/: No such file or directory
    epsilon:optickscore blyth$ 


After moving the direct gensteps aside CGenstepSourceTest no longer fails on Linux::

    [blyth@localhost ~]$ mv /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy.0

    [blyth@localhost ~]$ oe;OpticksRun=INFO OpticksGen=INFO G4StepNPY=INFO Opticks=INFO gdb $(which CGenstepSourceTest)


    2020-11-26 23:44:47.642 FATAL [322258] [Opticks::makeSimpleTorchStep@3572]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg 
    2020-11-26 23:44:47.643 ERROR [322258] [OpticksGen::makeTorchstep@429]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0 genstepTarget --gensteptarget : 0
    2020-11-26 23:44:47.643 INFO  [322258] [OpticksGen::targetGenstep@355] setting frame 0 Id
    2020-11-26 23:44:47.643 ERROR [322258] [OpticksGen::makeTorchstep@455]  generateoverride 0 num_photons0 10000 num_photons 10000
    2020-11-26 23:44:47.647 ERROR [322258] [main@63] --------------------------------
    2020-11-26 23:44:47.647 INFO  [322258] [Opticks::findGensteps@3427] [ tagoffset 
    2020-11-26 23:44:47.647 INFO  [322258] [Opticks::existsDirectGenstepPath@3393]  path /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy exists 0
    2020-11-26 23:44:47.647 INFO  [322258] [Opticks::findGensteps@3441] ] gs 0
    [Inferior 1 (process 322258) exited normally]

    [blyth@localhost ~]$ oe;OpticksRun=INFO OpticksGen=INFO G4StepNPY=INFO Opticks=INFO gdb $(which OKTest)


OKTest also completes with the direct gensteps moved aside::

    [blyth@localhost opticks]$ oe;OpticksRun=INFO OpticksGen=INFO G4StepNPY=INFO Opticks=INFO gdb $(which OKTest)

    2020-11-26 23:47:55.887 INFO  [326943] [OpticksHub::loadGeometry@312] ]
    2020-11-26 23:47:55.887 INFO  [326943] [Opticks::findGensteps@3427] [ tagoffset 
    2020-11-26 23:47:55.887 INFO  [326943] [Opticks::existsDirectGenstepPath@3393]  path /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy exists 0
    2020-11-26 23:47:55.887 INFO  [326943] [Opticks::findGensteps@3441] ] gs 0
    2020-11-26 23:47:55.887 INFO  [326943] [OpticksGen::init@129] 
    2020-11-26 23:47:55.887 INFO  [326943] [OpticksGen::initFromLegacyGensteps@189] 
    2020-11-26 23:47:55.887 INFO  [326943] [OpticksGen::initFromLegacyGensteps@199]  code 5 type torch
    2020-11-26 23:47:55.887 INFO  [326943] [OpticksGen::makeLegacyGensteps@227]  code 5 srctype torch


All tests pass on Linux without that file::

    [blyth@localhost ~]$ l /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy
    ls: cannot access /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy: No such file or directory

opticks-t::

    SLOW: tests taking longer that 15 seconds
    FAILS:  0   / 452   :  Thu Nov 26 23:54:50 2020   
    [blyth@localhost opticks]$ 


Also on Darwin all tests other than G4OKTest pass without that file::

    epsilon:opticks blyth$ l /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy
    ls: /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy: No such file or directory

    opticks-t
    ...
    SLOW: tests taking longer that 15 seconds
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         20.83  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 Passed                         15.88  


    FAILS:  1   / 452   :  Thu Nov 26 15:58:59 2020   
      1  /2   Test #1  : G4OKTest.G4OKTest                             Child aborted***Exception:     7.35   
    epsilon:opticks blyth$ l /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy
    ls: /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy: No such file or directory



Need to find what is writing that file. How to simplify the genstep handling to avoid the sensitivity to it ?
Looks like G4Opticks::propagateOpticalPhotons when not using "--production" option saves the 
direct gensteps::

     866 int G4Opticks::propagateOpticalPhotons(G4int eventID)
     867 {
     868     LOG(LEVEL) << "[[" ;
     869     assert( m_genstep_collector );
     870     m_gensteps = m_genstep_collector->getGensteps();
     871     m_gensteps->setArrayContentVersion(G4VERSION_NUMBER);
     872     m_gensteps->setArrayContentIndex(eventID);
     873 
     874     unsigned num_gensteps = m_gensteps->getNumItems();
     875     LOG(LEVEL) << " num_gensteps "  << num_gensteps ;
     876     if( num_gensteps == 0 )
     877     {   
     878         LOG(fatal) << "SKIP as no gensteps have been collected " ;
     879         return 0 ;
     880     }
     881 
     882 
     883     unsigned tagoffset = eventID ;  // tags are 1-based : so this will normally be the Geant4 eventID + 1
     884 
     885     if(!m_ok->isProduction()) // --production
     886     {   
     887         const char* gspath = m_ok->getDirectGenstepPath(tagoffset);
     888         LOG(LEVEL) << "[ saving gensteps to " << gspath ;
     889         m_gensteps->save(gspath);  
     890         LOG(LEVEL) << "] saving gensteps to " << gspath ;
     891     }





