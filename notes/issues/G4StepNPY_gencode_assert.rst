G4StepNPY_gencode_assert
==========================

FIXED Opticks::makeSimpleTorchStep was using outdated TORCH enum
-------------------------------------------------------------------

::

    -    TorchStepNPY* torchstep = new TorchStepNPY(TORCH, 1, cfg );
    +    //TorchStepNPY* torchstep = new TorchStepNPY(TORCH, 1, cfg );   // see notes/issues/G4StepNPY_gencode_assert.rst
    +    TorchStepNPY* torchstep = new TorchStepNPY(OpticksGenstep_TORCH, 1, cfg );
     


Issue : OKTest G4StepNPY::checkGencodes assert
-----------------------------------------------

::

    OpticksRun=INFO OpticksGen=INFO lldb_ OKTest -- --dbggsimport

    ...


    2020-09-30 16:19:27.024 FATAL [3079961] [Opticks::setProfileDir@546]  dir /tmp/blyth/opticks/OKTest/evt/g4live/torch
    2020-09-30 16:19:27.027 INFO  [3079961] [OpticksHub::loadGeometry@585] ]
    2020-09-30 16:19:27.655 INFO  [3079961] [OpticksGen::init@127] 
    2020-09-30 16:19:27.655 INFO  [3079961] [OpticksGen::initFromLegacyGensteps@189] 
    2020-09-30 16:19:27.655 INFO  [3079961] [OpticksGen::initFromLegacyGensteps@199]  code 5 type torch
    2020-09-30 16:19:27.655 INFO  [3079961] [OpticksGen::makeLegacyGensteps@227]  code 5 srctype torch
    2020-09-30 16:19:27.655 INFO  [3079961] [Opticks::makeSimpleTorchStep@3335]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg 
    2020-09-30 16:19:27.655 ERROR [3079961] [OpticksGen::makeTorchstep@404]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0
    2020-09-30 16:19:27.655 INFO  [3079961] [OpticksGen::targetGenstep@336] setting frame 0 Id
    2020-09-30 16:19:27.655 ERROR [3079961] [OpticksGen::makeTorchstep@428]  generateoverride 0 num_photons0 10000 num_photons 10000


    2020-09-30 16:41:23.532 INFO  [23171] [OpticksRun::createEvent@122] (0) 20200930_164123[  ok:Evt /tmp/blyth/opticks/OKTest/evt/g4live/torch/1 20200930_164123 /usr/local/opticks/lib/OKTest g4:Evt /tmp/blyth/opticks/OKTest/evt/g4live/torch/-1 20200930_164123 /usr/local/opticks/lib/OKTest] DONE 
    2020-09-30 16:41:23.532 INFO  [23171] [OpticksRun::annotateEvent@158]  testcsgpath - geotestconfig -
    2020-09-30 16:41:23.532 INFO  [23171] [OpticksRun::setGensteps@221] gensteps 1,6,4
    2020-09-30 16:41:23.532 FATAL [23171] [*OpticksRun::importGenstepData@372] (--dbggsimport) saving gs to $TMP/OpticksRun_importGenstepData/dbggsimport.npy
    2020-09-30 16:41:23.533 INFO  [23171] [*OpticksRun::importGenstepData@395] Run evt Evt /tmp/blyth/opticks/OKTest/evt/g4live/torch/1 20200930_164123 /usr/local/opticks/lib/OKTest g4evt Evt /tmp/blyth/opticks/OKTest/evt/g4live/torch/-1 20200930_164123 /usr/local/opticks/lib/OKTest shape 1,6,4 oac : GS_TORCH 
    2020-09-30 16:41:23.533 INFO  [23171] [*OpticksRun::importGenstepData@415]  checklabel of torch steps  oac : GS_TORCH 
    2020-09-30 16:41:23.533 ERROR [23171] [G4StepNPY::checkGencodes@281]  i 0 unexpected gencode label 4096 allowed gencodes 5,
    2020-09-30 16:41:23.533 FATAL [23171] [G4StepNPY::checkGencodes@293] G4StepNPY::checklabel FAIL numStep 1 mismatch 1
    Assertion failed: (mismatch == 0), function checkGencodes, file /Users/blyth/opticks/npy/G4StepNPY.cpp, line 298.
    Process 5254 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff522b4b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff522b4b66 <+10>: jae    0x7fff522b4b70            ; <+20>
        0x7fff522b4b68 <+12>: movq   %rax, %rdi
        0x7fff522b4b6b <+15>: jmp    0x7fff522abae9            ; cerror_nocancel
        0x7fff522b4b70 <+20>: retq   
    Target 0: (OKTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff522b4b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff5247f080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff522101ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff521d81ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000104ff1f57 libNPY.dylib`G4StepNPY::checkGencodes(this=0x00000001448e0190) at G4StepNPY.cpp:298
        frame #5: 0x0000000104b395e4 libOpticksCore.dylib`OpticksRun::importGenstepData(this=0x0000000109621e80, gs=0x00000001178154b0, oac_label=0x0000000000000000) at OpticksRun.cc:432
        frame #6: 0x0000000104b38739 libOpticksCore.dylib`OpticksRun::importGensteps(this=0x0000000109621e80) at OpticksRun.cc:253
        frame #7: 0x0000000104b380ad libOpticksCore.dylib`OpticksRun::setGensteps(this=0x0000000109621e80, gensteps=0x00000001178154b0) at OpticksRun.cc:225
        frame #8: 0x00000001000d4938 libOK.dylib`OKMgr::propagate(this=0x00007ffeefbfe8f8) at OKMgr.cc:123
        frame #9: 0x000000010000b957 OKTest`main(argc=2, argv=0x00007ffeefbfe9b8) at OKTest.cc:32
        frame #10: 0x00007fff52164015 libdyld.dylib`start + 1
        frame #11: 0x00007fff52164015 libdyld.dylib`start + 1
    (lldb) 




::

    (lldb) f 8
    frame #8: 0x00000001000d4938 libOK.dylib`OKMgr::propagate(this=0x00007ffeefbfe928) at OKMgr.cc:123
       120 	        {
       121 	            m_run->createEvent(i);
       122 	
    -> 123 	            m_run->setGensteps(m_gen->getInputGensteps()); 
       124 	
       125 	            m_propagator->propagate();
       126 	
    (lldb) 


::

    epsilon:optickscore blyth$ np.py $TMP/OpticksRun_importGenstepData/dbggsimport.npy -v -i
    a : /tmp/blyth/opticks/OpticksRun_importGenstepData/dbggsimport.npy :            (1, 6, 4) : 98618a7484b3aad5e9e0abdb2fcb4de5 : 20200930-1641 
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
    [[[[      4096          0         95      10000]
       [         0          0          0 1036831949]
       [         0          0 1065353216 1065353216]
       [         0          0 1065353216 1138163712]
       [         0 1065353216          0 1065353216]
       [         0          0          0          1]]]]
    epsilon:optickscore blyth$ 


::

    In [3]: 0x1 << 12
    Out[3]: 4096


Genstep labelling used to use OpticksPhoton.h codes::

     22 enum
     23 {
     24     CERENKOV          = 0x1 <<  0,
     25     SCINTILLATION     = 0x1 <<  1,
     26     MISS              = 0x1 <<  2,
     27     BULK_ABSORB       = 0x1 <<  3,
     28     BULK_REEMIT       = 0x1 <<  4,
     29     BULK_SCATTER      = 0x1 <<  5,
     30     SURFACE_DETECT    = 0x1 <<  6,
     31     SURFACE_ABSORB    = 0x1 <<  7,
     32     SURFACE_DREFLECT  = 0x1 <<  8,
     33     SURFACE_SREFLECT  = 0x1 <<  9,
     34     BOUNDARY_REFLECT  = 0x1 << 10,
     35     BOUNDARY_TRANSMIT = 0x1 << 11,
     36     TORCH             = 0x1 << 12, 
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     37     NAN_ABORT         = 0x1 << 13,
     38     __G4GUN           = 0x1 << 14,
     39     __FABRICATED      = 0x1 << 15,
     40     __NATURAL         = 0x1 << 16,
     41     __MACHINERY       = 0x1 << 17,
     42     __EMITSOURCE      = 0x1 << 18,
     43     PRIMARYSOURCE     = 0x1 << 19,
     44     GENSTEPSOURCE     = 0x1 << 20
     45 };


Following changes in summer 2020 using OpticksGenstep.h::

     19 enum
     20 {
     21     OpticksGenstep_INVALID                  = 0,
     22     OpticksGenstep_G4Cerenkov_1042          = 1,
     23     OpticksGenstep_G4Scintillation_1042     = 2,
     24     OpticksGenstep_DsG4Cerenkov_r3971       = 3,
     25     OpticksGenstep_DsG4Scintillation_r3971  = 4,
     26     OpticksGenstep_TORCH                    = 5,
     27     OpticksGenstep_FABRICATED               = 6,
     28     OpticksGenstep_EMITSOURCE               = 7,
     29     OpticksGenstep_NATURAL                  = 8,
     30     OpticksGenstep_MACHINERY                = 9,
     31     OpticksGenstep_G4GUN                    = 10,
     32     OpticksGenstep_PRIMARYSOURCE            = 11,
     33     OpticksGenstep_GENSTEPSOURCE            = 12,
     34     OpticksGenstep_NumType                  = 13
     35 };
     36 



    epsilon:optickscore blyth$ OpticksGenstepTest 
    2020-09-30 16:46:26.095 INFO  [55283] [main@32] OpticksGenstep::Dump()
    2020-09-30 16:46:26.096 INFO  [55283] [main@33] 
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




