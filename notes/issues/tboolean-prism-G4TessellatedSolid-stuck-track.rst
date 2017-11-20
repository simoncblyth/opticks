tboolean-prism-G4TessellatedSolid-stuck-track
================================================


ISSUE : 4/100000 stuck photons with G4TesselatedSolid
--------------------------------------------------------

::


    tboolean-;tboolean-prism --okg4 



    -------- WWWW ------- G4Exception-START -------- WWWW -------
    *** G4Exception : GeomNav1002
          issued by : G4Navigator::ComputeStep()
    Track stuck or not moving.
              Track stuck, not moving for 10 steps
              in volume -convexpolyhedron_pv1_- at point (-8.06931,80.519,56.2196)
              direction: (0.17915,0.0276245,0.983434).
              Potential geometry or navigation problem !
              Trying pushing it of 1e-07 mm ...Potential overlap in geometry!

    *** This is just a warning message. ***
    -------- WWWW -------- G4Exception-END --------- WWWW -------



Dump steps 
------------


* all 4 SC, then have some TIR before getting stuck in STS 

::

    tboolean-;tboolean-prism --okg4 --dindex 34103,49387,52757  --dbgzero


    2017-11-20 18:15:19.082 INFO  [5988239] [CGenerator::configureEvent@124] CGenerator:configureEvent fabricated TORCH genstep (STATIC RUNNING) 
    2017-11-20 18:15:19.143 INFO  [5988239] [CG4Ctx::initEvent@134] CG4Ctx::initEvent photons_per_g4event 10000 steps_per_photon 10 gen 4096
    2017-11-20 18:15:19.143 INFO  [5988239] [CWriter::initEvent@87] CWriter::initEvent dynamic STATIC(GPU style) record_max 100000 bounce_max  9 steps_per_photon 10 num_g4event 10
    2017-11-20 18:15:19.143 INFO  [5988239] [CRec::initEvent@82] CRec::initEvent note recstp
    2017-11-20 18:15:19.435 INFO  [5988239] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2017-11-20 18:15:19.917 INFO  [5988239] [CSteppingAction::setStep@111]  noZeroSteps 2 severity 1 ctx  --dindex  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:15:19.917 INFO  [5988239] [CSteppingAction::setStep@111]  noZeroSteps 3 severity 1 ctx  --dindex  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:15:19.917 INFO  [5988239] [CSteppingAction::setStep@111]  noZeroSteps 4 severity 2 ctx  --dindex  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:15:19.917 INFO  [5988239] [CSteppingAction::setStep@111]  noZeroSteps 5 severity 2 ctx  --dindex  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:15:19.917 INFO  [5988239] [CSteppingAction::setStep@111]  noZeroSteps 6 severity 3 ctx  --dindex  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:15:19.917 INFO  [5988239] [CSteppingAction::setStep@111]  noZeroSteps 7 severity 3 ctx  --dindex  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:15:19.917 INFO  [5988239] [CSteppingAction::setStep@111]  noZeroSteps 8 severity 4 ctx  --dindex  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:15:19.917 INFO  [5988239] [CSteppingAction::setStep@111]  noZeroSteps 9 severity 4 ctx  --dindex  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0

    -------- WWWW ------- G4Exception-START -------- WWWW -------
    *** G4Exception : GeomNav1002
          issued by : G4Navigator::ComputeStep()
    Track stuck or not moving.
              Track stuck, not moving for 10 steps
              in volume -convexpolyhedron_pv1_- at point (-8.06931,80.519,56.2196)
              direction: (0.17915,0.0276245,0.983434).
              Potential geometry or navigation problem !
              Trying pushing it of 1e-07 mm ...Potential overlap in geometry!

    *** This is just a warning message. ***
    -------- WWWW -------- G4Exception-END --------- WWWW -------

    2017-11-20 18:15:19.917 INFO  [5988239] [CSteppingAction::setStep@111]  noZeroSteps 10 severity 5 ctx  --dindex  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:15:19.917 INFO  [5988239] [CDebug::dump@159] CDebug::posttrack
    2017-11-20 18:15:19.917 INFO  [5988239] [CRec::dump@157] CDebug::dump record_id 34103  origin[ 16.59646.860599.000]   Ori[ 16.59646.860599.000] 
    2017-11-20 18:15:19.917 INFO  [5988239] [CRec::dump@163]  nstp 16
    ( 0)  TO/BT     FrT                       PRE_SAVE POST_SAVE STEP_START 
    [   0](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre                  box_pv0_          Vacuum          noProc           Undefined pos[      0.000     0.000     0.000]  dir[    0.000   0.000  -1.000]  pol[    0.707   0.707   0.000]  ns  0.100 nm 500.000 mm/ns 299.792
     post     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[      0.000     0.000  -449.000]  dir[    0.000   0.000  -1.000]  pol[    0.707   0.707   0.000]  ns  1.598 nm 500.000 mm/ns 175.914
     )
    ( 1)  BT/BT     FrT                                           POST_SAVE 
    [   1](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[      0.000     0.000  -449.000]  dir[    0.000   0.000  -1.000]  pol[    0.707   0.707   0.000]  ns  1.598 nm 500.000 mm/ns 175.914
     post                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[      0.000     0.000  -749.000]  dir[    0.000   0.000  -1.000]  pol[    0.707   0.707   0.000]  ns  3.303 nm 500.000 mm/ns 299.792
     )
    ( 2)  BT/SC     NAB                                           POST_SAVE 
    [   2](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[      0.000     0.000  -749.000]  dir[    0.000   0.000  -1.000]  pol[    0.707   0.707   0.000]  ns  3.303 nm 500.000 mm/ns 299.792
     post                  box_pv0_          Vacuum      OpRayleigh    PostStepDoItProc pos[      0.000     0.000  -761.024]  dir[   -0.175   0.238   0.955]  pol[    0.716   0.697  -0.043]  ns  3.343 nm 500.000 mm/ns 299.792
     )
    ( 3)  SC/BT     FrT                                           POST_SAVE 
    [   3](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre                  box_pv0_          Vacuum      OpRayleigh    PostStepDoItProc pos[      0.000     0.000  -761.024]  dir[   -0.175   0.238   0.955]  pol[    0.716   0.697  -0.043]  ns  3.343 nm 500.000 mm/ns 299.792
     post     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[     -2.198     2.999  -749.000]  dir[   -0.107   0.146   0.983]  pol[    0.883   0.468   0.027]  ns  3.385 nm 500.000 mm/ns 175.914
     )
    ( 4)  BT/BR     TIR                                  POST_SAVE MAT_SWAP 
    [   4](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[     -2.198     2.999  -749.000]  dir[   -0.107   0.146   0.983]  pol[    0.883   0.468   0.027]  ns  3.385 nm 500.000 mm/ns 175.914
     post                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     )
    ( 5)  BR/NA     STS                                  POST_SKIP MAT_SWAP 
    [   5](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     post     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     )
    ( 6)  NA/NA     STS                                  POST_SKIP MAT_SWAP 
    [   6](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     post                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     )
    ( 7)  NA/NA     STS                                  POST_SKIP MAT_SWAP 
    [   7](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     post     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     )
    ( 8)  NA/NA     STS                                  POST_SKIP MAT_SWAP 
    [   8](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     post                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     )
    ( 9)  NA/NA     STS                                  POST_SKIP MAT_SWAP 
    [   9](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     post     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     )
    (10)  NA/NA     STS                                  POST_SKIP MAT_SWAP 
    [  10](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     post                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     )
    (11)  NA/NA     STS                                  POST_SKIP MAT_SWAP 
    [  11](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     post     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     )
    (12)  NA/NA     STS                                  POST_SKIP MAT_SWAP 
    [  12](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     post                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     )
    (13)  NA/NA     STS                                           POST_SKIP 
    [  13](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     post     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     )
    (14)  NA/BR     TIR                                           POST_SAVE 
    [  14](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre     convexpolyhedron_pv1_   GlassSchottF2  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[    0.179   0.028   0.983]  pol[    0.293  -0.956  -0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     post                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[   -0.107   0.146   0.983]  pol[    0.883   0.468   0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     )
    (15)  BR/SA     Abs              POST_SAVE POST_DONE LAST_POST SURF_ABS 
    [  15](Stp ;opticalphoton stepNum   16(tk ;opticalphoton tid 4104 pid 0 nm    500 mm  ori[   16.596  46.860 599.000]  pos[  -83.910 114.505   1.000]  )
      pre                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -24.665    33.659  -542.780]  dir[   -0.107   0.146   0.983]  pol[    0.883   0.468   0.027]  ns  4.577 nm 500.000 mm/ns 175.914
     post               UNIVERSE_PV            Rock  Transportation        GeomBoundary pos[    -83.910   114.505     1.000]  dir[   -0.107   0.146   0.983]  pol[    0.883   0.468   0.027]  ns  7.720 nm 500.000 mm/ns 175.914
     )
    2017-11-20 18:15:19.919 INFO  [5988239] [CRec::dump@167]  npoi 0
    2017-11-20 18:15:19.919 INFO  [5988239] [CDebug::dump_brief@176] CRecorder::dump_brief m_ctx._record_id    34103 m_photon._badflag     0 --dindex  sas: POST_SAVE POST_DONE LAST_POST SURF_ABS 
    2017-11-20 18:15:19.920 INFO  [5988239] [CDebug::dump_brief@185]  seqhis         8bbc6ccd    TO BT BT SC BT BR BR SA                         
    2017-11-20 18:15:19.920 INFO  [5988239] [CDebug::dump_brief@190]  mskhis             1ca0    SC|SA|BR|BT|TO
    2017-11-20 18:15:19.920 INFO  [5988239] [CDebug::dump_brief@195]  seqmat         12332232    Vm F2 Vm Vm F2 F2 Vm Rk - - - - - - - - 
    2017-11-20 18:15:19.920 INFO  [5988239] [CDebug::dump_sequence@203] CDebug::dump_sequence
    2017-11-20 18:15:19.920 INFO  [5988239] [CDebug::dump_points@229] CDeug::dump_points
    2017-11-20 18:15:19.920 INFO  [5988239] [CRecorder::post





Identify the stuck photons, in CSteppingAction
--------------------------------------------------


::

    2017-11-20 18:08:44.585 INFO  [5986683] [CRec::initEvent@82] CRec::initEvent note recstp
    2017-11-20 18:08:44.875 INFO  [5986683] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2017-11-20 18:08:45.387 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 2 severity 1 ctx  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.387 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 3 severity 1 ctx  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.387 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 4 severity 2 ctx  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.387 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 5 severity 2 ctx  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.387 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 6 severity 3 ctx  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.387 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 7 severity 3 ctx  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.387 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 8 severity 4 ctx  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.387 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 9 severity 4 ctx  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0

    -------- WWWW ------- G4Exception-START -------- WWWW -------
    *** G4Exception : GeomNav1002
          issued by : G4Navigator::ComputeStep()
    Track stuck or not moving.
              Track stuck, not moving for 10 steps
              in volume -convexpolyhedron_pv1_- at point (-8.06931,80.519,56.2196)
              direction: (0.17915,0.0276245,0.983434).
              Potential geometry or navigation problem !
              Trying pushing it of 1e-07 mm ...Potential overlap in geometry!

    *** This is just a warning message. ***
    -------- WWWW -------- G4Exception-END --------- WWWW -------

    2017-11-20 18:08:45.387 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 10 severity 5 ctx  record_id 34103 event_id 3 track_id 4103 photon_id 4103 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.457 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 2 severity 1 ctx  record_id 49387 event_id 4 track_id 9387 photon_id 9387 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.457 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 3 severity 1 ctx  record_id 49387 event_id 4 track_id 9387 photon_id 9387 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.457 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 4 severity 2 ctx  record_id 49387 event_id 4 track_id 9387 photon_id 9387 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.457 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 5 severity 2 ctx  record_id 49387 event_id 4 track_id 9387 photon_id 9387 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.457 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 6 severity 3 ctx  record_id 49387 event_id 4 track_id 9387 photon_id 9387 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.457 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 7 severity 3 ctx  record_id 49387 event_id 4 track_id 9387 photon_id 9387 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.457 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 8 severity 4 ctx  record_id 49387 event_id 4 track_id 9387 photon_id 9387 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.457 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 9 severity 4 ctx  record_id 49387 event_id 4 track_id 9387 photon_id 9387 parent_id -1 primary_id -2 reemtrack 0

    -------- WWWW ------- G4Exception-START -------- WWWW -------
    *** G4Exception : GeomNav1002
          issued by : G4Navigator::ComputeStep()
    Track stuck or not moving.
              Track stuck, not moving for 10 steps
              in volume -convexpolyhedron_pv1_- at point (30.7987,25.6454,-101.635)
              direction: (-0.145311,-0.934096,-0.326113).
              Potential geometry or navigation problem !
              Trying pushing it of 1e-07 mm ...Potential overlap in geometry!

    *** This is just a warning message. ***
    -------- WWWW -------- G4Exception-END --------- WWWW -------

    2017-11-20 18:08:45.457 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 10 severity 5 ctx  record_id 49387 event_id 4 track_id 9387 photon_id 9387 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.682 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 2 severity 1 ctx  record_id 52757 event_id 5 track_id 2757 photon_id 2757 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.682 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 3 severity 1 ctx  record_id 52757 event_id 5 track_id 2757 photon_id 2757 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.682 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 4 severity 2 ctx  record_id 52757 event_id 5 track_id 2757 photon_id 2757 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.682 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 5 severity 2 ctx  record_id 52757 event_id 5 track_id 2757 photon_id 2757 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.682 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 6 severity 3 ctx  record_id 52757 event_id 5 track_id 2757 photon_id 2757 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.682 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 7 severity 3 ctx  record_id 52757 event_id 5 track_id 2757 photon_id 2757 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.682 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 8 severity 4 ctx  record_id 52757 event_id 5 track_id 2757 photon_id 2757 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:08:45.682 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 9 severity 4 ctx  record_id 52757 event_id 5 track_id 2757 photon_id 2757 parent_id -1 primary_id -2 reemtrack 0

    -------- WWWW ------- G4Exception-START -------- WWWW -------
    *** G4Exception : GeomNav1002
          issued by : G4Navigator::ComputeStep()
    Track stuck or not moving.
              Track stuck, not moving for 10 steps
              in volume -convexpolyhedron_pv1_- at point (13.3368,67.8021,-102.268)
              direction: (-0.641094,0.0443596,-0.766179).
              Potential geometry or navigation problem !
              Trying pushing it of 1e-07 mm ...Potential overlap in geometry!

    *** This is just a warning message. ***
    -------- WWWW -------- G4Exception-END --------- WWWW -------

    2017-11-20 18:08:45.682 INFO  [5986683] [CSteppingAction::setStep@111]  noZeroSteps 10 severity 5 ctx  record_id 52757 event_id 5 track_id 2757 photon_id 2757 parent_id -1 primary_id -2 reemtrack 0




G4Navigator
---------------


::

    g4-;g4-cls G4Navigator


    0995     if( fNumberZeroSteps > fActionThreshold_NoZeroSteps-1 )
     996     {
     997        // Act to recover this stuck track. Pushing it along direction
     998        //
     999        Step += 100*kCarTolerance;
    1000 #ifdef G4VERBOSE
    1001        if ((!fPushed) && (fWarnPush))
    1002        {
    1003          std::ostringstream message;
    1004          message << "Track stuck or not moving." << G4endl
    1005                  << "          Track stuck, not moving for "
    1006                  << fNumberZeroSteps << " steps" << G4endl
    1007                  << "          in volume -" << motherPhysical->GetName()
    1008                  << "- at point " << pGlobalpoint << G4endl
    1009                  << "          direction: " << pDirection << "." << G4endl
    1010                  << "          Potential geometry or navigation problem !"
    1011                  << G4endl
    1012                  << "          Trying pushing it of " << Step << " mm ...";
    1013          G4Exception("G4Navigator::ComputeStep()", "GeomNav1002",
    1014                      JustWarning, message, "Potential overlap in geometry!");
    1015        }
    1016 #endif
    1017        fPushed = true;
    1018     }




