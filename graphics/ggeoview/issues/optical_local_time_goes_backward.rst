Optical Local Time Goes Backward
===================================

Presumably some bad material props ? 

::

    ggv-;ggv-g4gun --dbg


::

      G4ParticleChange::CheckIt    : the local time goes back  !!  Difference:  6.9082[ns] 
      opticalphoton E=3.19939e-06 pos=-18.0795, -799.699, -6.59899 global time=11.7134 local time=0 proper time=0
          -----------------------------------------------
            G4ParticleChange Information  
          -----------------------------------------------
            # of 2ndaries       :                    0
          -----------------------------------------------
            Energy Deposit (MeV):                    0
            Non-ionizing Energy Deposit (MeV):                    0
            Track Status        :                Alive
            True Path Length (mm) :            -1.29e+03
            Stepping Control      :                    0
        First Step In the voulme  : 
            Mass (GeV)   :                    0
            Charge (eplus)   :                    0
            MagneticMoment   :                    0
                    :  =                    0*[e hbar]/[2 m]
            Position - x (mm)   :            -1.87e+04
            Position - y (mm)   :               -8e+05
            Position - z (mm)   :            -5.59e+03
            Time (ns)           :                -6.91
            Proper Time (ns)    :                    0
            Momentum Direct - x :                0.461
            Momentum Direct - y :                0.425
            Momentum Direct - z :               -0.779
            Kinetic Energy (MeV):              3.2e-06
            Velocity  (/c):                    1
            Polarization - x    :               -0.515
            Polarization - y    :               -0.587
            Polarization - z    :               -0.625
            Touchable (pointer) :          0x10f015e50

::

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : TRACK003
          issued by : G4ParticleChange::CheckIt
    momentum, energy, and/or time was illegal
    *** Event Must Be Aborted ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------

      G4VParticleChange::CheckIt    : the true step length is negative  !!  Difference:  1289.99[MeV] 
    opticalphoton E=3.19939e-06 pos=-18.6742, -800.248, -5.59423
          -----------------------------------------------
            G4ParticleChange Information  


::

    (lldb) b "G4ParticleChange::CheckIt(G4Track const&)" 


    (lldb) bt
    * thread #1: tid = 0x71d4a0, 0x0000000105ed6e60 libG4track.dylib`G4ParticleChange::CheckIt(this=0x00000001092886c0, aTrack=0x000000010e3338c0) + 32 at G4ParticleChange.cc:508, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000105ed6e60 libG4track.dylib`G4ParticleChange::CheckIt(this=0x00000001092886c0, aTrack=0x000000010e3338c0) + 32 at G4ParticleChange.cc:508
        frame #1: 0x0000000105ee063f libG4track.dylib`G4ParticleChangeForTransport::UpdateStepForAlongStep(this=0x00000001092886c0, pStep=0x000000010910e410) + 1519 at G4ParticleChangeForTransport.cc:202
        frame #2: 0x0000000102ecf96e libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x000000010910e280) + 254 at G4SteppingManager2.cc:420
        frame #3: 0x0000000102ecb168 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000010910e280) + 504 at G4SteppingManager.cc:191
        frame #4: 0x0000000102ee292d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000010910e240, apValueG4Track=0x000000010e3338c0) + 1357 at G4TrackingManager.cc:126
        frame #5: 0x0000000102dbfe44 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000010910e1b0, anEvent=0x000000010e3323d0) + 3188 at G4EventManager.cc:185
        frame #6: 0x0000000102dc0b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x000000010910e1b0, anEvent=0x000000010e3323d0) + 47 at G4EventManager.cc:336
        frame #7: 0x0000000102cedc75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000109003060, i_event=0) + 69 at G4RunManager.cc:399
        frame #8: 0x0000000102cedab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000109003060, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #9: 0x0000000102cec8e4 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000109003060, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #10: 0x000000010155b38d libcfg4.dylib`CG4::propagate(this=0x0000000108721880) + 605 at CG4.cc:181
        frame #11: 0x000000010000d542 CG4Test`main(argc=16, argv=0x00007fff5fbfdca0) + 210 at CG4Test.cc:20
        frame #12: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) 



::

    simon:ggeo blyth$ g4-cc debugFlag
    /usr/local/env/g4/geant4.10.02/source/processes/parameterisation/src/G4FastStep.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/processes/parameterisation/src/G4FastStep.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChange.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChange.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChange.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChangeForDecay.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChangeForGamma.cc:  debugFlag = false;
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChangeForLoss.cc:  debugFlag = false;
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChangeForTransport.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4VParticleChange.cc:   debugFlag(false)
    /usr/local/env/g4/geant4.10.02/source/track/src/G4VParticleChange.cc:  debugFlag = true;
    /usr/local/env/g4/geant4.10.02/source/track/src/G4VParticleChange.cc:   debugFlag(right.debugFlag)
    /usr/local/env/g4/geant4.10.02/source/track/src/G4VParticleChange.cc:    debugFlag = right.debugFlag;
    /usr/local/env/g4/geant4.10.02/source/track/src/G4VParticleChange.cc:  if (debugFlag) CheckSecondary(*aTrack);
    simon:ggeo blyth$ 




