BR_PhysicalStep_zero_misalignment
==================================


Strategy
----------

Understand what G4 condition yields the zero-step, detect it 
on Opticks side and burn the requisite number(4) of RNG 
to stay in alignment. 



Turnaround dump
----------------

::

    2017-12-11 16:05:40.024 ERROR [1689439] [CRandomEngine::pretrack@256] CRandomEngine::pretrack record_id:  ctx.record_id 0 index 1230 mask.size 1
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[0] : 0.001117024919949472 1  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[1] : 0.50264734029769897 2  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[2] : 0.60150414705276489 3  
    G4SteppingManager2_cc_181_ : Dumping lengths collected by _181 after PostStep process loop  
    //                                                  .fCurrentProcess.theProcessName :  OpBoundary  
    //                                                                   .physIntLength :  1.79769e+308  
    //                                                  .fCurrentProcess.theProcessName :  OpRayleigh  
    //                                                                   .physIntLength :  687866  
    //                                                  .fCurrentProcess.theProcessName :  OpAbsorption  
    //                                                                   .physIntLength :  5.08322e+06  
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  1.79769e+308  

    //                                opticks.ana.cfg4lldb.G4Transportation_cc_517_.[0] : AlongStepGetPhysicalInteractionLength Exit  
    //                                                                             this : G4Transportation_cc_517 
    //                                                                   /startPosition :  ( -37.879   11.823 -449.900)  
    //                                                                /startMomentumDir :  (  -0.000   -0.000    1.000)  
    //                                                                       /newSafety :  0.100006  
    //                                                            .fGeometryLimitedStep : True 
    //                                                              .fFirstStepInVolume : True 
    //                                                               .fLastStepInVolume : False 
    //                                                                .fMomentumChanged : False 
    //                                                           .fTransportEndPosition :  ( -37.879   11.823 -100.000)  
    //                                                        .fTransportEndMomentumDir :  (  -0.000   -0.000    1.000)  
    //                                                               .fEndPointDistance :  349.9  
    //                                               .fParticleChange.thePositionChange :  (   0.000    0.000    0.000)  
    //                                      .fParticleChange.theMomentumDirectionChange :  (   0.000    0.000    0.000)  
    Process 75886 stopped
    * thread #1: tid = 0x19c75f, 0x0000000105b5a3ce libG4processes.dylib`G4Transportation::AlongStepGetPhysicalInteractionLength(this=0x0000000110964190, track=<unavailable>, (null)=<unavailable>, currentMinimumStep=<unavailable>, currentSafety=<unavailable>, selection=<unavailable>) + 3486 at G4Transportation.cc:518, queue = 'com.apple.main-thread', stop reason = breakpoint 6.1
        frame #0: 0x0000000105b5a3ce libG4processes.dylib`G4Transportation::AlongStepGetPhysicalInteractionLength(this=0x0000000110964190, track=<unavailable>, (null)=<unavailable>, currentMinimumStep=<unavailable>, currentSafety=<unavailable>, selection=<unavailable>) + 3486 at G4Transportation.cc:518
       515    fParticleChange.ProposeTrueStepLength(geometryStepLength) ;
       516  
       517    return geometryStepLength ;
    -> 518  }
       519  
       520  //////////////////////////////////////////////////////////////////////////
       521  //
    (lldb) c
    Process 75886 resuming

    //                              opticks.ana.cfg4lldb.G4SteppingManager2_cc_270_.[0] : Near end of DefinePhysicalStepLength : Inside MAXofAlongStepLoops after AlongStepGPIL 
    //                                                                             this : G4SteppingManager2_cc_270 
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  349.9  
    //                                                                    .PhysicalStep :  349.9  
    //                                                                     .fStepStatus :  fGeomBoundary  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[3] : 0.93871349096298218 4  

    //                               opticks.ana.cfg4lldb.G4TrackingManager_cc_131_.[0] : Step Conclusion : TrackingManager step loop just after Stepping()  
    //                                                                             this : G4TrackingManager 
    //                                                   .fpSteppingManager.fStepStatus :  fGeomBoundary  
    //                                                  .fpSteppingManager.PhysicalStep :  349.9  
    //                                .fpSteppingManager.fCurrentProcess.theProcessName :  OpBoundary  
    //                                .fpSteppingManager.fStep.fpPreStepPoint.fPosition :  ( -37.879   11.823 -449.900)  
    //                              .fpSteppingManager.fStep.fpPreStepPoint.fGlobalTime :  0.2  
    //                       .fpSteppingManager.fStep.fpPreStepPoint.fMomentumDirection :  (  -0.000   -0.000    1.000)  
    //                               .fpSteppingManager.fStep.fpPostStepPoint.fPosition :  ( -37.879   11.823 -100.000)  
    //                             .fpSteppingManager.fStep.fpPostStepPoint.fGlobalTime :  1.36714  
    //                      .fpSteppingManager.fStep.fpPostStepPoint.fMomentumDirection :  (   0.000    0.000   -1.000)  
    //                                                             CRandomEngine_cc_210 : 4 
    //                                                         G4Transportation_cc_517_ : 1 
    //                                                        G4TrackingManager_cc_131_ : 1 
    //                                                       G4SteppingManager2_cc_270_ : 1 
    //                                                       G4SteppingManager2_cc_181_ : 0 
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[4] : 0.75380146503448486 5  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[5] : 0.99984675645828247 6  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[6] : 0.43801957368850708 7  
    G4SteppingManager2_cc_181_ : Dumping lengths collected by _181 after PostStep process loop  
    //                                                  .fCurrentProcess.theProcessName :  OpBoundary  
    //                                                                   .physIntLength :  1.79769e+308  
    //                                                  .fCurrentProcess.theProcessName :  OpRayleigh  
    //                                                                   .physIntLength :  153.255  
    //                                                  .fCurrentProcess.theProcessName :  OpAbsorption  
    //                                                                   .physIntLength :  825492  
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  1.79769e+308  

    //                                opticks.ana.cfg4lldb.G4Transportation_cc_517_.[1] : AlongStepGetPhysicalInteractionLength Exit  
    //                                                                             this : G4Transportation_cc_517 
    //                                                                   /startPosition :  ( -37.879   11.823 -100.000)  
    //                                                                /startMomentumDir :  (   0.000    0.000   -1.000)  
    //                                                                       /newSafety :  0  
    //                                                            .fGeometryLimitedStep : True 
    //                                                              .fFirstStepInVolume : True 
    //                                                               .fLastStepInVolume : False 
    //                                                                .fMomentumChanged : False 
    //                                                           .fTransportEndPosition :  ( -37.879   11.823 -100.000)  
    //                                                        .fTransportEndMomentumDir :  (   0.000    0.000   -1.000)  
    //                                                               .fEndPointDistance :  0  
    //                                               .fParticleChange.thePositionChange :  ( -37.879   11.823 -100.000)  
    //                                      .fParticleChange.theMomentumDirectionChange :  (  -0.000   -0.000    1.000)  
    Process 75886 stopped
    * thread #1: tid = 0x19c75f, 0x0000000105b5a3ce libG4processes.dylib`G4Transportation::AlongStepGetPhysicalInteractionLength(this=0x0000000110964190, track=<unavailable>, (null)=<unavailable>, currentMinimumStep=<unavailable>, currentSafety=<unavailable>, selection=<unavailable>) + 3486 at G4Transportation.cc:518, queue = 'com.apple.main-thread', stop reason = breakpoint 6.1
        frame #0: 0x0000000105b5a3ce libG4processes.dylib`G4Transportation::AlongStepGetPhysicalInteractionLength(this=0x0000000110964190, track=<unavailable>, (null)=<unavailable>, currentMinimumStep=<unavailable>, currentSafety=<unavailable>, selection=<unavailable>) + 3486 at G4Transportation.cc:518
       515    fParticleChange.ProposeTrueStepLength(geometryStepLength) ;
       516  
       517    return geometryStepLength ;
    -> 518  }
       519  
       520  //////////////////////////////////////////////////////////////////////////
       521  //
    (lldb) 




Smouldering evidence : PhysicalStep-zero/StepTooSmall results in RNG mis-alignment 
------------------------------------------------------------------------------------

Some G4 technicality yields zero step at BR, that means the lucky scatter 
throw that Opticks saw was not seen by G4 : as the sequence gets out of alignment.

::

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_boundary_burn:    0.0011 
    propagate_to_boundary  u_scattering:    0.5026   scattering_distance:687866.4375 
    propagate_to_boundary  u_absorption:    0.6015   absorption_distance:5083218.0000 
    propagate_at_boundary  u_reflect:       0.93871  reflect:1   TransCoeff:   0.93847 
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:1 
    propagate_to_boundary  u_boundary_burn:    0.7538 
    propagate_to_boundary  u_scattering:    0.9998   scattering_distance:  153.2073 
    propagate_to_boundary  u_absorption:    0.4380   absorption_distance:8254916.0000 
    rayleigh_scatter
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:2 
    propagate_to_boundary  u_boundary_burn:    0.2825 
    propagate_to_boundary  u_scattering:    0.4325   scattering_distance:838178.1875 
    propagate_to_boundary  u_absorption:    0.9078   absorption_distance:966772.9375 
    propagate_at_surface   u_surface:       0.9121 
    propagate_at_surface   u_surface_burn:       0.2018 





::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 -D

    2017-12-11 14:57:46.221 ERROR [1667660] [CRandomEngine::pretrack@256] CRandomEngine::pretrack record_id:  ctx.record_id 0 index 1230 mask.size 1
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[0] : 0.001117024919949472 1  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[1] : 0.50264734029769897 2  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[2] : 0.60150414705276489 3  
    G4SteppingManager2_cc_181_ : Dumping lengths collected by _181 ffter PostStep process loop  
    //                                                  .fCurrentProcess.theProcessName :  OpBoundary  
    //                                                                   .physIntLength :  1.79769e+308  
    //                                                  .fCurrentProcess.theProcessName :  OpRayleigh  
    //                                                                   .physIntLength :  687866  
    //                                                  .fCurrentProcess.theProcessName :  OpAbsorption  
    //                                                                   .physIntLength :  5.08322e+06  
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  1.79769e+308  

    //                              opticks.ana.cfg4lldb.G4SteppingManager2_cc_270_.[0] : Near end of DefinePhysicalStepLength : Inside MAXofAlongStepLoops after AlongStepGPIL 
    //                                                                             this : G4SteppingManager2_cc_270 
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  349.9  
    //                                                                    .PhysicalStep :  349.9  
    //                                                                     .fStepStatus :  fGeomBoundary  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[3] : 0.93871349096298218 4  

    //                               opticks.ana.cfg4lldb.G4TrackingManager_cc_131_.[0] : Step Conclusion : TrackingManager step loop just after Stepping()  
    //                                                                             this : G4TrackingManager 
    //                                                   .fpSteppingManager.fStepStatus :  fGeomBoundary  
    //                                                  .fpSteppingManager.PhysicalStep :  349.9  
    //                                .fpSteppingManager.fCurrentProcess.theProcessName :  OpBoundary  
    //                                .fpSteppingManager.fStep.fpPreStepPoint.fPosition :  ( -37.879   11.823 -449.900)  
    //                              .fpSteppingManager.fStep.fpPreStepPoint.fGlobalTime :  0.2  
    //                       .fpSteppingManager.fStep.fpPreStepPoint.fMomentumDirection :  (  -0.000   -0.000    1.000)  
    //                               .fpSteppingManager.fStep.fpPostStepPoint.fPosition :  ( -37.879   11.823 -100.000)  
    //                             .fpSteppingManager.fStep.fpPostStepPoint.fGlobalTime :  1.36714  
    //                      .fpSteppingManager.fStep.fpPostStepPoint.fMomentumDirection :  (   0.000    0.000   -1.000)  
    //                                                             CRandomEngine_cc_210 : 4 
    //                                                        G4TrackingManager_cc_131_ : 1 
    //                                                       G4SteppingManager2_cc_270_ : 1 
    //                                                       G4SteppingManager2_cc_181_ : 0 
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[4] : 0.75380146503448486 5  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[5] : 0.99984675645828247 6  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[6] : 0.43801957368850708 7  
    G4SteppingManager2_cc_181_ : Dumping lengths collected by _181 ffter PostStep process loop  
    //                                                  .fCurrentProcess.theProcessName :  OpBoundary  
    //                                                                   .physIntLength :  1.79769e+308  
    //                                                  .fCurrentProcess.theProcessName :  OpRayleigh  
    //                                                                   .physIntLength :  153.255  
    //                                                  .fCurrentProcess.theProcessName :  OpAbsorption  
    //                                                                   .physIntLength :  825492  
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  1.79769e+308  

    //                              opticks.ana.cfg4lldb.G4SteppingManager2_cc_270_.[1] : Near end of DefinePhysicalStepLength : Inside MAXofAlongStepLoops after AlongStepGPIL 
    //                                                                             this : G4SteppingManager2_cc_270 
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  0  
    //                                                                    .PhysicalStep :  0  
    //                                                                     .fStepStatus :  fGeomBoundary  

    //                               opticks.ana.cfg4lldb.G4TrackingManager_cc_131_.[1] : Step Conclusion : TrackingManager step loop just after Stepping()  
    //                                                                             this : G4TrackingManager 
    //                                                   .fpSteppingManager.fStepStatus :  fGeomBoundary  
    //                                                  .fpSteppingManager.PhysicalStep :  0  
    //                                .fpSteppingManager.fCurrentProcess.theProcessName :  OpBoundary  
    //                                .fpSteppingManager.fStep.fpPreStepPoint.fPosition :  ( -37.879   11.823 -100.000)  
    //                              .fpSteppingManager.fStep.fpPreStepPoint.fGlobalTime :  1.36714  
    //                       .fpSteppingManager.fStep.fpPreStepPoint.fMomentumDirection :  (   0.000    0.000   -1.000)  
    //                               .fpSteppingManager.fStep.fpPostStepPoint.fPosition :  ( -37.879   11.823 -100.000)  
    //                             .fpSteppingManager.fStep.fpPostStepPoint.fGlobalTime :  1.36714  
    //                      .fpSteppingManager.fStep.fpPostStepPoint.fMomentumDirection :  (   0.000    0.000   -1.000)  
    //                                                             CRandomEngine_cc_210 : 7 
    //                                                        G4TrackingManager_cc_131_ : 2 
    //                                                       G4SteppingManager2_cc_270_ : 2 
    //                                                       G4SteppingManager2_cc_181_ : 0 
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[7] : 0.71403157711029053 8  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[8] : 0.33040395379066467 9  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[9] : 0.57074165344238281 10  
    G4SteppingManager2_cc_181_ : Dumping lengths collected by _181 ffter PostStep process loop  
    //                                                  .fCurrentProcess.theProcessName :  OpBoundary  
    //                                                                   .physIntLength :  1.79769e+308  
    //                                                  .fCurrentProcess.theProcessName :  OpRayleigh  
    //                                                                   .physIntLength :  1.10744e+06  
    //                                                  .fCurrentProcess.theProcessName :  OpAbsorption  
    //                                                                   .physIntLength :  5.60819e+06  
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  1.79769e+308  

    //                              opticks.ana.cfg4lldb.G4SteppingManager2_cc_270_.[2] : Near end of DefinePhysicalStepLength : Inside MAXofAlongStepLoops after AlongStepGPIL 
    //                                                                             this : G4SteppingManager2_cc_270 
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  350  
    //                                                                    .PhysicalStep :  350  
    //                                                                     .fStepStatus :  fGeomBoundary  
    //                                   opticks.ana.cfg4lldb.CRandomEngine_cc_210.[10] : 0.37590867280960083 11  
    //                                   opticks.ana.cfg4lldb.CRandomEngine_cc_210.[11] : 0.78497833013534546 12  

    //                               opticks.ana.cfg4lldb.G4TrackingManager_cc_131_.[2] : Step Conclusion : TrackingManager step loop just after Stepping()  
    //                                                                             this : G4TrackingManager 
    //                                                   .fpSteppingManager.fStepStatus :  fGeomBoundary  
    //                                                  .fpSteppingManager.PhysicalStep :  350  
    //                                .fpSteppingManager.fCurrentProcess.theProcessName :  OpBoundary  
    //                                .fpSteppingManager.fStep.fpPreStepPoint.fPosition :  ( -37.879   11.823 -100.000)  
    //                              .fpSteppingManager.fStep.fpPreStepPoint.fGlobalTime :  1.36714  
    //                       .fpSteppingManager.fStep.fpPreStepPoint.fMomentumDirection :  (   0.000    0.000   -1.000)  
    //                               .fpSteppingManager.fStep.fpPostStepPoint.fPosition :  ( -37.879   11.823 -450.000)  
    //                             .fpSteppingManager.fStep.fpPostStepPoint.fGlobalTime :  2.53462  
    //                      .fpSteppingManager.fStep.fpPostStepPoint.fMomentumDirection :  (   0.000    0.000   -1.000)  
    //                                                             CRandomEngine_cc_210 : 12 
    //                                                        G4TrackingManager_cc_131_ : 3 
    //                                                       G4SteppingManager2_cc_270_ : 3 
    //                                                       G4SteppingManager2_cc_181_ : 0 
    2017-12-11 14:57:46.775 INFO  [1667660] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1

