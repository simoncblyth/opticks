G4Transportation_AlongStepGPIL_zero
====================================



G4Transportation::AlongStepGPIL returning zero ?
---------------------------------------------------

::

   g4-;g4-cls G4Transportation



G4VProcess::AlongStepGPIL
---------------------------

::

   g4-;g4-cls G4VProcess

    187       // These three GPIL methods are used by Stepping Manager.
    188       // They invoke virtual GPIL methods listed above.
    189       // As for AtRest and PostStep the returned value is multipled by thePILfactor 
    190       // 
    191       G4double AlongStepGPIL( const G4Track& track,
    192                               G4double  previousStepSize,
    193                               G4double  currentMinimumStep,
    194                               G4double& proposedSafety,
    195                               G4GPILSelection* selection     );
    196 

    479 inline G4double G4VProcess::AlongStepGPIL( const G4Track& track,
    480                                      G4double  previousStepSize,
    481                                      G4double  currentMinimumStep,
    482                                      G4double& proposedSafety,
    483                                      G4GPILSelection* selection     )
    484 {
    485   G4double value
    486    =AlongStepGetPhysicalInteractionLength(track, previousStepSize, currentMinimumStep, proposedSafety, selection);
    487   return value;
    488 }


