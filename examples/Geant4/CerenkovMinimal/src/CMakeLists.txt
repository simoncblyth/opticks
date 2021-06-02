
#SteppingAction.cc
#TrackingAction.cc
#EventAction.cc
#RunAction.cc
#Ctx.cc

#SteppingAction.hh
#TrackingAction.hh
#EventAction.hh
#RunAction.hh
#Ctx.hh


target_sources(${name}
    PRIVATE
        DetectorConstruction.cc
        PhysicsList.cc
        L4Cerenkov.cc
        CKMScintillation.cc
        PrimaryGeneratorAction.cc

        RAction.cc
        EAction.cc
        TAction.cc
        SAction.cc

        AManager.cc
        CKM.cc
        SensitiveDetector.cc
        OpHit.cc
    PUBLIC
        DetectorConstruction.hh
        PhysicsList.hh
        L4Cerenkov.hh
        CKMScintillation.h
        PrimaryGeneratorAction.hh
        AManager.hh

        RAction.hh
        EAction.hh
        TAction.hh
        SAction.hh

        CKM.hh
        SensitiveDetector.hh
        OpHit.hh
)
