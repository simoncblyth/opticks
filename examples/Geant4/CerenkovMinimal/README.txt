CerenkovMinimal
==================

Objective
-----------

CerenkovMinimal aims to show minimal use of embedded Opticks
to act as an starting point for users familiar with Geant4 examples.

Overview
----------

The idea is to simplify Opticks usage by everything going 
through the G4Opticks interface. This intentionally constrains
Opticks functionality and flexibility.   

Because of this, much of the development of CerekovMinimal will 
involve adaption to g4ok- G4Opticks in order to 


Full Opticks usage, via geocache
----------------------------------

The geocache and gensteps created from the simple CerenkovMinimal 
Geant4 geometry can be used from full unconstrained Opticks executables 
by setting the OPTICKS_KEY reported by CerenkovMinimal.

NB This only works in one direction, general geocache cannot 
be used by CerenkovMinimal.  


Classes
--------

G4.hh
    main struct thats hold Geant4 actions, managers etc..

DetectorConstruction.hh
    simple hardcoded Geant4 construction of detector

RunAction.hh
    RunAction::BeginOfRunAction passes the Geant4 geometry to Opticks 
    with G4Opticks::GetOpticks()->setGeometry

Ctx.hh
    context struct used via setEvent setTrack setStep etc.. 
    to maintain convenient state access

    NB Ctx::setTrack has hardcoded track killing 
    after collecting a single genstep for fast cycle debugging

Cerenkov.hh
    shim on top of G4Cerenkov for debug dumping only

EventAction.hh
    EventAction::EndOfEventAction invokes the Opticks propagation

PrimaryGeneratorAction.hh
    simple hardcoded setup of a primary

L4Cerenkov.hh
    G4VProcess subclass based on G4Cerenkov that collects Cerenkov gensteps
    using G4Opticks. Currently the CPU photon generation loop is still being 
    done for comparison purposes.

OpHit.hh
    undeveloped 

PhysicsList.hh
    EM and Optical physics, ConstructOp is templated, currently using 
    L4Cerenkov 

SensitiveDetector.hh
    SensitiveDetector::ProcessHits is for standard Geant4 hits not ones from GPU.
    Fairly undeveloped, needs machinery to shovel GPU hits in bulk into 
    collections.

SteppingAction.hh
    SteppingAction::UserSteppingAction sets the G4Step into the Ctx

TrackInfo.hh
    G4VUserTrackInformation subclass that holds a photon_record_id, this 
    is convenient for example when multiple Geant4 events correspond 
    to a single Opticks event and also when doing reemission continuation : 
    ie maintaining the original identity of a photon through a reemission in 
    Geant4 to allow comparison with Opticks that does this naturally

TrackingAction.hh
    currently PreUserTrackingAction PostUserTrackingAction just updates Ctx









