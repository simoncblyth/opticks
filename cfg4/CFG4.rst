CFG4
======

Control
----------

CG4(OpticksHub* hub) 
    high level steering 

High Level Geometry
--------------------------------

CGeometry(OpticksHub* hub) 
    holder of CDetector 

CTestDetector
    G4VUserDetectorConstruction subclass, converts simple 
    russian doll Opticks GGeoTest/NCSG/nnode geometries 
    into Geant4 volume structures, using constituents: CMaker and CPropLib

CGDMLDetector
    G4VUserDetectorConstruction subclass, loads GDML persisted G4 geometries
    with help from constituent: CPropLib
    addMPT() fixup info omitted from older GDML 
     
CDetector(G4VUserDetectorConstruction)
    base class of CTestDetector and CGDMLDetector providing geometry to Geant4
    holding CMaterialLib, CSurfaceLib.
    The attachSurfaces() method fixes up info lost by GDML


Low Level Geometry : Materials/Surfaces
------------------------------------------

CPropLib  
    CPropLib is a constituent of CTestDetector that converts
    GGeo materials and surfaces into G4 materials and surfaces
CMPT
    extends G4MaterialPropertiesTable 

CMaterialLib
    converts GGeo materials into G4 materials, a CPropLib subclass
CMaterial
    extends G4Material just with a Digest 
CMaterialBridge
    wraps GMaterialLib, provides mapping between G4 and Opticks materials

CSurfaceBridge
    wraps GSurfaceLib, contains CSkinSurfaceTable, CBorderSurfaceTable
CSkinSurfaceTable
    subclass of CSurfaceTable
CBorderSurfaceTable
    subclass of CSurfaceTable
CSurfaceTable


Low Level Geometry : Solids
-----------------------------

CSolid
    wrapper for G4VSolid, providing extent 

CMaker
    converts NCSG/nnode shapes into G4VSolid 

Event
-------

CRandomEngine
    m_engine instance resident of CG4, provides control of random number stream for aligned running 
   
CCollector
    collects gensteps and primaries

CGenerator 
    m_generator instance resident of CG4 initializes CSource.m_source 
    in initSource of subclass CGunSource/CTorchSource/CInputPhotonSource
    depending of source code of G4GUN/TORCH/EMITSOURCE   

    This converts Opticks photon sources from hub into CSource(G4VPrimaryGenerator) 
    to be consumed by CPrimaryGeneratorAction further down the CG4 initializer list::

         m_generator(new CGenerator(m_hub, this)),
         ...
         m_pga(new CPrimaryGeneratorAction(m_generator->getSource())), 


CSource
    m_source instance resident of CGenerator of type G4GUN/TORCH/EMITSOURCE

    G4VPrimaryGenerator subclass, with `GeneratePrimaryVertex(G4Event *evt)`
    providing common functionality for the various source types

CTorchSource 
    Provides TorchStepNPY configurable optical photon squadrons just like the GPU eqivalent.
    Implemented using distribution generators from SingleParticleSource: 

    G4SPSPosDistribution
    G4SPSAngDistribution
    G4SPSEneDistribution

CInputPhotonSource 
    convert NPY buffer of input photons into an G4Event with primary vertices
    for each photon up to the maximum configured number per event       

CGunSource
    Converts NGunConfig into G4VPrimaryGenerator 
    with `GeneratePrimaryVertex(G4Event *evt)`

CPrimaryGeneratorAction
    isa G4VUserPrimaryGeneratorAction that uses the G4VPrimaryGenerator capabilities
    of the various CSource subclasses to `GeneratePrimaries(G4Event*)`
    which is invoked by Geant4 beamOn within CG4::propagate


OpStatus
     G4 status/enum code formatters and translation of G4 codes to Opticks flags 

Recorder
     Collects G4Step/G4StepPoint optical photon data  
     into NPY arrays in Opticks array format
     which are persisted to .npy  within an OpticksEvent

     *RecordStep* is called for all G4Step
     each of which is comprised of *pre* and *post* G4StepPoint, 
     as a result the same G4StepPoint are "seen" twice, 
     thus *RecordStep* only records the 1st of the pair 
     (the 2nd will come around as the first at the next call)
     except for the last G4Step pair where both points are recorded

Rec 
     Alternative implementation of Recorder using a vector of State instances

State 
     holds copy of G4Step together with G4OpBoundaryProcessStatus, 
     a vector of State instances is held by Rec

CRecorderLive
    DEAD CODE



Geant4 Plumbing
-----------------

CPrimaryGeneratorAction
     G4VUserPrimaryGeneratorAction subclass, "shell" class just 
     providing GeneratePrimaries(G4Event*)
     which passes through to CSource::GeneratePrimaryVertex

CSteppingAction
     G4UserSteppingAction subclass, which feeds G4Step to the recorders
     in method CSteppingAction::UserSteppingAction(G4Step*)

ActionInitialization
     G4VUserActionInitialization subclass, providing UserAction plumbing 
     for CPrimaryGeneratorAction and CSteppingAction

PhysicsList
     G4VModularPhysicsList subclass, follow chroma : registered just 

     G4OpticalPhysics() 
     G4EmPenelopePhysics(0) 


Geant4 Utilities
-------------------

Format
    G4 object string formatters for debugging 

CRayTracer
    interface to G4TheRayTracer, CPU ray tracer for G4 geometries


Others
--------

::

    CAction
    CBndLib
    CBoundaryProcess
    CCheck
    CDebug
    CEventAction
    CG4Ctx
    CGenerator
    CGunSource
    CInputPhotonSource
    CMaterialSort
    CMaterialTable
    CMath
    COptical
    COpticalSurface
    CPhoton
    CPhysics
    CPoi
    CPrimaryGeneratorAction
    CProcess
    CProcessManager
    CProcessSwitches

    CRec
    CRecState
    CRunAction
    CSource
    CStage
    CStep
    CStepRec
    CStepStatus
    CStepping
    CSteppingAction
    CSteppingState
    CStp
    CSurfaceLib
    CTorchSource
    CTrack
    CTrackingAction
    CTraverser
    CVec
    CVis
    CWriter

    Cerenkov
    Scintillation

    DebugG4Transportation
    OpNovicePhysicsList
    OpNovicePhysicsListMessenger
    OpRayleigh
    OpStatus
    PhysicsList
    State
    SteppingVerbose

