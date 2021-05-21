cfg4 : Comparing Opticks and Geant4 Simulations
==================================================

Package Infrastructure
-------------------------


::

    CFG4_API_EXPORT.hh
    CFG4_BODY.hh
    CFG4_HEAD.hh
    CFG4_LOG.hh
    CFG4_POP.hh
    CFG4_PUSH.hh
    CFG4_TAIL.hh



Low level Geant4 Utilities
----------------------------

::

    CVec.hh
        ctor: G4PhysicsOrderedFreeVector* vec

    CVis.hh
        static G4VisAttributes methods 

    Format.hh
        static formatters for G4Track, G4Step, ..

    CTrack.hh
        simple G4Track wrapper providing getTrackStatusString

    CDump.hh
       Dumps Geant4 materials and surfaces 


Random Control
-----------------

::

    CAlignEngine.hh
        predecessor of CRandomEngine.hh

    CRandomListener.hh
        pure virtual protocol base : preTrack, postTrack, postStep, postpropagate, flat_instrumented 

    CRandomEngine.hh
        isa CRandomListener, CLHEP::HepRandomEngine
        works with TCURAND

    CMixMaxRng.hh
        isa CRandomListener, CLHEP::MixMaxRng



Geant4 Physics and Process
-----------------------------

::

    OpNovicePhysicsList.hh
    OpNovicePhysicsListMessenger.hh
    PhysicsList.hh

    Cerenkov.hh
    G4Cerenkov1042.hh

    OpRayleigh.hh
        isa G4VDiscreteProcess : old Geant4 process 

    Scintillation.hh

    CCerenkov.hh
        trying to hide different Cerenkov process implementations inside this wrapper class

    CCerenkovGenerator.hh


    C4Cerenkov1042.hh
    CParticleDefinition.hh

    CPhysics.hh
    CPhysicsList.hh

    CProcess.hh
    CProcessManager.hh
    CProcessSwitches.hh



High level managers
-----------------------

::

    CDetector.hh
       isa G4VUserDetectorConstruction
       is the base class of *CGDMLDetector* and *CTestDetector*

    CTestDetector.hh
       isa CDetector
       constructs simple Geant4 detector test geometries based on commandline specifications

    CGDMLDetector.hh
       isa CDetector


    CG4.hh
       ctor OpticksHub
       holder of the actions and instrumentation classes like CRecorder and CStepRec 

    CGenerator.hh
       ctor from OpticksGen and CG4


Geant4 Actions
-----------------

::

    CPrimaryGeneratorAction.hh
       isa G4VUserPrimaryGeneratorAction
       ctor CSource 

    CRunAction.hh
       isa G4UserRunAction, ctor OpticksHub

    CEventAction.hh
       isa G4UserEventAction : BeginOfEventAction, EndOfEventAction
       ctor CG4

    CEventInfo.hh
       isa G4VUserEventInformation 
       holds gencode 

    CTrackingAction.hh
       isa G4UserTrackingAction


Collectors
--------------

::

    C4PhotonCollector.hh
    CPrimaryCollector.hh
    CPhotonCollector.hh
    CGenstepCollector.hh
       used by G4Opticks

    CHit.hh
       isa G4VHit


Sources
----------

::

    CSource.hh
       isa G4VPrimaryGenerator
       ctor Opticks
       holds CRecorder 

    CGenstepSource.hh
       isa CSource, ctor Opticks

    CGunSource.hh
       isa CSource, ctor Opticks

    CInputPhotonSource.hh
       isa CSource, ctor Opticks

    CPrimarySource.hh
       isa CSource, ctor Opticks

    CTorchSource.hh
       isa CSource, ctor Opticks, TorchStepNPY


Misc 
------------

::

    ActionInitialization.hh
    C4FPEDetection.hh

    CRayTracer.hh
       ctor CG4


GDML Utilities
----------------

::

    CGDML.hh
       higher level GDML Parse and Export 


    CGDMLKludge.hh
    CGDMLKludgeErrorHandler.hh
    CGDMLKludgeRead.hh
    CGDMLKludgeWrite.hh
       xerces-c level tools to fix broken GDML 


Geometry/Material/Surface Classes
--------------------------------------

::

    CGeometry.hh
       ctor OpticksHub, CSensitiveDetector
       hookup CG4

    CMath.hh
        just make_affineTransform

    CMaker.hh
        static methods creating G4VSolid from nnode and NCSG 

    CMPT.hh
        works with G4MaterialPropertiesTable 

    CMaterial.hh
    CMaterialBridge.hh
    CMaterialLib.hh
    CMaterialSort.hh
    CMaterialTable.hh

    CPropLib.hh

    CSensitiveDetector.hh
    CSkinSurfaceTable.hh

    CSolid.hh
       ctor G4VSolid, holder 

    COptical.hh
         Model Finish Type statics 

    COpticalSurface.hh


    CSurfaceBridge.hh
    CSurfaceLib.hh
    CSurfaceTable.hh

    CBndLib.hh
        ctor OpticksHub

    CBorderSurfaceTable.hh
    CBoundaryProcess.hh


    CCheck.hh
        Geant4 volume tree traversal checking G4LogicalBorderSurface     


Misc
-------

::

    CTrackInfo.hh
        isa G4VUserTrackInformation
        holding photon_record_id   
        HMM : appears not to be used

    CTraverser.hh
        ctor: Opticks* ok, G4VPhysicalVolume* top, NBoundingBox* bbox, OpticksQuery* query


    DebugG4Transportation.hh
        isa: G4Transportation 
        ctor: CG4* g4 

    SteppingVerbose.hh
       isa: G4SteppingVerbose

     

    OpStatus.hh



Recording Machinery Classes
------------------------------


::

    CRecorderLive.hh
       ctor CG4, CGeometry

    CPoi.hh
       holds G4StepPoint

    CStp.hh
       holds G4Step

    CRec.hh
       ctor CG4, CRecState
       holds vectors of CStp and CPoi

    CRecorder.hh
       ctor CG4, CGeometry
       holds CRec

       CRecorder::initEvent invoked by CG4::initEvent
       CRecorder::postTrackWritePoints 

    CG4Ctx.hh
       context struct shared between the CEventAction, CTrackingAction, CSteppingAction

       via calls: setEvent, setTrack, setStep this acts as interface between the Geant4 
       world and the recording into OpticksEvent 

    CAction.hh
       enum:  PRE_SAVE, POST_SAVE, ... HARD_TRUNCATE, TOPSLOT_REWRITE, POST_SKIP and labels 
       used by: CRecState CPhoton CRec CRecorder CStp

    CStage.hh
       enum: UNKNOWN, START, COLLECT, REJOIN, RECOLL and labels 
       used by: CSteppingAction CDebug CG4Ctx CPoi CRec CStp OpStatus State 

    CRecState.hh
       ctor: const CG4Ctx& ctx
       slot/truncation/topslot-rewrite (for reemission)

    CDebug.hh
       looks like it might be trying to debug recording machinery in an expensive way, 
       checking what CPhoton does cheaply 

    State.hh 
       holds G4Step, G4OpBoundaryProcessStatus, premat, postmat, CStage::CStage_t 
       used by: CPhoton.cc CRec.cc CRecState.cc CSteppingState.cc 

    CPhoton.hh
       ctor: const CG4Ctx& ctx, CRecState& state
       struct that builds seqhis and seqmat step by step

    CWriter.hh
       ctor CG4, CPhoton 
       canonical m_writer instance is resident of CRecorder and is instanciated with it

    CStep.hh
       ctor copies G4Step with step_id
       used by: CMaterialBridge.cc CPhoton.cc CRecorder.cc CStep.cc CStepRec.cc Format.cc 

    CStepRec.hh
       records non-optical particle step points into nopstep array, using vector of CStep pointers


    CSteppingAction.hh
       isa G4UserSteppingAction
       ctor CG4
       holds CRecorder, CStepRec, CMaterialBridge, CGeometry, CRandomEngine


    CSteppingState.hh
        struct fCurrentProcess, fPostStepGetPhysIntVector, MAXofPostStepLoops, fStepStatus 

    CStepping.hh
        interrogate G4SteppingManager providing CSteppingState 

    CStepStatus.hh
        G4StepStatus enum labels 





CRecorder::postTrackWriteSteps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CRecorder::postTrackWriteSteps is invoked from CRecorder::postTrack in --recstp mode (not --recpoi), 
once for the primary photon track and then for 0 or more reemtracks
via the record_id (which survives reemission) the info is written 
onto the correct place in the photon record buffer

The steps recorded into m_crec(CRec) are used to determine 
flags and the m_state(CRecState) is updated enabling 
appropriate step points are to be saved with WriteStepPoint.


CRecorder::WriteStepPoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* In --recpoi mode is invoked from CRecorder::postTrackWritePoints very simply.
* In --recstp mode is invoked several times from CRecorder::postTrackWriteSteps.

NB the last argumnent is only relevant to --recpoi mode



Class Details 
---------------------


.. toctree::

   CTraverser
   CDetector

   CGDMLDetector
   CTestDetector



