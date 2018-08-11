G4OK_SensitiveDetector
========================

::

    g4-cls G4VSensitiveDetector
    g4-cls G4SDManager
    g4-cls G4SDStructure
    g4-cls G4HCtable

    g4-cls G4VHitsCollection
         GetName, GetSDName

    g4-cls G4THitsCollection  
         class G4HitsCollection : public G4VHitsCollection
         template <class T> class G4THitsCollection : public G4HitsCollection


     38 // class description:
     39 //
     40 //  This is a class which stores hits collections generated at one event.
     41 // This class is exclusively constructed by G4SDManager when the first
     42 // hits collection of an event is passed to the manager, and this class
     43 // object is deleted by G4RunManager when a G4Event class object is deleted.
     44 //  Almost all public methods must be used by Geant4 kernel classes and
     45 // the user should not invoke them. The user can use two const methods,
     46 // GetHC() and GetNumberOfCollections() for accessing to the stored hits
     47 // collection(s).
     48 
     49 class G4HCofThisEvent

::

    epsilon: blyth$ g4-cc PrepareNewEvent
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/digits_hits/detector/src/G4SDManager.cc:G4HCofThisEvent* G4SDManager::PrepareNewEvent()
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/event/src/G4UserStackingAction.cc:void G4UserStackingAction::PrepareNewEvent()
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/event/src/G4AdjointStackingAction.cc:void G4AdjointStackingAction::PrepareNewEvent()
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/event/src/G4AdjointStackingAction.cc:  if ( !adjoint_mode && theFwdStackingAction)  theFwdStackingAction->PrepareNewEvent();
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/event/src/G4AdjointStackingAction.cc:  else if (adjoint_mode && theUserAdjointStackingAction)   theUserAdjointStackingAction->PrepareNewEvent();
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/event/src/G4EventManager.cc:  trackContainer->PrepareNewEvent();
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/event/src/G4EventManager.cc:  { currentEvent->SetHCofThisEvent(sdManager->PrepareNewEvent()); }
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/event/src/G4StackManager.cc:G4int G4StackManager::PrepareNewEvent()
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/event/src/G4StackManager.cc:  if(userStackingAction) userStackingAction->PrepareNewEvent();
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/event/src/G4StackManager.cc:              G4Exception("G4StackManager::PrepareNewEvent","Event0053",

::

    099 void G4EventManager::DoProcessing(G4Event* anEvent)
    100 {
    ...
    144 
    145   sdManager = G4SDManager::GetSDMpointerIfExist();
    146   if(sdManager)
    147   { currentEvent->SetHCofThisEvent(sdManager->PrepareNewEvent()); }
    148 
    149   if(userEventAction) userEventAction->BeginOfEventAction(currentEvent);
    150 

G4Event::

    185       inline G4HCofThisEvent* GetHCofThisEvent()  const
    186       { return HC; }


    105 G4HCofThisEvent* G4SDManager::PrepareNewEvent()
    106 {
    107   G4HCofThisEvent* HCE = new G4HCofThisEvent(HCtable->entries());
    108   treeTop->Initialize(HCE);
    109   return HCE;
    110 }



    offline-vi   # take look at JUNO SD

    offline-cls dywHit_PMT_Collection
    offline-cls PMTHitMerger


Formerly approach to stealing the hit collections was a bit overcomplicated, allowing adding hits in bulk::

    env-cls G4DAESensDet
    env-cls G4DAECollector
    env-cls DybG4DAECollector







