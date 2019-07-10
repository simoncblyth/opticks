plugging-cfg4-leaks
======================

Context
----------

* :doc:`lifting-the-3M-photon-limitation`


ISSUE : Leaking like a sieve 
---------------------------------


4M running : OKG4Test  profile time and memory usage, looks real leaky, DYNAMIC_CURAND doesnt bend over like 
-------------------------------------------------------------------------------------------------------------------------------------------

::

    OpticksProfile=ERROR ts box          ## simple showing stamps

    ip tprofile.py                       ## plotting the time vs memory profile 


    TBOOLEAN_TAG=100 ts box --generateoverride 4000000 --rngmax 10
    # use non-default tag, to prevent accidental stomping 

    TBOOLEAN_TAG=200 ts box --generateoverride 2000000 --rngmax 3

::

    .  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND    
    232213 blyth     20   0   55.4g  43.8g 195952 R  99.3 70.0   9:26.92 OKG4Test      

    232213 blyth     20   0   55.6g  44.0g 195952 R 100.0 70.4   9:29.95 OKG4Test            # during python ana


* HMM : but the Opticks.npy with profile info is placed above tag, TODO Change this 



105M prior to each::

          1.254         534.527          1.254      55126.379        104.449 : _CInputPhotonSource::GeneratePrimaryVertex_0
          0.012         534.539          0.012      55126.379          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
          1.281         535.820          1.281      55231.848        105.469 : _CInputPhotonSource::GeneratePrimaryVertex_0
          0.012         535.832          0.012      55231.848          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
          1.273         537.105          1.273      55336.297        104.449 : _CInputPhotonSource::GeneratePrimaryVertex_0
          0.012         537.117          0.012      55336.297          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
          1.242         538.359          1.242      55441.770        105.473 : _CInputPhotonSource::GeneratePrimaryVertex_0
          0.012         538.371          0.012      55441.770          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0


::

    OpticksProfile=ERROR ts box --generateoverride 100000 



ip tprofile::

     263 :                         CG4Ctx::setTrackOptical_1k :     23.537  12395.760      0.115     10.240   
     264 :                         CG4Ctx::setTrackOptical_1k :     23.648  12406.000      0.111     10.240   
     265 :                         CG4Ctx::setTrackOptical_1k :     23.762  12417.264      0.113     11.264   
     266 :                         CG4Ctx::setTrackOptical_1k :     23.877  12427.504      0.115     10.240   
     267 :                     CEventAction::EndOfEventAction :     23.877  12427.504      0.000      0.000   
     268 :         _CInputPhotonSource::GeneratePrimaryVertex :     23.877  12427.504      0.000      0.000   
     269 :          CInputPhotonSource::GeneratePrimaryVertex :     23.889  12427.504      0.012      0.000   
     270 :                   CEventAction::BeginOfEventAction :     23.889  12427.504      0.000      0.000   
     271 :                                   CG4Ctx::setEvent :     23.889  12427.504      0.000      0.000   
     272 :                         CG4Ctx::setTrackOptical_1k :     24.016  12437.743      0.127     10.239   
     273 :                         CG4Ctx::setTrackOptical_1k :     24.135  12447.983      0.119     10.240   
     274 :                         CG4Ctx::setTrackOptical_1k :     24.248  12459.247      0.113     11.264   
     275 :                         CG4Ctx::setTrackOptical_1k :     24.359  12469.487      0.111     10.240   
     276 :                         CG4Ctx::setTrackOptical_1k :     24.475  12479.728      0.115     10.240   
     277 :                         CG4Ctx::setTrackOptical_1k :     24.588  12489.968      0.113     10.240   
     278 :                         CG4Ctx::setTrackOptical_1k :     24.701  12500.208      0.113     10.240   
     279 :                         CG4Ctx::setTrackOptical_1k :     24.814  12511.472      0.113     11.264   
     280 :                         CG4Ctx::setTrackOptical_1k :     24.930  12521.712      0.115     10.240   
     281 :                         CG4Ctx::setTrackOptical_1k :     25.055  12531.951      0.125     10.239   
     282 :                     CEventAction::EndOfEventAction :     25.055  12531.951      0.000      0.000   
     283 :         _CInputPhotonSource::GeneratePrimaryVertex :     25.055  12531.951      0.000      0.000   
     284 :          CInputPhotonSource::GeneratePrimaryVertex :     25.066  12531.951      0.012      0.000   
     285 :                   CEventAction::BeginOfEventAction :     25.066  12531.951      0.000      0.000   
     286 :                                   CG4Ctx::setEvent :     25.066  12531.951      0.000      0.000   
     287 :                         CG4Ctx::setTrackOptical_1k :     25.188  12542.191      0.121     10.240   


* for each 1000 photons VM increases by 10M bytes, 100 photons 1M , 1 photon 0.010M  10,000 bytes  1250 doubles ??? (unreasonable) 

* are using static recording so step points are allocated ahead ?


* https://indico.fnal.gov/event/9717/session/3/contribution/60/material/slides/0.pdf




After plugging the CStp CPoi leak in CRec::clear
---------------------------------------------------

::

      233          0.469          38.770          0.469      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      234          0.516          39.285          0.516      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      235          0.453          39.738          0.453      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      236          0.453          40.191          0.453      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      237          0.461          40.652          0.461      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      238          0.465          41.117          0.465      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      239          0.512          41.629          0.512      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      240          0.488          42.117          0.488      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      241          0.477          42.594          0.477      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      243          0.004          42.598          0.004      11146.132          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
      246          0.004          42.602          0.004      11146.132          0.000 : CSource::collectPrimaryVertex_1k_0
      249          0.004          42.605          0.004      11146.132          0.000 : CSource::collectPrimaryVertex_1k_0
      253          0.004          42.609          0.004      11146.132          0.000 : CSource::collectPrimaryVertex_1k_0
      257          0.469          43.078          0.469      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      258          0.453          43.531          0.453      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      259          0.441          43.973          0.441      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      260          0.457          44.430          0.457      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      261          0.449          44.879          0.449      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      262          0.469          45.348          0.469      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      263          0.484          45.832          0.484      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      264          0.438          46.270          0.438      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      265          0.449          46.719          0.449      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      266          0.465          47.184          0.465      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      267          0.004          47.188          0.004      11146.132          0.000 : CEventAction::EndOfEventAction_0
      272          0.004          47.191          0.004      11146.132          0.000 : CSource::collectPrimaryVertex_1k_0
      275          0.004          47.195          0.004      11146.132          0.000 : CSource::collectPrimaryVertex_1k_0
      278          0.004          47.199          0.004      11146.132          0.000 : CSource::collectPrimaryVertex_1k_0
      282          0.480          47.680          0.480      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      283          0.441          48.121          0.441      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      284          0.449          48.570          0.449      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      285          0.461          49.031          0.461      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      286          0.473          49.504          0.473      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0
      287          0.441          49.945          0.441      11146.132          0.000 : CG4Ctx::setTrackOptical_1k_0



Investigate G4Event cleanup
------------------------------

::

    g4-;g4-cls G4RunManager  


Hmm maybe could repeatedly create and delete run managers ? Thats means starting 
from scratch ?

Perhaps arranging to have more runs (a run for every event) will clean more often  ?


::

     15 CPhysics::CPhysics(CG4* g4)  
     16     :
     17     m_g4(g4),
     18     m_hub(g4->getHub()),
     19     m_ok(g4->getOpticks()),
     20     m_runManager(new G4RunManager),
     21 #ifdef OLDPHYS
     22     m_physicslist(new PhysicsList())
     23 #else
     24     m_physicslist(new CPhysicsList(m_g4))
     25     //m_physicslist(new OpNovicePhysicsList(m_g4))
     26 #endif
     27 {   
     28     init();
     29 }   



::

    477 void G4RunManager::RunTermination()
    478 {
    479   if(!fakeRun)
    480   {
    481     CleanUpUnnecessaryEvents(0);
    482     if(userRunAction) userRunAction->EndOfRunAction(currentRun);
    483     G4VPersistencyManager* fPersM = G4VPersistencyManager::GetPersistencyManager();
    484     if(fPersM) fPersM->Store(currentRun);
    485     runIDCounter++;
    486   }
    487 
    488   kernel->RunTermination();
    489 }


    510 void G4RunManager::CleanUpUnnecessaryEvents(G4int keepNEvents)
    511 {
    512   // Delete events that are no longer necessary for post
    513   // processing such as visualization.
    514   // N.B. If ToBeKept() is true, the pointer of this event is
    515   // kept in G4Run of the previous run, and deleted along with
    516   // the deletion of G4Run.
    517 
    518   std::list<G4Event*>::iterator evItr = previousEvents->begin();
    519   while(evItr!=previousEvents->end())
    520   {
    521     if(G4int(previousEvents->size()) <= keepNEvents) return;
    522 
    523     G4Event* evt = *evItr;
    524     if(evt)
    525     {
    526       if(evt->GetNumberOfGrips()==0)
    527       {
    528         if(!(evt->ToBeKept())) delete evt;
    529         evItr = previousEvents->erase(evItr);
    530       }
    531       else
    532       { evItr++; }
    533     }
    534     else
    535     { evItr = previousEvents->erase(evItr); }
    536   }
    537 }
    538 


::

    360 void G4RunManager::DoEventLoop(G4int n_event,const char* macroFile,G4int n_select)
    361 {
    362   InitializeEventLoop(n_event,macroFile,n_select);
    363 
    364 // Event loop
    365   for(G4int i_event=0; i_event<n_event; i_event++ )
    366   {
    367     ProcessOneEvent(i_event);
    368     TerminateOneEvent();
    369     if(runAborted) break;
    370   }
    371 
    372   // For G4MTRunManager, TerminateEventLoop() is invoked after all threads are finished.
    373   if(runManagerType==sequentialRM) TerminateEventLoop();
    374 }

    396 void G4RunManager::ProcessOneEvent(G4int i_event)
    397 {
    398   currentEvent = GenerateEvent(i_event);
    399   eventManager->ProcessOneEvent(currentEvent);
    400   AnalyzeEvent(currentEvent);
    401   UpdateScoring();
    402   if(i_event<n_select_msg) G4UImanager::GetUIpointer()->ApplyCommand(msgText);
    403 }
    404 
    405 void G4RunManager::TerminateOneEvent()
    406 {
    407   StackPreviousEvent(currentEvent);
    408   currentEvent = 0;
    409   numberOfEventProcessed++;
    410 }


Hmm, looks like events are being deleted anyhow::

    539 void G4RunManager::StackPreviousEvent(G4Event* anEvent)
    540 {
    541   if(anEvent->ToBeKept()) currentRun->StoreEvent(anEvent);
    542 
    543   if(n_perviousEventsToBeStored==0)
    544   {
    545     if(anEvent->GetNumberOfGrips()==0)
    546     { if(!(anEvent->ToBeKept())) delete anEvent; }
    547     else
    548     { previousEvents->push_back(anEvent); }
    549   }
    550 
    551   CleanUpUnnecessaryEvents(n_perviousEventsToBeStored);
    552 }


::

    OpticksProfile=ERROR ts box --generateoverride 100000 --cg4sigint  -D

    (gdb) b 'G4Event::~G4Event()' 

    (gdb) bt
    #0  G4Event::~G4Event (this=0xe1fcf90, __in_chrg=<optimized out>) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4Event.cc:66
    #1  0x00007ffff15640e4 in G4RunManager::StackPreviousEvent (this=0x7a687e0, anEvent=0xe1fcf90) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:546
    #2  0x00007ffff15636f0 in G4RunManager::TerminateOneEvent (this=0x7a687e0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:407
    #3  0x00007ffff15634ee in G4RunManager::DoEventLoop (this=0x7a687e0, n_event=10, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:368
    #4  0x00007ffff1562d2d in G4RunManager::BeamOn (this=0x7a687e0, n_event=10, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:273
    #5  0x00007ffff4ca305a in CG4::propagate (this=0x67f6c90) at /home/blyth/opticks/cfg4/CG4.cc:348
    #6  0x00007ffff7bd570f in OKG4Mgr::propagate_ (this=0x7fffffffcbc0) at /home/blyth/opticks/okg4/OKG4Mgr.cc:177
    #7  0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffcbc0) at /home/blyth/opticks/okg4/OKG4Mgr.cc:117
    #8  0x00000000004039a9 in main (argc=35, argv=0x7fffffffcef8) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) 



Am missing some profiling machinery that accumulates deltaT and deltaVM in a slice of code
across all calls, eg CRecorder::postTrack 

* https://igprof.org/


