plugging-cfg4-leaks
======================

Context
----------

* :doc:`lifting-the-3M-photon-limitation`

Next
--------

* :doc:`large-vm-for-cuda-process`


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




Investigating the knee of the profile
-------------------------------------------

::

      11 :                                   OpticksHub::init :      0.621    245.596      0.117     11.284   

      12 :                                          _CG4::CG4 :      0.621    245.596      0.000      0.000   
      13 :                      _CRandomEngine::CRandomEngine :      0.621    245.596      0.000      0.000   
      14 :                                  _TCURAND::TCURAND :      0.621    245.596      0.000      0.000   
      15 :                            _TCURANDImp::TCURANDImp :      0.621    245.596      0.000      0.000   
      16 :                             TCURANDImp::TCURANDImp :      0.980   5685.544      0.359   5439.948   
      17 :                                   TCURAND::TCURAND :      0.980   5685.544      0.000      0.000   
      18 :                       CRandomEngine::CRandomEngine :      0.980   5685.544      0.000      0.000   
      19 :                                _CPhysics::CPhysics :      0.980   5685.544      0.000      0.000   
      20 :                                 CPhysics::CPhysics :      1.023   5687.456      0.043      1.912   
      21 :                                           CG4::CG4 :      1.035   5687.904      0.012      0.448   

      22 :                           _OpticksRun::createEvent :      2.461   9706.856      1.426   4018.953   
      23 :                            OpticksRun::createEvent :      2.461   9706.856      0.000      0.000   
      24 :                           _OKPropagator::propagate :      2.480   9706.856      0.020      0.000   
      25 :                                    _OEvent::upload :      2.504   9748.640      0.023     41.783   
      26 :                                     OEvent::upload :      2.508   9748.640      0.004      0.000   
      27 :                            _OPropagator::prelaunch :      2.516   9745.568      0.008     -3.071   
      28 :                             OPropagator::prelaunch :      3.773  10329.144      1.258    583.575   
      29 :                               _OPropagator::launch :      3.773  10329.144      0.000      0.000   
      30 :                                OPropagator::launch :      3.781  10558.520      0.008    229.376   
      31 :                          _OpIndexer::indexSequence :      3.781  10558.520      0.000      0.000   
      32 :                   _OpIndexer::indexSequenceInterop :      3.781  10558.520      0.000      0.000   
      33 :                       _OpIndexer::seqhisMakeLookup :      3.785  10558.520      0.004      0.000   
      34 :                        OpIndexer::seqhisMakeLookup :      3.793  10558.520      0.008      0.000   
      35 :                       OpIndexer::seqhisApplyLookup :      3.793  10558.520      0.000      0.000   



* The 5.4G from TCURAND is accounted for, thats just how CUDA does UVA (unified virtual addressing) :doc:`large-vm-for-cuda-process`


Pinnning down the 4G, mostly OKPropagator : confirmed to be mostly from OptiX context creation
--------------------------------------------------------------------------------------------------


::

    OpticksProfile=ERROR ts box --generateoverride 100000 



::

    19 :                                _CPhysics::CPhysics :      0.980   5685.544      0.000      0.000   
    20 :                                 CPhysics::CPhysics :      1.023   5687.456      0.043      1.912   
    21 :                                           CG4::CG4 :      1.035   5687.904      0.012      0.448   
    22 :                           _OpticksRun::createEvent :      2.461   9706.856      1.426   4018.953   
    23 :                            OpticksRun::createEvent :      2.461   9706.856      0.000      0.000   
    24 :                           _OKPropagator::propagate :      2.480   9706.856      0.020      0.000   



::

      15 :                                  _TCURAND::TCURAND :      0.618    245.596      0.000      0.000   
      16 :                            _TCURANDImp::TCURANDImp :      0.618    245.596      0.000      0.000   
      17 :                                          _dvec_dox :      0.618    245.596      0.000      0.000   
      18 :                                           dvec_dox :      1.141   5485.636      0.522   5240.040   
      19 :                                  _TRngBuf::TRngBuf :      1.142   5485.636      0.001      0.000   
      20 :                                   TRngBuf::TRngBuf :      1.142   5485.636      0.000      0.000   
      21 :                             TCURANDImp::TCURANDImp :      1.257   5685.640      0.115    200.004   
      22 :                                   TCURAND::TCURAND :      1.258   5685.640      0.001      0.000   
      23 :                       CRandomEngine::CRandomEngine :      1.258   5685.640      0.000      0.000   
      24 :                                _CPhysics::CPhysics :      1.258   5685.640      0.000      0.000   
      25 :                                 CPhysics::CPhysics :      1.306   5687.368      0.048      1.728   
      26 :                                           CG4::CG4 :      1.315   5687.904      0.010      0.536   
      27 :                            _OpticksViz::OpticksViz :      1.323   5689.224      0.008      1.320   
      28 :                             OpticksViz::OpticksViz :      1.327   5689.356      0.004      0.132   
      29 :                        _OKPropagator::OKPropagator :      1.644   5751.948      0.316     62.592   
      30 :                         OKPropagator::OKPropagator :      4.155   9706.349      2.512   3954.400   
      31 :                                   OKG4Mgr::OKG4Mgr :      4.155   9706.349      0.000      0.000   
      32 :                           _OpticksRun::createEvent :      4.155   9706.349      0.000      0.000   
      33 :                            OpticksRun::createEvent :      4.157   9706.349      0.002      0.000   
      34 :                           _OKPropagator::propagate :      4.177   9706.349      0.020      0.000   
      35 :                                    _OEvent::upload :      4.202   9748.137      0.025     41.788   

::

      24 :                                _CPhysics::CPhysics :      1.007   5685.580      0.000      0.000   
      25 :                                 CPhysics::CPhysics :      1.053   5687.456      0.046      1.876   
      26 :                                           CG4::CG4 :      1.062   5687.904      0.009      0.448   
      27 :                            _OpticksViz::OpticksViz :      1.069   5689.224      0.008      1.320   
      28 :                             OpticksViz::OpticksViz :      1.073   5689.356      0.004      0.132   
      29 :                        _OKPropagator::OKPropagator :      1.218   5751.948      0.145     62.592   
      30 :                                _OpEngine::OpEngine :      1.218   5751.948      0.000      0.000   
      31 :                                 OpEngine::OpEngine :      2.432   9675.900      1.214   3923.952   
      32 :                         OKPropagator::OKPropagator :      2.464   9706.345      0.032     30.444   
      33 :                                   OKG4Mgr::OKG4Mgr :      2.464   9706.345      0.000      0.000   
      34 :                           _OpticksRun::createEvent :      2.464   9706.345      0.000      0.000   
      35 :                            OpticksRun::createEvent :      2.465   9706.345      0.001      0.000   
      36 :                           _OKPropagator::propagate :      2.486   9706.345      0.021      0.000   



::

       21          0.113           1.021          0.113       5685.656        200.004 : TCURANDImp::TCURANDImp_0
       22          0.000           1.021          0.000       5685.656          0.000 : TCURAND::TCURAND_0
       23          0.000           1.021          0.000       5685.656          0.000 : CRandomEngine::CRandomEngine_0
       24          0.000           1.021          0.000       5685.656          0.000 : _CPhysics::CPhysics_0
       25          0.045           1.066          0.045       5687.372          1.716 : CPhysics::CPhysics_0
       26          0.010           1.076          0.010       5687.904          0.532 : CG4::CG4_0
       27          0.008           1.084          0.008       5689.224          1.320 : _OpticksViz::OpticksViz_0
       28          0.004           1.088          0.004       5689.356          0.132 : OpticksViz::OpticksViz_0
       29          0.154           1.242          0.154       5751.948         62.592 : _OKPropagator::OKPropagator_0
       30          0.000           1.242          0.000       5751.948          0.000 : _OpEngine::OpEngine_0
       31          0.000           1.242          0.000       5751.948          0.000 : _OScene::OScene_0
       32          0.000           1.242          0.000       5751.948          0.000 : _OContext::Create_0
       33          0.020           1.262          0.020       5811.740         59.792 : _optix::Context::create_0
       34          0.051           1.312          0.051       9384.692       3572.952 : optix::Context::create_0
       35          0.000           1.312          0.000       9384.692          0.000 : OContext::Create_0
       36          0.328           1.641          0.328       9394.137          9.444 : OScene::OScene_0
       37          0.820           2.461          0.820       9675.393        281.256 : OpEngine::OpEngine_0
       38          0.037           2.498          0.037       9706.860         31.468 : OKPropagator::OKPropagator_0
       39          0.000           2.498          0.000       9706.860          0.000 : OKG4Mgr::OKG4Mgr_0







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


