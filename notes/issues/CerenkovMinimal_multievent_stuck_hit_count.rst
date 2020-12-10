CerenkovMinimal_multievent_stuck_hit_count
============================================

Updating ckm to current G4Opticks API : run into SensorLib::checkSensorCategories assert
-------------------------------------------------------------------------------------------

::

    ckm
    ./go.sh

    epsilon:CerenkovMinimal blyth$ lldb_ CerenkovMinimal
    ...
    SensitiveDetector::EndOfEvent HCE 0x10fae88e0 hitCollectionA->entries() 4 hitCollectionB->entries() 0 A+B 4 m_hit_count 4

    ###[ EventAction::EndOfEventAction G4Opticks.propagateOpticalPhotons

    2020-12-10 10:51:01.994 FATAL [5389281] [OpPropagator::propagate@73] evtId(0) OK COMPUTE DEVELOPMENT
    2020-12-10 10:51:01.997 INFO  [5389281] [SensorLib::checkSensorCategories@398] [ SensorLib closed Y loaded N sensor_data 1,4 sensor_num 1 sensor_angular_efficiency N num_category 0
     sensorIndex      1 efficiency_1          0 efficiency_2          0 category      0 identifier          0 category_expected N
    Assertion failed: (category_expected), function checkSensorCategories, file /Users/blyth/opticks/optickscore/SensorLib.cc, line 425.
    ...
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #3: 0x00007fff528071ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000106d598a8 libOpticksCore.dylib`SensorLib::checkSensorCategories(this=0x000000010e6db740, dump=false) at SensorLib.cc:425
        frame #5: 0x0000000106d5900d libOpticksCore.dylib`SensorLib::close(this=0x000000010e6db740) at SensorLib.cc:367
        frame #6: 0x00000001063c3735 libOKOP.dylib`OpEngine::close(this=0x000000010e6e2b50) at OpEngine.cc:173
        frame #7: 0x00000001063c3bfa libOKOP.dylib`OpEngine::propagate(this=0x000000010e6e2b50) at OpEngine.cc:200
        frame #8: 0x00000001063c5f12 libOKOP.dylib`OpPropagator::propagate(this=0x000000010e6e2b10) at OpPropagator.cc:77
        frame #9: 0x00000001063c4c73 libOKOP.dylib`OpMgr::propagate(this=0x000000010e6ddcc0) at OpMgr.cc:136
        frame #10: 0x000000010011b21f libG4OK.dylib`G4Opticks::propagateOpticalPhotons(this=0x000000010fa03000, eventID=0) at G4Opticks.cc:915
        frame #11: 0x000000010002ad1b CerenkovMinimal`EventAction::EndOfEventAction(this=0x000000010f89f510, event=0x000000010faf5060) at EventAction.cc:60
        frame #12: 0x000000010203bfd7 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000010f8690d0, anEvent=0x000000010faf5060) at G4EventManager.cc:265
        frame #13: 0x000000010203cc2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x000000010f8690d0, anEvent=0x000000010faf5060) at G4EventManager.cc:338
        frame #14: 0x0000000101f489f5 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010e75d670, i_event=0) at G4RunManager.cc:399
        frame #15: 0x0000000101f48825 libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010e75d670, n_event=3, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
        frame #16: 0x0000000101f46ce1 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010e75d670, n_event=3, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #17: 0x000000010002f48d CerenkovMinimal`G4::beamOn(this=0x00007ffeefbfe568, nev=3) at G4.cc:81
        frame #18: 0x000000010002f337 CerenkovMinimal`G4::G4(this=0x00007ffeefbfe568, nev=3) at G4.cc:69
        frame #19: 0x000000010002f4bb CerenkovMinimal`G4::G4(this=0x00007ffeefbfe568, nev=3) at G4.cc:51
        frame #20: 0x000000010000f9d7 CerenkovMinimal`main(argc=1, argv=0x00007ffeefbfe628) at CerenkovMinimal.cc:26
        frame #21: 0x00007fff52793015 libdyld.dylib`start + 1
        frame #22: 0x00007fff52793015 libdyld.dylib`start + 1
    (lldb) 

Avoid this issue by adding calls to setSensorData::

     36 void RunAction::BeginOfRunAction(const G4Run*)
     37 {
     38 #ifdef WITH_OPTICKS
     39     G4cout << "\n\n###[ RunAction::BeginOfRunAction G4Opticks.setGeometry\n\n" << G4endl ;
     40     G4VPhysicalVolume* world = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ;
     41     assert( world ) ;
     42     bool standardize_geant4_materials = false ;   // required for alignment 
     43 
     44     G4Opticks* g4ok = G4Opticks::Get();
     45     g4ok->setGeometry(world, standardize_geant4_materials );
     46 
     47     const std::vector<G4PVPlacement*>& sensor_placements = g4ok->getSensorPlacements() ;
     48     assert( sensor_placements.size() == 1 ); 
     49     for(unsigned i=0 ; i < sensor_placements.size()  ; i++)
     50     {
     51         float efficiency_1 = 0.5f ;
     52         float efficiency_2 = 1.0f ;
     53         int sensor_cat = -1 ;   // -1:means no angular info 
     54         int sensor_identifier = 0xc0ffee + i ;  // mockup a detector specific identifier
     55         unsigned sensorIndex = 1+i ;  // 1-based
     56         g4ok->setSensorData( sensorIndex, efficiency_1, efficiency_2, sensor_cat, sensor_identifier );
     57     }   
     58     
     59     G4cout << "\n\n###] RunAction::BeginOfRunAction G4Opticks.setGeometry\n\n" << G4endl ;
     60 #endif
     61 }




Issue : running CerenkovMinimal with 3 events yields same number of hits for each event
--------------------------------------------------------------------------------------------

::

    CerenkovMinimal 


Check the saved event arrays. Observe all gensteps are the same::

    epsilon:natural blyth$ cd /tmp/blyth/opticks/source/evt/g4live/natural/
    epsilon:natural blyth$ np.py ?/gs.npy
    a :                                                     1/gs.npy :            (1, 6, 4) : e4c31843a4e3e124f0d2d90b311a4236 : 20201210-1111 
    b :                                                     2/gs.npy :            (1, 6, 4) : e4c31843a4e3e124f0d2d90b311a4236 : 20201210-1111 
    c :                                                     3/gs.npy :            (1, 6, 4) : e4c31843a4e3e124f0d2d90b311a4236 : 20201210-1111 


Adding the "g4ok->reset()" after hit dumping is necessary to clear the genstep collectors::

     58     G4Opticks* g4ok = G4Opticks::Get() ;
     59     G4int eventID = event->GetEventID() ;
     60     int num_hits = g4ok->propagateOpticalPhotons(eventID) ;
     61 
     62     G4cout
     63            << "EventAction::EndOfEventAction"
     64            << " eventID " << eventID
     65            << " num_hits " << num_hits
     66            << G4endl
     67            ;
     68 
     69     G4OpticksHit hit ;
     70     for(unsigned i=0 ; i < num_hits ; i++)
     71     {
     72         g4ok->getHit(i, &hit);
     73         std::cout
     74             << std::setw(5) << i
     75             << " boundary "           << std::setw(4) << hit.boundary
     76             << " sensorIndex "        << std::setw(5) << hit.sensorIndex
     77             << " nodeIndex "          << std::setw(5) << hit.nodeIndex
     78             << " photonIndex "        << std::setw(5) << hit.photonIndex
     79             << " flag_mask    "       << std::setw(10) << std::hex << hit.flag_mask  << std::dec
     80             << " sensor_identifier "  << std::setw(10) << std::hex << hit.sensor_identifier << std::dec
     81             << " wavelength "         << std::setw(8) << hit.wavelength
     82             << " time "               << std::setw(8) << hit.time
     83             << " global_position "    << hit.global_position
     84             << " " << OpticksFlags::FlagMask(hit.flag_mask, true)
     85             << std::endl
     86             ;
     87     }
     88 
     89     g4ok->reset();  // necessary to prevent gensteps keeping to accumulate



After that see that have different gensteps and hits for each event::

    epsilon:natural blyth$ cd /tmp/blyth/opticks/source/evt/g4live/natural/
    epsilon:natural blyth$ np.py ?/gs.npy
    a :                                                     1/gs.npy :            (1, 6, 4) : e4c31843a4e3e124f0d2d90b311a4236 : 20201210-1136 
    b :                                                     2/gs.npy :            (1, 6, 4) : 437a1e2718d56e9b5b547bcb2897b20d : 20201210-1136 
    c :                                                     3/gs.npy :            (1, 6, 4) : 0cf20c68b490f414d697bab203e8d650 : 20201210-1136 

    epsilon:natural blyth$ np.py ?/ht.npy
    a :                                                     2/ht.npy :           (12, 4, 4) : 905d816fc87d39c1e5a80a4c9bccd315 : 20201210-1136 
    b :                                                     3/ht.npy :            (8, 4, 4) : 815c302629bc6c383f0d1660d51e06aa : 20201210-1136 
    c :                                                     1/ht.npy :            (2, 4, 4) : e519c24ed65aded3a4a6719d1b169a58 : 20201210-1136 


And logging also shows the varying numbers of hits for each event::

    2020-12-10 11:36:41.302 INFO  [5428748] [OpIndexer::indexSequenceCompute@237] OpIndexer::indexSequenceCompute
    2020-12-10 11:36:41.306 INFO  [5428748] [OEvent::downloadHits@443]  nhit 2 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    2020-12-10 11:36:41.306 FATAL [5428748] [OpPropagator::propagate@84] evtId(0) DONE nhit: 2
    2020-12-10 11:36:41.306 INFO  [5428748] [OpticksEvent::save@1705] /tmp/blyth/opticks/source/evt/g4live/natural/-1
    2020-12-10 11:36:41.320 INFO  [5428748] [OpticksEvent::save@1705] /tmp/blyth/opticks/source/evt/g4live/natural/1
    EventAction::EndOfEventAction eventID 0 num_hits 2
        0 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex     1 flag_mask          8041 sensor_identifier     c0ffee wavelength  64.7406 time  1.52996 global_position (303.048,116.92,90) CK|SD|EC
        1 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex     8 flag_mask          8041 sensor_identifier     c0ffee wavelength  117.882 time 0.809772 global_position (153.107,16.8344,90) CK|SD|EC

    ###] EventAction::EndOfEventAction G4Opticks.propagateOpticalPhotons

    EventAction::EndOfEventAction DumpHitCollections 
    SensitiveDetector::DumpHitCollections query SD0/OpHitCollectionA hcid    0 hc 0x7fd0a996c800 hc.entries 4
    SensitiveDetector::DumpHitCollections query SD0/OpHitCollectionB hcid    1 hc 0x7fd0a996c848 hc.entries 0
    SensitiveDetector::Initialize HCE 0x7fd0a9904640 HCE.Capacity 2 SensitiveDetectorName SD0 collectionName[0] OpHitCollectionA collectionName[1] OpHitCollectionB
    Ctx::setTrack _track_particle_name e+ _track_id 0 _step_id -1 num_gs 0 max_gs 1 kill 0
    L4Cerenkov::PostStepDoIt DONE -- NumberOfSecondaries = 50
    L4Cerenkov::PostStepDoIt G4Opticks.collectSecondaryPhotons
    2020-12-10 11:36:41.341 INFO  [5428748] [C4PhotonCollector::collectSecondaryPhotons@100]  numberOfSecondaries 50
    Ctx::setStep _step_id 0 num_gs 1
    Ctx::postTrack _track_particle_name : e+
    Ctx::setTrack _track_particle_name e+ _track_id 0 _step_id -1 num_gs 1 max_gs 1 kill 1
    Ctx::postTrack _track_particle_name : e+
    SensitiveDetector::EndOfEvent HCE 0x7fd0a9904640 hitCollectionA->entries() 18 hitCollectionB->entries() 0 A+B 18 m_hit_count 22

    ###[ EventAction::EndOfEventAction G4Opticks.propagateOpticalPhotons

    2020-12-10 11:36:41.344 FATAL [5428748] [OpPropagator::propagate@73] evtId(2) OK COMPUTE DEVELOPMENT
    2020-12-10 11:36:41.344 INFO  [5428748] [OpSeeder::seedPhotonsFromGenstepsViaOptiX@174] SEEDING TO SEED BUF  
    2020-12-10 11:36:41.345 INFO  [5428748] [OPropagator::launch@266] LAUNCH NOW  --printenabled  printLaunchIndex ( 0 0 0) -
    2020-12-10 11:36:41.346 INFO  [5428748] [OPropagator::launch@275] LAUNCH DONE
    2020-12-10 11:36:41.346 INFO  [5428748] [OPropagator::launch@277] 0 : (0;50,1) 
    2020-12-10 11:36:41.346 INFO  [5428748] [BTimes::dump@177] OPropagator::launch
                    launch002                 0.000993
    2020-12-10 11:36:41.346 INFO  [5428748] [OpIndexer::indexSequenceCompute@237] OpIndexer::indexSequenceCompute
    2020-12-10 11:36:41.350 INFO  [5428748] [OEvent::downloadHits@443]  nhit 12 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    2020-12-10 11:36:41.350 FATAL [5428748] [OpPropagator::propagate@84] evtId(2) DONE nhit: 12
    2020-12-10 11:36:41.350 INFO  [5428748] [OpticksEvent::save@1705] /tmp/blyth/opticks/source/evt/g4live/natural/-2
    2020-12-10 11:36:41.366 INFO  [5428748] [OpticksEvent::save@1705] /tmp/blyth/opticks/source/evt/g4live/natural/2
    EventAction::EndOfEventAction eventID 1 num_hits 12
        0 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex     1 flag_mask          8041 sensor_identifier     c0ffee wavelength  64.7406 time   1.4689 global_position (268.94,155.589,90) CK|SD|EC
        1 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex     8 flag_mask          8041 sensor_identifier     c0ffee wavelength  117.882 time 0.819729 global_position (150.124,43.8023,90) CK|SD|EC
        2 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    15 flag_mask          4041 sensor_identifier     c0ffee wavelength  174.511 time  1.95018 global_position (357.073,220.873,90) CK|SD|EX
        3 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    19 flag_mask          8041 sensor_identifier     c0ffee wavelength  123.517 time 0.741574 global_position (135.883,-2.04602,90) CK|SD|EC
        4 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    20 flag_mask          8041 sensor_identifier     c0ffee wavelength  225.608 time 0.744174 global_position (136.305,11.7514,90) CK|SD|EC
        5 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    21 flag_mask          8041 sensor_identifier     c0ffee wavelength  93.8313 time  1.23818 global_position (226.747,121.511,90) CK|SD|EC
        6 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    24 flag_mask          8041 sensor_identifier     c0ffee wavelength  206.685 time  1.07057 global_position (196.067,94.721,90) CK|SD|EC
        7 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    26 flag_mask          4041 sensor_identifier     c0ffee wavelength  89.2701 time 0.735311 global_position (134.608,3.74598,90) CK|SD|EX
        8 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    42 flag_mask          4041 sensor_identifier     c0ffee wavelength  221.518 time 0.858482 global_position (157.151,54.4581,90) CK|SD|EX
        9 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    44 flag_mask          8041 sensor_identifier     c0ffee wavelength  579.553 time  1.24398 global_position (228.305,-115.233,90) CK|SD|EC
       10 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    47 flag_mask          8041 sensor_identifier     c0ffee wavelength  149.333 time   1.7122 global_position (313.455,189.378,90) CK|SD|EC
       11 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    49 flag_mask          8041 sensor_identifier     c0ffee wavelength  60.8645 time  1.04946 global_position (192.119,91.7998,90) CK|SD|EC

    ###] EventAction::EndOfEventAction G4Opticks.propagateOpticalPhotons

    EventAction::EndOfEventAction DumpHitCollections 
    SensitiveDetector::DumpHitCollections query SD0/OpHitCollectionA hcid    0 hc 0x7fd0a996c848 hc.entries 18
    SensitiveDetector::DumpHitCollections query SD0/OpHitCollectionB hcid    1 hc 0x7fd0a996c800 hc.entries 0
    SensitiveDetector::Initialize HCE 0x7fd0a9904640 HCE.Capacity 2 SensitiveDetectorName SD0 collectionName[0] OpHitCollectionA collectionName[1] OpHitCollectionB
    Ctx::setTrack _track_particle_name e+ _track_id 0 _step_id -1 num_gs 0 max_gs 1 kill 0
    L4Cerenkov::PostStepDoIt DONE -- NumberOfSecondaries = 43
    L4Cerenkov::PostStepDoIt G4Opticks.collectSecondaryPhotons
    2020-12-10 11:36:41.394 INFO  [5428748] [C4PhotonCollector::collectSecondaryPhotons@100]  numberOfSecondaries 43
    Ctx::setStep _step_id 0 num_gs 1
    Ctx::postTrack _track_particle_name : e+
    Ctx::setTrack _track_particle_name e+ _track_id 0 _step_id -1 num_gs 1 max_gs 1 kill 1
    Ctx::postTrack _track_particle_name : e+
    SensitiveDetector::EndOfEvent HCE 0x7fd0a9904640 hitCollectionA->entries() 13 hitCollectionB->entries() 0 A+B 13 m_hit_count 35

    ###[ EventAction::EndOfEventAction G4Opticks.propagateOpticalPhotons

    2020-12-10 11:36:41.397 FATAL [5428748] [OpPropagator::propagate@73] evtId(4) OK COMPUTE DEVELOPMENT
    2020-12-10 11:36:41.398 INFO  [5428748] [OpSeeder::seedPhotonsFromGenstepsViaOptiX@174] SEEDING TO SEED BUF  
    2020-12-10 11:36:41.398 INFO  [5428748] [OPropagator::launch@266] LAUNCH NOW  --printenabled  printLaunchIndex ( 0 0 0) -
    2020-12-10 11:36:41.399 INFO  [5428748] [OPropagator::launch@275] LAUNCH DONE
    2020-12-10 11:36:41.399 INFO  [5428748] [OPropagator::launch@277] 0 : (0;43,1) 
    2020-12-10 11:36:41.399 INFO  [5428748] [BTimes::dump@177] OPropagator::launch
                    launch003                 0.000951
    2020-12-10 11:36:41.399 INFO  [5428748] [OpIndexer::indexSequenceCompute@237] OpIndexer::indexSequenceCompute
    2020-12-10 11:36:41.402 INFO  [5428748] [OEvent::downloadHits@443]  nhit 8 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    2020-12-10 11:36:41.402 FATAL [5428748] [OpPropagator::propagate@84] evtId(4) DONE nhit: 8
    2020-12-10 11:36:41.402 INFO  [5428748] [OpticksEvent::save@1705] /tmp/blyth/opticks/source/evt/g4live/natural/-3
    2020-12-10 11:36:41.423 INFO  [5428748] [OpticksEvent::save@1705] /tmp/blyth/opticks/source/evt/g4live/natural/3
    EventAction::EndOfEventAction eventID 2 num_hits 8
        0 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex     1 flag_mask          8041 sensor_identifier     c0ffee wavelength  64.7406 time  1.45614 global_position (265.549,155.589,90) CK|SD|EC
        1 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex     8 flag_mask          8041 sensor_identifier     c0ffee wavelength  117.882 time 0.812617 global_position (148.236,43.8023,90) CK|SD|EC
        2 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    19 flag_mask          8041 sensor_identifier     c0ffee wavelength  123.517 time 0.735156 global_position (134.185,-2.04602,90) CK|SD|EC
        3 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    20 flag_mask          8041 sensor_identifier     c0ffee wavelength  225.608 time 0.737722 global_position (134.593,11.7514,90) CK|SD|EC
        4 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    21 flag_mask          8041 sensor_identifier     c0ffee wavelength  93.8313 time  1.22744 global_position (223.895,121.511,90) CK|SD|EC
        5 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    24 flag_mask          8041 sensor_identifier     c0ffee wavelength  206.685 time  1.06128 global_position (193.603,94.721,90) CK|SD|EC
        6 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    26 flag_mask          4041 sensor_identifier     c0ffee wavelength  89.2701 time 0.728915 global_position (132.909,3.74598,90) CK|SD|EX
        7 boundary   -3 sensorIndex     1 nodeIndex     2 photonIndex    42 flag_mask          4041 sensor_identifier     c0ffee wavelength  221.518 time 0.851013 global_position (155.167,54.4581,90) CK|SD|EX

    ###] EventAction::EndOfEventAction G4Opticks.propagateOpticalPhotons

    EventAction::EndOfEventAction DumpHitCollections 
    SensitiveDetector::DumpHitCollections query SD0/OpHitCollectionA hcid    0 hc 0x7fd0a996c800 hc.entries 13
    SensitiveDetector::DumpHitCollections query SD0/OpHitCollectionB hcid    1 hc 0x7fd0a996c848 hc.entries 0


    ###[ RunAction::EndOfRunAction G4Opticks.Finalize







