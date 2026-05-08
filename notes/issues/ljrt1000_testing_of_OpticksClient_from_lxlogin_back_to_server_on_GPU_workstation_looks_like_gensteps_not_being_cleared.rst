ljrt1000_testing_of_OpticksClient_from_lxlogin_back_to_server_on_GPU_workstation_looks_like_gensteps_not_being_cleared
=========================================================================================================================


Setup
-------

1. start Opticks server on GPU workstation::

    (ok) [lo] A[blyth@localhost ~]$ ~/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/CSGOptiXService_FastAPI_test.sh

2. open reverse ssh tunnel from GPU workstation to lxlogin::

    A> ssh LT

3. on lxlogin check ~/.opticks/GEOM/ENVSET.sh points to an OJ build against OpticksClient then run::

    ljrt1000



Issue 1 : server qcerenkov logging too verbose : bad matline ? or just ever accumulating gensteps ?
-----------------------------------------------------------------------------------------------------

::

    //qcerenkov::wavelength_sampled_bndtex idx 9075124 sampledRI   1.263 cosTheta   1.004 sin2Theta   0.000 wavelength  99.582 count 100 gs.matline 39 gs.pos (-151.039, -1.577,-41.422)
    //qcerenkov::wavelength_sampled_bndtex idx 9075125 sampledRI   1.263 cosTheta   1.004 sin2Theta   0.000 wavelength  83.345 count 100 gs.matline 39 gs.pos (-151.039, -1.577,-41.422)
    //qcerenkov::wavelength_sampled_bndtex idx 9075126 sampledRI   1.263 cosTheta   1.004 sin2Theta   0.000 wavelength  90.974 count 100 gs.matline 39 gs.pos (-151.039, -1.577,-41.422)
    //qcerenkov::wavelength_sampled_bndtex idx 9075127 sampledRI   1.263 cosTheta   1.004 sin2Theta   0.000 wavelength 194.656 count 100 gs.matline 39 gs.pos (-151.039, -1.577,-41.422)
    //qcerenkov::wavelength_sampled_bndtex idx 9075128 sampledRI   1.263 cosTheta   1.004 sin2Theta   0.000 wavelength 118.450 count 100 gs.matline 39 gs.pos (-151.039, -1.577,-41.422)
    //qcerenkov::wavelength_sampled_bndtex idx 9075129 sampledRI   1.263 cosTheta   1.004 sin2Theta   0.000 wavelength  86.463 count 100 gs.matline 39 gs.pos (-151.039, -1.577,-41.422)
    //qcerenkov::wavelength_sampled_bndtex idx 9075744 sampledRI   1.263 cosTheta   1.128 sin2Theta   0.000 wavelength 724.662 count 100 gs.matline 39 gs.pos (-150.997, -1.725,-41.745)
    //qcerenkov::wavelength_sampled_bndtex idx 9075745 sampledRI   1.263 cosTheta   1.128 sin2Theta   0.000 wavelength 468.014 count 100 gs.matline 39 gs.pos (-150.997, -1.725,-41.745)
    //qcerenkov::wavelength_sampled_bndtex idx 9075746 sampledRI   1.263 cosTheta   1.128 sin2Theta   0.000 wavelength 116.840 count 100 gs.matline 39 gs.pos (-150.997, -1.725,-41.745)
    //qcerenkov::wavelength_sampled_bndtex idx 9075747 sampledRI   1.263 cosTheta   1.128 sin2Theta   0.000 wavelength 124.767 count 100 gs.matline 39 gs.pos (-150.997, -1.725,-41.745)
    //qcerenkov::wavelength_sampled_bndtex idx 9075748 sampledRI   1.263 cosTheta   1.128 sin2Theta   0.000 wavelength 172.011 count 100 gs.matline 39 gs.pos (-150.997, -1.725,-41.745)
    //qcerenkov::wavelength_sampled_bndtex idx 9075749 sampledRI   1.263 cosTheta   1.128 sin2Theta   0.000 wavelength 141.308 count 100 gs.matline 39 gs.pos (-150.997, -1.725,-41.745)
    //qcerenkov::wavelength_sampled_bndtex idx 9075750 sampledRI   1.263 cosTheta   1.128 sin2Theta   0.000 wavelength 134.997 count 100 gs.matline 39 gs.pos (-150.997, -1.725,-41.745)
    //qcerenkov::wavelength_sampled_bndtex idx 9075751 sampledRI   1.263 cosTheta   1.128 sin2Theta   0.000 wavelength 272.860 count 100 gs.matline 39 gs.pos (-150.997, -1.725,-41.745)
    //qcerenkov::wavelength_sampled_bndtex idx 9075743 sampledRI   1.263 cosTheta   1.128 sin2Theta   0.000 wavelength 148.201 count 100 gs.matline 39 gs.pos (-150.997, -1.725,-41.745)
    SProf::Write DISABLED, enable[export SProf__WRITE=1] disable[unset SProf__WRITE]
    2026-05-07 18:33:36.024 INFO  [897661] [QSim::simulate@736]  eventID    999 gs (114095, 6, 4, ) ht (6519, 4, 4, ) tot_dt   0.350463 server_settings HitCompOneName:hit,PhotonCompOneName:photon tree_digest 79f17049c1f5806abe058cf4449eb712
          INFO   127.0.0.1:59338 - "POST /simulate HTTP/1.1" 200


Issue 2 : gs monotonically increasing - not cleared in client running ?
--------------------------------------------------------------------------

::

     2026-05-07 18:33:36.024 INFO  [897661] [QSim::simulate@736]  eventID    999 gs (114095, 6, 4, ) ht (6519, 4, 4, ) tot_dt   0.350463 server_settings HitCompOneName:hit,PhotonCompOneName:photon tree_digest 79f17049c1f5806abe058cf4449eb712


::

    149 inline double SOpticksClientSimulator::simulate(int eventID, bool reset )
    150 {
    151     sev->beginOfEvent(eventID);
    152     NP* gs = sev->makeGenstepArrayFromVector();
    153     if(gs == nullptr)
    154     {
    155         std::cerr
    156             << "SOpticksClientSimulator::simulate"
    157             << " eventID " << eventID
    158             << " NO GENSTEPS - NOTHING TO DO "
    159             << "\n"
    160             ;
    161         return 0;
    162     }
    163 
    164 #ifdef WITH_SEVT_MOCK
    165 #else
    166     std::string client_settings = SEventConfig::Settings();
    167     std::string client_digest = tree_digest ;
    168     gs->set_meta<std::string>("Settings",client_settings);
    169     gs->set_meta<std::string>("TreeDigest",client_digest);
    170 #endif
    171 
    172     NP* hc = NP_CURL::TransformRemote(gs,eventID);  // "hc" hit-component one of : hit/hitlite/hitlitemerged/hitmerged
    173 
    174     if(hc == nullptr)
    175     {
    176         std::cerr << "SOpticksClientSimulator::simulate  ERROR NP_CURL::TransformRemote gave hc (NP*)nullptr - IS THE SERVER RUNNING ?\n" ;
    177         return -1. ;
    178     }
    179 
    180 
    181     sev->setHit(hc);
    182     double dt = hc ? hc->get_meta<double>("QSim__simulate_tot_dt", 0. ) : -1. ;
    183 
    184 
    185     std::cout
    186           << "SOpticksClientSimulator::simulate "
    187           << " eventID " << eventID
    188           << " reset " << reset
    189           << " gs " << ( gs ? gs->sstr() : "-" )
    190           << " hc " << ( hc ? hc->sstr() : "-" )
    191           << " dt " << dt
    192           << "\n"
    193           ;


::

    3466 NP* SEvt::makeGenstepArrayFromVector() const
    3467 {
    3468     return NPX::ArrayFromData<float>( (float*)genstep.data(), int(genstep.size()), 6, 4 ) ;
    3469 }
    3470 


    2069 void SEvt::clear_genstep_vector()
    2070 {
    2071     numgenstep_collected = 0 ;
    2072     numphoton_collected = 0 ;
    2073     numphoton_genstep_max = 0 ;
    2074 
    2075     clear_genstep_vector_count += 1 ;
    2076 
    2077     setNumPhoton(0);
    2078 
    2079     gs.clear();
    2080     genstep.clear();
    2081     gather_done = false ;
    2082 }
    2083 


    2163 /**
    2164 SEvt::clear_genstep
    2165 ---------------------
    2166 
    2167 * canonical call from SEvt::endOfEvent
    2168 
    2169 **/
    2170 
    2171 
    2172 void SEvt::clear_genstep()
    2173 {
    2174     setStage(SEvt__clear_genstep);
    2175     LOG_IF(info, LIFECYCLE) << id() << " BEFORE clear_genstep_vector " ;
    2176 
    2177     clear_genstep_vector();
    2178     topfold->clear_only("genstep", false, ',');
    2179 
    2180     LOG_IF(info, LIFECYCLE) << id() << " AFTER clear_genstep_vector " ;
    2181 }


    1906 void SEvt::endOfEvent(int eventID)
    1907 {
    1908 
    1909     setStage(SEvt__endOfEvent);
    1910     LOG_IF(info, LIFECYCLE) << id() ;
    1911     sprof::Stamp(p_SEvt__endOfEvent_0);
    1912 
    1913     endIndex(eventID);   // eventID is 0-based
    1914     endMeta();
    1915     gather_metadata();
    1916 
    1917     save();              // gather and save SEventConfig configured arrays
    1918     clear_output();
    1919     clear_genstep();
    1920     clear_extra();
    1921     reset_counter();
    1922 
    1923     SaveRunMeta(); // saving run_meta.txt at end of every event incase of crashes
    1924 
    1925 
    1926     bool is_last_eventID = SEventConfig::IsLastEvent(eventID) ;
    1927     if(is_last_eventID)
    1928     {
    1929         //SetRunProf( isEGPU() ? "SEvt__endOfEvent_LAST_EGPU" : "SEvt__endOfEvent_LAST_ECPU" ) ;
    1930         bool is_last_evt_instance = isLastEvtInstance() ;
    1931 
    1932         LOG(LEVEL)
    1933             << " is_last_eventID " << ( is_last_eventID ? "YES" : "NO " )
    1934             << " is_last_evt_instance " << ( is_last_evt_instance ? "YES" : "NO " )
    1935             ;
    1936 
    1937         if(is_last_evt_instance) SEvt::EndOfRun();   // invokes SaveRunMeta
    1938     }
    1939 
    1940 
    1941 }
    1942 


trace the reset in monolithic running 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Q: How does the below mono-reset chain differ from the client one ?
A: No CSGOptiX or QSim client side, so SOpticksClientSimulator.h needs to do the equivalent.


HMM in ordinary "monolithic" running QSim::reset calls SEvt::endOfEvent::

    0832 /**
     833 QSim::reset
     834 ------------
     835 
     836 When *QSim::simulate* is called with argument *reset:true* the
     837 *QSim::reset* method is called which invokes SEvt::endOfEvent in order
     838 to clean up the SEvt. Normally that would be done after saving
     839 any Opticks configured arrays.
     840 
     841 When *QSim::simulate* is called with argument *reset:false*
     842 the *QSim::reset* method must be called separately in order to avoid a memory leak.
     843 Using *reset:false* is typically done in order to keep arrays alive longer
     844 to enable copying from the gathered arrays into non-Opticks collections.
     845 
     846 **/
     847 void QSim::reset(int eventID)
     848 {
     849     SProf::Add("QSim__reset_HEAD");
     850     qev->clear();
     851     sev->endOfEvent(eventID);
     852     LOG_IF(info, SEvt::LIFECYCLE) << "] eventID " << eventID ;
     853     SProf::Add("QSim__reset_TAIL");
     854 }


     737 void CSGOptiX::reset(int eventID)
     738 {
     739     assert(sim);
     740     sim->reset(eventID); // (QSim)
     741 }


    480 void G4CXOpticks::reset(int eventID)
    481 {
    482     LOG_IF(fatal, NoSim) << "NoSim SKIP" ;
    483     if(NoSim) return ;
    484 
    485     assert( SEventConfig::IsRGModeSimulate() );
    486 
    487     unsigned num_hit_0 = SEvt::GetNumHit_EGPU() ;
    488     LOG(LEVEL) << "[ " << eventID << " num_hit_0 " << num_hit_0  ;
    489     cx->reset(eventID);
    490 
    491     unsigned num_hit_1 = SEvt::GetNumHit_EGPU() ;
    492     LOG(LEVEL) << "] " << eventID << " num_hit_1 " << num_hit_1  ;
    493 }


    660 void junoSD_PMT_v2_Opticks::EndOfEvent_reset(int eventID )
    661 {
    662 #ifdef WITH_G4CXOPTICKS_EPH
    663     EndOfEvent_recordStats(eventID);
    664 #endif
    665     G4CXOpticks* gx = G4CXOpticks::Get() ;
    666     gx->reset(eventID) ;
    667 }


    [lo] A[blyth@localhost junosw]$ jgr EndOfEvent_reset
    ./Simulation/DetSimV2/PMTSim/PMTSim/junoSD_PMT_v2_Opticks.hh:        void EndOfEvent_reset(   int eventID );
    ./Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2.cc:        m_jpmt_opticks->EndOfEvent_reset(m_eventID );
    ./Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2_Opticks.cc:void junoSD_PMT_v2_Opticks::EndOfEvent_reset(int eventID )
    [lo] A[blyth@localhost junosw]$ 



    1149 void junoSD_PMT_v2::EndOfEvent(G4HCofThisEvent* HCE)
    1150 {
    1151     G4cout << J::Log("[junoSD_PMT_v2::EndOfEvent") << desc() << G4endl ;
    1152 
    1153 #ifdef WITH_G4CXOPTICKS
    1154     if(m_eph_stats)
    1155     {
    1156         m_eph_stats->EndOfEvent_hitCollection_entries0 = hitCollection->entries();
    1157         m_eph_stats->EndOfEvent_hitCollectionAlt_entries0 = hitCollectionAlt ? hitCollectionAlt->entries() : 0 ;
    1158     }
    1159 
    1160     if(m_jpmt_opticks)
    1161     {
    1162         m_jpmt_opticks->EndOfEvent(HCE, m_eventID );
    1163     }
    1164 
    1165     if(m_eph_stats)
    1166     {
    1167         m_eph_stats->EndOfEvent_hitCollection_entries1 = hitCollection->entries();
    1168         m_eph_stats->EndOfEvent_hitCollectionAlt_entries1 = hitCollectionAlt ? hitCollectionAlt->entries() : 0 ;
    1169     }
    1170 
    1171     if(gpu_simulation())
    1172     {
    1173         m_jpmt_opticks->EndOfEvent_reset(m_eventID );
    1174     }
    1175 #endif
    1176     G4cout << J::Log("]junoSD_PMT_v2::EndOfEvent") << desc() << G4endl ;
    1177 }

















    199 void junoSD_PMT_v2_Opticks::EndOfEvent(G4HCofThisEvent*, int eventID )
    200 {
    201     G4CXOpticks* gx = G4CXOpticks::Get() ;
    202     gx->SensitiveDetector_EndOfEvent(eventID) ; // invokes U4Recorder::EndOfEventAction_ when recorder enabled
    203 
    204     if(m_merger == nullptr) m_merger = m_jpmt->getMergerOpticks(); // Alt merger for opticksMode:3
    205 
    206     LOG(LEVEL) << "[ " << m_jpmt->desc() ;
    207 
    208     LOG(LEVEL) << std::endl << m_jpmt->descHitCollection() ;
    209 
    210 
    211     if(m_gpu_simulation)
    212     {
    213         EndOfEvent_Simulate(eventID) ; // calls G4CXOpticks::simulate,  collectHit,  G4CXOpticks::reset
    214     }
    215 
    216     LOG(LEVEL) << "] " << m_jpmt->desc() ;
    217 }



    248 void junoSD_PMT_v2_Opticks::EndOfEvent_Simulate(int eventID )
    249 {
    250     SProf::Add("junoSD_PMT_v2_Opticks__EndOfEvent_Simulate_HEAD");
    251     G4CXOpticks* gx = G4CXOpticks::Get() ;
    252 
    253     bool reset_ = false ;
    254     gx->simulate(eventID, reset_ ) ;
    255 
    256     LOG(LEVEL)
    257         << "[ eventID " << eventID
    258         << " " << gx->descSimulate()
    259         ;
    260 
    261 
    262 
    263     switch(m_hit_type)
    264     {
    265 #ifdef WITH_U4HIT
    266        case 1: EndOfEvent_CollectNormHits(eventID) ; break ;
    267 #else
    268        case 1: EndOfEvent_CollectFullHits(eventID) ; break ;
    269 #endif
    270        case 2: EndOfEvent_CollectMuonHits(eventID) ; break ;
    271     }
    272     SProf::Add("junoSD_PMT_v2_Opticks__EndOfEvent_Simulate_TAIL");
    273 
    274 }









Issue 3 : client side also too verbose
----------------------------------------

::

     SOpticksClientSimulator::simulate  eventID 998 reset 0 hc (6527, 4, 4, ) dt 0.35110

     SOpticksClientSimulator::simulate  eventID 999 reset 0 hc (6519, 4, 4, ) dt 0.35046



