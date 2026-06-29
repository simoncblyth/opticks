NP_SProf_annotation_parsing_bugfix_reveals_inconsistent_last_event_tagging
===========================================================================

While trying to propagate SProf.txt annotations thru to the ranges_names
discover SProf parsing bug. Lines with annotations were failing U::LooksLikeProfileTriplet
because the value parsing did not handle the annotation string

Switch on debug for report creation with::

    sreport__CONFIG=creator cxs_min.sh report


SProf.txt annotations::

    A[blyth@localhost ALL1_Debug_Philox_medium_scan_first]$ cat SProf.txt
    SEvt__Init_RUN_META:1782097507944130,49476,9408
    CSGOptiX__SimulateMain_HEAD:1782097507950990,49600,13440
    CSGFoundry__Load_HEAD:1782097507951014,49600,13440
    CSGFoundry__Load_TAIL:1782097509113453,5427492,871680
    CSGOptiX__Create_HEAD:1782097509113494,5427492,871680
    CSGOptiX__Create_TAIL:1782097509501418,8128336,1264128
    SEvt__BeginOfRun:1782097509501485,8128336,1264128
    SEvt__beginOfEvent_FIRST_EGPU:1782097509501630,8128336,1264128
    A000_SEvt__setIndex:1782097509501661,8128336,1264128
    A000_QSim__simulate_HEAD:1782097509501775,8128336,1264128
    A000_QSim__simulate_LBEG:1782097509501818,8128336,1264128
    A000_QSim__simulate_PRUP:1782097509501825,8128336,1264128
    A000_QSim__simulate_PREL:1782097509515735,15468368,1265024
    A000_QSim__simulate_POST:1782097509813007,15468368,1270400
    A000_QSim__simulate_DOWN:1782097509831240,15481740,1283840
    A000_QSim__simulate_LEND:1782097509831276,15481740,1283840 # slice=1,max_slot_M=100
    A000_QSim__simulate_PCAT:1782097509831298,15481740,1283840
    A000_QSim__simulate_BRES:1782097509831315,15481740,1283840 # numGenstepCollected=1,numPhotonCollected=1000000,numHit=213947
    A000_QSim__reset_HEAD:1782097509831323,15481740,1283840
    A000_SEvt__endIndex:1782097509831341,15481740,1283840
    A000_QSim__reset_TAIL:1782097509897853,15468368,1271048
    A000_QSim__simulate_TAIL:1782097509897879,15468368,1271048
    A001_SEvt__setIndex:1782097509897933,15468368,1271048
    A001_QSim__simulate_HEAD:1782097509897984,15468368,1271048
    A001_QSim__simulate_LBEG:1782097509898007,15468368,1271048
    A001_QSim__simulate_PRUP:1782097509898014,15468368,1271048
    A001_QSim__simulate_PREL:1782097509898896,15468368,1271048
    A001_QSim__simulate_POST:1782097510191688,15468368,1271048
    A001_QSim__simulate_DOWN:1782097510212450,15481680,1284040
    A001_QSim__simulate_LEND:1782097510212476,15481680,1284040 # slice=1,max_slot_M=100
    A001_QSim__simulate_PCAT:1782097510212491,15481680,1284040
    A001_QSim__simulate_BRES:1782097510212500,15481680,1284040 # numGenstepCollected=1,numPhotonCollected=1000000,numHit=212942
    A001_QSim__reset_HEAD:1782097510212505,15481680,1284040
    A001_SEvt__endIndex:1782097510212520,15481680,1284040
    SEvt__EndOfRun:1782097510272031,15481680,1284040
    QSim__reset_TAIL:1782097510272088,15481680,1284040
    QSim__simulate_TAIL:1782097510272095,15481680,1284040
    CSGOptiX__SimulateMain_TAIL:1782097510272723,15481680,1284040




Before fix::

    A[blyth@localhost ALL1_Debug_Philox_medium_scan_first_sreport]$ cat ranges_names.txt
    SEvt__Init_RUN_META:CSGFoundry__Load_HEAD:init
    CSGFoundry__Load_HEAD:CSGFoundry__Load_TAIL:load_geom
    CSGOptiX__Create_HEAD:CSGOptiX__Create_TAIL:upload_geom
    A000_QSim__simulate_HEAD:A000_QSim__simulate_LBEG:slice_genstep
    A000_QSim__simulate_PRUP:A000_QSim__simulate_PREL:upload genstep slice
    A000_QSim__simulate_PREL:A000_QSim__simulate_POST:simulate slice
    A000_QSim__simulate_POST:A000_QSim__simulate_DOWN:download slice
    A001_QSim__simulate_HEAD:A001_QSim__simulate_LBEG:slice_genstep
    A001_QSim__simulate_PRUP:A001_QSim__simulate_PREL:upload genstep slice
    A001_QSim__simulate_PREL:A001_QSim__simulate_POST:simulate slice
    A001_QSim__simulate_POST:A001_QSim__simulate_DOWN:download slice

With sreport::RANGES::

   static constexpr const char* RANGES = R"(
        SEvt__Init_RUN_META:CSGFoundry__Load_HEAD                     ## init
        CSGFoundry__Load_HEAD:CSGFoundry__Load_TAIL                   ## load_geom
        CSGOptiX__Create_HEAD:CSGOptiX__Create_TAIL                   ## upload_geom
        A%0.3d_QSim__simulate_HEAD:A%0.3d_QSim__simulate_LBEG         ## slice_genstep
        A%0.3d_QSim__simulate_PRUP:A%0.3d_QSim__simulate_PREL         ## upload genstep slice
        A%0.3d_QSim__simulate_PREL:A%0.3d_QSim__simulate_POST         ## simulate slice
        A%0.3d_QSim__simulate_POST:A%0.3d_QSim__simulate_DOWN         ## download slice
        A%0.3d_QSim__simulate_LEND:A%0.3d_QSim__simulate_PCAT         ## concat slices
        A%0.3d_QSim__simulate_BRES:A%0.3d_QSim__simulate_TAIL         ## save arrays
       )" ;




After parsing fix - and addition of anno propagation - see more ranges::

    A[blyth@localhost ALL1_Debug_Philox_medium_scan_first_sreport]$ cat ranges_names.txt
    SEvt__Init_RUN_META:CSGFoundry__Load_HEAD:init
    CSGFoundry__Load_HEAD:CSGFoundry__Load_TAIL:load_geom
    CSGOptiX__Create_HEAD:CSGOptiX__Create_TAIL:upload_geom

    A000_QSim__simulate_HEAD:A000_QSim__simulate_LBEG:slice_genstep
    A000_QSim__simulate_PRUP:A000_QSim__simulate_PREL:upload genstep slice
    A000_QSim__simulate_PREL:A000_QSim__simulate_POST:simulate slice
    A000_QSim__simulate_POST:A000_QSim__simulate_DOWN:download slice
    A000_QSim__simulate_LEND:A000_QSim__simulate_PCAT:concat slices # slice=1,max_slot_M=100
    A000_QSim__simulate_BRES:A000_QSim__simulate_TAIL:save arrays # numGenstepCollected=1,numPhotonCollected=1000000,numHit=213947

    A001_QSim__simulate_HEAD:A001_QSim__simulate_LBEG:slice_genstep
    A001_QSim__simulate_PRUP:A001_QSim__simulate_PREL:upload genstep slice
    A001_QSim__simulate_PREL:A001_QSim__simulate_POST:simulate slice
    A001_QSim__simulate_POST:A001_QSim__simulate_DOWN:download slice
    A001_QSim__simulate_LEND:A001_QSim__simulate_PCAT:concat slices # slice=1,max_slot_M=100

BUT: second and last event misses final "save arrays" range ? Maybe off-by-one bug ?


sreport logging suggests the problem is with inconsistent tagging for last evt::

    [NP::MakeMetaKVS_ranges2_table num_specs 14 num_keys 38 num_anno 38
     (specs correspond to ranges - so will be smaller than keys which should be same as anno)
    -ranges2_table  k   0 key            SEvt__Init_RUN_META anno [                              ]
    -ranges2_table  k   1 key    CSGOptiX__SimulateMain_HEAD anno [                              ]
    -ranges2_table  k   2 key          CSGFoundry__Load_HEAD anno [                              ]
    -ranges2_table  k   3 key          CSGFoundry__Load_TAIL anno [                              ]
    -ranges2_table  k   4 key          CSGOptiX__Create_HEAD anno [                              ]
    -ranges2_table  k   5 key          CSGOptiX__Create_TAIL anno [                              ]
    -ranges2_table  k   6 key               SEvt__BeginOfRun anno [                              ]
    -ranges2_table  k   7 key  SEvt__beginOfEvent_FIRST_EGPU anno [                              ]
    -ranges2_table  k   8 key            A000_SEvt__setIndex anno [                              ]
    -ranges2_table  k   9 key       A000_QSim__simulate_HEAD anno [                              ]
    -ranges2_table  k  10 key       A000_QSim__simulate_LBEG anno [                              ]
    -ranges2_table  k  11 key       A000_QSim__simulate_PRUP anno [                              ]
    -ranges2_table  k  12 key       A000_QSim__simulate_PREL anno [                              ]
    -ranges2_table  k  13 key       A000_QSim__simulate_POST anno [                              ]
    -ranges2_table  k  14 key       A000_QSim__simulate_DOWN anno [                              ]
    -ranges2_table  k  15 key       A000_QSim__simulate_LEND anno [        slice=1,max_slot_M=100]
    -ranges2_table  k  16 key       A000_QSim__simulate_PCAT anno [                              ]
    -ranges2_table  k  17 key       A000_QSim__simulate_BRES anno [numGenstepCollected=1,numPhotonCollected=1000000,numHit=213947]
    -ranges2_table  k  18 key          A000_QSim__reset_HEAD anno [                              ]
    -ranges2_table  k  19 key            A000_SEvt__endIndex anno [                              ]
    -ranges2_table  k  20 key          A000_QSim__reset_TAIL anno [                              ]
    -ranges2_table  k  21 key       A000_QSim__simulate_TAIL anno [                              ]
    -ranges2_table  k  22 key            A001_SEvt__setIndex anno [                              ]
    -ranges2_table  k  23 key       A001_QSim__simulate_HEAD anno [                              ]
    -ranges2_table  k  24 key       A001_QSim__simulate_LBEG anno [                              ]
    -ranges2_table  k  25 key       A001_QSim__simulate_PRUP anno [                              ]
    -ranges2_table  k  26 key       A001_QSim__simulate_PREL anno [                              ]
    -ranges2_table  k  27 key       A001_QSim__simulate_POST anno [                              ]
    -ranges2_table  k  28 key       A001_QSim__simulate_DOWN anno [                              ]
    -ranges2_table  k  29 key       A001_QSim__simulate_LEND anno [        slice=1,max_slot_M=100]
    -ranges2_table  k  30 key       A001_QSim__simulate_PCAT anno [                              ]
    -ranges2_table  k  31 key       A001_QSim__simulate_BRES anno [numGenstepCollected=1,numPhotonCollected=1000000,numHit=212942]

    -ranges2_table  k  32 key          A001_QSim__reset_HEAD anno [                              ]
    -ranges2_table  k  33 key            A001_SEvt__endIndex anno [                              ]

    -ranges2_table  k  34 key                 SEvt__EndOfRun anno [                              ]
    -ranges2_table  k  35 key               QSim__reset_TAIL anno [                              ]
    -ranges2_table  k  36 key            QSim__simulate_TAIL anno [                              ]
    -ranges2_table  k  37 key    CSGOptiX__SimulateMain_TAIL anno [                              ]


::

     578     int64_t t_BRES  = SProf::Add("QSim__simulate_BRES", counts.c_str() );
     579     if(reset_) reset(eventID) ;   // reset calles SEvt::endOfEvent
     580
     581     int64_t t_TAIL  = SProf::Add("QSim__simulate_TAIL");
     582
     583     SProf::Write(); // per-event write, so have something in case of crash
     584



     848 void QSim::reset(int eventID)
     849 {
     850     SProf::Add("QSim__reset_HEAD");
     851     qev->clear();
     852     sev->endOfEvent(eventID);
     853     LOG_IF(info, SEvt::LIFECYCLE) << "] eventID " << eventID ;
     854     SProf::Add("QSim__reset_TAIL");
     855 }


    1968 void SEvt::endOfEvent(int eventID)
    1969 {
    1970
    1971     setStage(SEvt__endOfEvent);
    1972     LOG_IF(info, LIFECYCLE) << " eventID[" << eventID << "] " << id() ;
    1973     sprof::Stamp(p_SEvt__endOfEvent_0);
    1974
    1975     endIndex(eventID);   // eventID is 0-based
    1976     endMeta();
    1977     gather_metadata();
    1978
    1979     save();              // gather and save SEventConfig configured arrays
    1980     clear_output();
    1981     clear_genstep();
    1982     clear_extra();
    1983     reset_counter();
    1984
    1985     SaveRunMeta(); // saving run_meta.txt at end of every event incase of crashes
    1986
    1987
    1988     bool is_last_eventID = SEventConfig::IsLastEvent(eventID) ;
    1989     if(is_last_eventID)
    1990     {
    1991         //SetRunProf( isEGPU() ? "SEvt__endOfEvent_LAST_EGPU" : "SEvt__endOfEvent_LAST_ECPU" ) ;
    1992         bool is_last_evt_instance = isLastEvtInstance() ;
    1993
    1994         LOG(LEVEL)
    1995             << " is_last_eventID " << ( is_last_eventID ? "YES" : "NO " )
    1996             << " is_last_evt_instance " << ( is_last_evt_instance ? "YES" : "NO " )
    1997             ;
    1998
    1999         if(is_last_evt_instance)
    2000         {
    2001             SProf::UnsetTag();
    2002             SEvt::EndOfRun();   // invokes SaveRunMeta
    2003         }
    2004     }
    2005
    2006
    2007 }




Tagging Inconsistency from SProf::UnsetTag called too soon for the last event ?::

     184 int CSGOptiX::SimulateMain() // static
     185 {
     186     SProf::Add("CSGOptiX__SimulateMain_HEAD");
     187     SEventConfig::SetRGModeSimulate();
     188     CSGFoundry* fd = CSGFoundry::Load();
     189     CSGOptiX* cx = CSGOptiX::Create(fd) ;
     190     bool reset = true ;
     191     for(int i=0 ; i < SEventConfig::NumEvent() ; i++) cx->simulate(i, reset);
     192     //SProf::UnsetTag();  // WIP: RELOCATED TO SEvt::endOfEvent
     193     SProf::Add("CSGOptiX__SimulateMain_TAIL");
     194     SProf::Write();
     195     cx->write_Ctx_log();
     196     delete cx ;
     197     return 0 ;
     198 }


::

    1999         if(is_last_evt_instance)
    2000         {
    2001             // SProf::UnsetTag();
    2002             //
    2003             //   LOOKS LIKE ITS TOO SOON TO DROP THE TAG OF LAST EVT HERE
    2004             //   AS IT CAUSES INCONSISTENT LAST EVENT SProf.hh TAGGING
    2005             //
    2006             //   Try relocating UnsetTag to just after QSim__simulate_TAIL
    2007             //   in attempt for last event tagging to be more consistent with other events
    2008             //
    2009             //   see ~/o/notes/issues/NP_SProf_annotation_parsing_bugfix_reveals_inconsistent_last_event_tagging.rst
    2010             //
    2011
    2012             SEvt::EndOfRun();   // invokes SaveRunMeta
    2013         }
    2014     }
    2015
    2016    // tail of SEvt::endOfEvent
    2017 }
    2018




After moving UnsetTag to after QSim__simulate_TAIL get consistent evt tagging::

    A[blyth@localhost ALL1_Debug_Philox_medium_scan_first_sreport]$ cat ranges_names.txt
    SEvt__Init_RUN_META:CSGFoundry__Load_HEAD:init
    CSGFoundry__Load_HEAD:CSGFoundry__Load_TAIL:load_geom
    CSGOptiX__Create_HEAD:CSGOptiX__Create_TAIL:upload_geom

    A000_QSim__simulate_HEAD:A000_QSim__simulate_LBEG:slice_genstep
    A000_QSim__simulate_PRUP:A000_QSim__simulate_PREL:upload genstep slice
    A000_QSim__simulate_PREL:A000_QSim__simulate_POST:simulate slice
    A000_QSim__simulate_POST:A000_QSim__simulate_DOWN:download slice
    A000_QSim__simulate_LEND:A000_QSim__simulate_PCAT:concat slices # slice=1,max_slot_M=100
    A000_QSim__simulate_BRES:A000_QSim__simulate_TAIL:save arrays # numGenstepCollected=1,numPhotonCollected=1000000,numHit=213947

    A001_QSim__simulate_HEAD:A001_QSim__simulate_LBEG:slice_genstep
    A001_QSim__simulate_PRUP:A001_QSim__simulate_PREL:upload genstep slice
    A001_QSim__simulate_PREL:A001_QSim__simulate_POST:simulate slice
    A001_QSim__simulate_POST:A001_QSim__simulate_DOWN:download slice
    A001_QSim__simulate_LEND:A001_QSim__simulate_PCAT:concat slices # slice=1,max_slot_M=100
    A001_QSim__simulate_BRES:A001_QSim__simulate_TAIL:save arrays # numGenstepCollected=1,numPhotonCollected=1000000,numHit=212942

    A[blyth@localhost ALL1_Debug_Philox_medium_scan_first_sreport]$


