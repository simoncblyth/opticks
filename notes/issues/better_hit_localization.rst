better_hit_localization
========================


Issue : sore thumb CPU hitmerged+hit localization
----------------------------------------------------

See https://simoncblyth.github.io/env/presentation/opticks_20260122_wuhan.html?p=8

For "hitmerged" running the localization of 6.4M took 6.712 seconds. 
That is too large an overhead given that simulation time is ~23 seconds.
Clearly need to use GPU to do this localization.


Current OJ Localization
-------------------------

HMM: need to merge this branch to make sense of localization::

    (ok) A[blyth@localhost junosw]$ js
    /home/blyth/junosw
    origin	git@code.ihep.ac.cn:JUNO/offline/junosw.git (fetch)
    origin	git@code.ihep.ac.cn:JUNO/offline/junosw.git (push)
    On branch blyth-OJ-pmt-hit-type-2-for-muon-hits
    Your branch is up to date with 'origin/blyth-OJ-pmt-hit-type-2-for-muon-hits'.

    nothing to commit, working tree clean
    (ok) A[blyth@localhost junosw]$ 

::

    (ok) A[blyth@localhost junosw]$ git lg -n20
    * 2025-12-08 a65ea362 - (HEAD -> blyth-OJ-pmt-hit-type-2-for-muon-hits, origin/blyth-OJ-pmt-hit-type-2-for-muon-hits) temporarily skip setting OPTICKS_MODE_MERGE as want to compare performance with and without (7 weeks ago) <Simon C Blyth>
    * 2025-12-08 b2a6795d - switch to new FullHit impl removing pointless intermediary U4Hit (7 weeks ago) <Simon C Blyth>
    * 2025-12-01 cbe401b9 - start revival of opticksMode:0 for standard CPU simulation with some Opticks instrumentation (8 weeks ago) <Simon C Blyth>
    * 2025-12-01 03087ce2 - make it explicit that SEventConfig::ModeMerge dictates if the hits are premerged or not (8 weeks ago) <Simon C Blyth>
    * 2025-12-01 6dd97627 - explicitly restrict use of Geant4 track info to opticksMode 3 (8 weeks ago) <Simon C Blyth>
    * 2025-12-01 4b526503 - overhaul NormalTrackInfo and its usage, replacing all dangerous C-style casts with dynamic cast from static helper NormalTrackInfo::FromTrack, add standalone test and optional degug registry functionality (8 weeks ago) <Simon C Blyth>
    * 2025-12-01 e7ba84bd - replace all? python print with log.info for time stamping and consistency (8 weeks ago) <Simon C Blyth>
    * 2025-11-25 2319df6f - add JLog.h which provides J::Log for millisecond formatted time stamps (9 weeks ago) <Simon C Blyth>
    * 2025-11-25 f01f65a4 - enable Opticks hitlitemerged for pmt_hit_type:2 hitmerged not yet impl (9 weeks ago) <Simon C Blyth>
    * 2025-11-14 a5231d99 - add EndOfEvent_CollectMuonHits_premerged that handles hitlite arrays that have been merged already, old traditional PMTHitMerger approach is moved to EndOfEvent_CollectMuonHits_cpumerge (2 months ago) <Simon C Blyth>
    * 2025-11-14 e703d28a - improve python logging of the init methods, add init_opticks_environ that passes pmt hit type and merge window config to Opticks (2 months ago) <Simon C Blyth>
    * 2025-11-12 1fe32f2e - use new opticks photonlite/hitlite to implement junoSD_PMT_v2_Opticks::EndOfEvent_CollectMuonHits not yet assuming GPU hit merging (3 months ago) <Simon C Blyth>
    * 2025-11-04 deeae12d - working out whats needed to get muon hits from sphotonlite hitlite array (3 months ago) <Simon C Blyth>
    *   2025-10-29 429466da - Merge branch 'blyth-fix-opticks-PMT-serialization-issue-from-partial-addition-of-600-MPMT' into 'main' (3 months ago) <lintao@ihep.ac.cn>
    |\  
    | * 2025-10-29 4976b4ff - Avoids Opticks PMT serialization inconsistency between PMTParamData and PMTSimParamData by collecting added pmtNum info from PMTParamData (3 months ago) <blyth@ihep.ac.cn>
    * |   2025-10-29 8abae80c - Merge branch 'lintao/reprod25c/enable-trigger-edm-in-esd' into 'main' (3 months ago) <lintao@ihep.ac.cn>
    |\ \  
    | |/  
    |/|   





OPTICKS_MODE_MERGE OPTICKS_MODE_LITE
---------------------------------------

::

    1438 unsigned SEventConfig::HitCompOne() // static
    1439 {
    1440     unsigned comp = 0 ;
    1441     int64_t lite = ModeLite();
    1442     int64_t merge = ModeMerge();
    1443     int64_t lite_merge = lite*10 + merge ;
    1444 
    1445     if( lite == 0 || lite == 1 )
    1446     {
    1447         switch(lite_merge)
    1448         {
    1449             case   0: comp = SCOMP_HIT            ; break ;
    1450             case   1: comp = SCOMP_HITMERGED      ; break ;
    1451             case  10: comp = SCOMP_HITLITE        ; break ;
    1452             case  11: comp = SCOMP_HITLITEMERGED  ; break ;
    1453         }
    1454     }
    1455     else if( lite == 2 )  // debug only
    1456     {
    1457         switch(lite_merge)
    1458         {
    1459             case  20: comp = SCOMP_HITLITE        ; break ;
    1460             case  21: comp = SCOMP_HITLITEMERGED  ; break ;
    1461         }
    1462     }
    1463 
    1464     return comp ;
    1465 }




First consider photons that fit in VRAM
-------------------------------------------

The "hitmerged" sphoton are merged on GPU via QEvt::PerLaunchMerge::

    1158 template<typename T>
    1159 NP* QEvt::PerLaunchMerge(sevent* evt, cudaStream_t stream ) // static
    1160 {
    1161     // below four calls return whats appropriate depending on template type of sphoton OR sphotonlite
    1162     // dealing with either  photonlite/hitlitemerged OR photon/hitmerged
    1163 
    1164     T* d_in = evt->get_photon_ptr<T>();
    1165     size_t num_in = evt->get_photon_num<T>();
    1166 
    1167     T** d_out_ref = evt->get_hitmerged_ptr_ref<T>();
    1168     size_t* num_out_ref = evt->get_hitmerged_num_ref<T>();
    1169 
    1170     SPM::merge_partial_select(
    1171          d_in,
    1172          num_in,
    1173          d_out_ref,
    1174          num_out_ref,
    1175          SEventConfig::HitMask(),
    1176          SEventConfig::MergeWindow(),
    1177          stream);
    1178 
    1179     cudaStreamSynchronize(stream); // blocks until all preceeding operations in stream complete
    1180 
    1181     NP* out = T::zeros( *num_out_ref ); // hitmerged OR hitlitemerged
    1182 
    1183     SPM::copy_device_to_host_async<T>( (T*)out->bytes(), *d_out_ref, *num_out_ref, stream );

    

