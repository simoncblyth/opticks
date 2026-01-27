better_hit_localization
========================


Issue : sore thumb CPU hitmerged+hit localization
----------------------------------------------------

See https://simoncblyth.github.io/env/presentation/opticks_20260122_wuhan.html?p=8

For "hitmerged" running the localization of 6.4M took 6.712 seconds.
That is too large an overhead given that simulation time is ~23 seconds.
Clearly need to use GPU to do this localization.

Steps for GPU side localization
---------------------------------

::

    In [2]: 50000*4*4*8/1e6
    Out[2]: 6.4


1. upload 50000*4*4*8  about 6.4 MB of iinst transforms (double precision) or half that in float in CSGFoundry::upload
2. hold d_iinst in CSGFoundry
3. add QEvt::gatherLocalHit QEvt::gatherLocalHitMerged QEvt::gatherLocalPhoton
4. add kernel/functor using method called by the gatherLocal.. either:

   * "QEvt::PerLaunchLocalize" analogous to QEvt::PerLaunchMerge
   * "SU::localize_device_to_device_sphoton" analogous to SU::copy_if_device_to_device_presized_sphoton

     * device pointers managed by sevent.h

5. add method to do the transforms to sphoton (prefer not to use glm device side)
6. remove/disable CPU side localization


::

    1113 NP* QEvt::gatherHit_() const
    1114 {
    1115     LOG_IF(info, LIFECYCLE) ;
    1116
    1117     evt->hit = QU::device_alloc<sphoton>( evt->num_hit, "QEvt::gatherHit_:sphoton" );
    1118
    1119     SU::copy_if_device_to_device_presized_sphoton( evt->hit, evt->photon, evt->num_photon,  *photon_selector );
    1120
    1121     NP* hit = sphoton::zeros( evt->num_hit );
    1122
    1123     QU::copy_device_to_host<sphoton>( (sphoton*)hit->bytes(), evt->hit, evt->num_hit );
    1124
    1125     QU::device_free<sphoton>( evt->hit );
    1126
    1127     evt->hit = nullptr ;
    1128
    1129     LOG(LEVEL) << " hit.sstr " << hit->sstr() ;
    1130
    1131     return hit ;
    1132 }



QEvt::gatherLocalPhoton,gatherLocalHit,gatherLocalHitMerged variants of the below ?
-------------------------------------------------------------------------------------


::

    169     NP*      gatherPhoton() const ;
    170     NP*      gatherPhotonLite() const ;
    171
    172     NP*      gatherHit() const ;
    173     NP*      gatherHitLite() const ;
    174     NP*      gatherHitLiteMerged() const ;
    175     NP*      gatherHitMerged() const ;
    176


* mostly need : gatherLocalHit and gatherLocalHitMerged
* mainly debug : gatherLocalPhoton (many will not have iindex so no transforming needed)
* "Local" with "Lite" makes no sense : as no distinction between global and local for Lite photons


photonlocal currently only CPU side thing
-------------------------------------------

::

    (ok) A[blyth@localhost opticks]$ opticks-f photonlocal
    ./CSGOptiX/cxs_min_lite.py:    _pl = a.photonlocal
    ./CSGOptiX/cxs_min_lite.py:    _pl.shape   # non-standard photonlocal array
    ./sysrap/SComp.h:    static constexpr const char* PHOTONLOCAL_ = "photonlocal" ;
    ./sysrap/SEvt.cc:5. when "photon" array is configured for save and is present and "photonlocal" is also configured for save,
    ./sysrap/SEvt.cc:   derive "photonlocal" from "photon" and add to save_fold
    ./sysrap/SEvt.cc:    // 5. when "photon" array is configured for save and is present and "photonlocal" is also configured for save,
    ./sysrap/SEvt.cc:    //    derive "photonlocal" from "photon" and add to save_fold
    ./sysrap/SEvt.cc:        NP* photonlocal = localize_photon(photon, consistency_check);
    ./sysrap/SEvt.cc:        assert(photonlocal);
    ./sysrap/SEvt.cc:        save_fold->add(SComp::PHOTONLOCAL_, photonlocal );
    (ok) A[blyth@localhost opticks]$


SEvt::save tacks on CPU derived locals
----------------------------------------

::

    4483     // 4. when "hit" array is configured for save and is present and "hitlocal" is also configured for save,
    4484     //    derive "hitlocal" from "hit" and add to save_fold
    4485
    4486     const NP* hit = save_fold->get(SComp::HIT_);
    4487     if(hit && SEventConfig::HasSaveComp(SComp::HITLOCAL_))
    4488     {
    4489         bool consistency_check = true ;
    4490         NP* hitlocal = localize_photon(hit, consistency_check);
    4491         assert(hitlocal);
    4492         save_fold->add(SComp::HITLOCAL_, hitlocal );
    4493     }
    4494
    4495     // 5. when "photon" array is configured for save and is present and "photonlocal" is also configured for save,
    4496     //    derive "photonlocal" from "photon" and add to save_fold
    4497
    4498     const NP* photon = save_fold->get(SComp::PHOTON_);
    4499     if(photon && SEventConfig::HasSaveComp(SComp::PHOTONLOCAL_))
    4500     {
    4501         bool consistency_check = true ;
    4502         NP* photonlocal = localize_photon(photon, consistency_check);
    4503         assert(photonlocal);
    4504         save_fold->add(SComp::PHOTONLOCAL_, photonlocal );
    4505     }
    4506
    4507     // 6. when "extrafold" is defined add all extra_items arrays from it into save_fold




stree has the inst and iinst transforms in float and double
-------------------------------------------------------------

* inst are model2world
* iinst are world2model : so these are needed for localization

::

    0401     std::vector<glm::tmat4x4<double>> inst ;
     402     std::vector<glm::tmat4x4<float>>  inst_f4 ;
     403     std::vector<glm::tmat4x4<double>> iinst ;
     404     std::vector<glm::tmat4x4<float>>  iinst_f4 ;


    6071 inline const glm::tmat4x4<double>* stree::get_inst(int idx) const
    6072 {
    6073     return idx > -1 && idx < int(inst.size()) ? &inst[idx] : nullptr ;
    6074 }
    6075 inline const glm::tmat4x4<double>* stree::get_iinst(int idx) const
    6076 {
    6077     return idx > -1 && idx < int(iinst.size()) ? &iinst[idx] : nullptr ;
    6078 }
    6079
    6080 inline const glm::tmat4x4<float>* stree::get_inst_f4(int idx) const
    6081 {
    6082     return idx > -1 && idx < int(inst_f4.size()) ? &inst_f4[idx] : nullptr ;
    6083 }
    6084 inline const glm::tmat4x4<float>* stree::get_iinst_f4(int idx) const
    6085 {
    6086     return idx > -1 && idx < int(iinst_f4.size()) ? &iinst_f4[idx] : nullptr ;
    6087 }
    6088


CPU side localize uses iinst in double
---------------------------------------

::

    6804 inline void stree::localize_photon_inplace( sphoton& p ) const
    6805 {
    6806     unsigned iindex   = p.iindex() ;
    6807     assert( iindex != 0xffffffffu );
    6808     const glm::tmat4x4<double>* tr = get_iinst(iindex) ;
    6809     assert( tr );
    6810
    6811     bool normalize = true ;
    6812     p.transform( *tr, normalize );   // inplace transforms l (pos, mom, pol) into local frame
    6813
    6814 #ifdef NDEBUG
    6815 #else
    6816     unsigned sensor_identifier = p.pmtid() ;
    6817
    6818     glm::tvec4<int64_t> col3 = {} ;
    6819     strid::Decode( *tr, col3 );
    6820
    6821     sphit ht = {};
    6822     ht.iindex            = col3[0] ;
    6823     ht.sensor_identifier = col3[2] ;
    6824     ht.sensor_index      = col3[3] ;
    6825
    6826     assert( ht.iindex == iindex );
    6827     assert( ht.sensor_identifier == sensor_identifier );
    6828 #endif
    6829
    6830 }




CSGFoundry has inst (from st->inst_f4)
---------------------------------------

::

    597 void CSGImport::importInst()
    598 {
    599     fd->addInstanceVector( st->inst_f4 );
    600 }


    2150 void CSGFoundry::addInstanceVector( const std::vector<glm::tmat4x4<float>>& v_inst_f4 )
    2151 {
    2152     assert( inst.size() == 0 );
    2153     int num_inst = v_inst_f4.size() ;
    2154
    2155     for(int i=0 ; i < num_inst ; i++)
    2156     {
    2157         const glm::tmat4x4<float>& inst_f4 = v_inst_f4[i] ;
    2158         const float* tr16 = glm::value_ptr(inst_f4) ;
    2159         qat4 instance(tr16) ;
    2160         instance.incrementSensorIdentifier() ; // GPU side needs 0 to mean "not-a-sensor"
    2161         inst.push_back( instance );
    2162     }
    2163 }



But the upload doesnt include that because that info is passed in the optixInstance
------------------------------------------------------------------------------------

::

    3370 void CSGFoundry::upload()
    3371 {
    3372     inst_find_unique();
    3373
    3374     //LOG(LEVEL) << desc() ;
    3375
    3376     assert( tran.size() == itra.size() );
    3377
    3378
    3379 #ifdef WITH_CUDA
    3380     bool is_uploaded_0 = isUploaded();
    3381     LOG_IF(fatal, is_uploaded_0) << "HAVE ALREADY UPLOADED : THIS CANNOT BE DONE MORE THAN ONCE " ;
    3382     assert(is_uploaded_0 == false);
    3383
    3384     // allocates and copies
    3385     d_prim = prim.size() > 0 ? CU::UploadArray<CSGPrim>(prim.data(), prim.size() ) : nullptr ;
    3386     d_node = node.size() > 0 ? CU::UploadArray<CSGNode>(node.data(), node.size() ) : nullptr ;
    3387     d_plan = plan.size() > 0 ? CU::UploadArray<float4>(plan.data(), plan.size() ) : nullptr ;
    3388     d_itra = itra.size() > 0 ? CU::UploadArray<qat4>(itra.data(), itra.size() ) : nullptr ;
    3389
    3390     bool is_uploaded_1 = isUploaded();
    3391     LOG_IF(fatal, !is_uploaded_1) << "FAILED TO UPLOAD" ;
    3392     assert(is_uploaded_1 == true);
    3393 #else
    3394     LOG(fatal) << " COMPILATION WITH_CUDA required to upload " ;
    3395     std::exit(1);
    3396 #endif
    3397
    3398 }






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



From blyth-OJ-pmt-hit-type-2-for-muon-hits
----------------------------------------------

::

    438 void junoSD_PMT_v2_Opticks::EndOfEvent_CollectFullHits_premerged(int eventID, const SEvt* sev, const sphoton* hit, size_t num_hit )
    439 {
    440     SProf::Add("junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_premerged_HEAD");
    441     junoHit_PMT_Collection* hitCollection = m_jpmt->getHitCollection() ;
    442     assert( hitCollection );
    443
    444     for(size_t i=0 ; i < num_hit ; i++)
    445     {
    446         const sphoton& p = hit[i];
    447         sphoton l = p ;
    448         sev->localize_photon_inplace(l);
    449
    450         junoHit_PMT* hit = new junoHit_PMT();
    451         PopulateFullHit(hit, l, p );
    452         hitCollection->insert(hit);
    453     }
    454     std::string anno = SProf::Annotation("hit",num_hit);
    455     SProf::Add("junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_premerged_TAIL", anno.c_str());
    456 }


    5033 void SEvt::localize_photon_inplace( sphoton& p ) const
    5034 {
    5035     assert(tree);
    5036     tree->localize_photon_inplace(p);
    5037 }


    6791 /**
    6792 stree::localize_photon_inplace
    6793 --------------------------------
    6794
    6795 Argument photon is assumed to be a copy of a global frame photon.
    6796 This method transforms pos, mom, pol according to the transform
    6797 looked up from the iindex.
    6798
    6799 similar to SEvt::getLocalHit
    6800
    6801 **/
    6802
    6803
    6804 inline void stree::localize_photon_inplace( sphoton& p ) const
    6805 {
    6806     unsigned iindex   = p.iindex() ;
    6807     assert( iindex != 0xffffffffu );
    6808     const glm::tmat4x4<double>* tr = get_iinst(iindex) ;
    6809     assert( tr );
    6810
    6811     bool normalize = true ;
    6812     p.transform( *tr, normalize );   // inplace transforms l (pos, mom, pol) into local frame
    6813
    6814 #ifdef NDEBUG
    6815 #else
    6816     unsigned sensor_identifier = p.pmtid() ;
    6817
    6818     glm::tvec4<int64_t> col3 = {} ;
    6819     strid::Decode( *tr, col3 );
    6820
    6821     sphit ht = {};
    6822     ht.iindex            = col3[0] ;
    6823     ht.sensor_identifier = col3[2] ;
    6824     ht.sensor_index      = col3[3] ;
    6825
    6826     assert( ht.iindex == iindex );
    6827     assert( ht.sensor_identifier == sensor_identifier );
    6828 #endif
    6829
    6830 }






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



