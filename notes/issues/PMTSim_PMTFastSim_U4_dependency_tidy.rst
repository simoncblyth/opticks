PMTSim_PMTFastSim_U4_dependency_tidy
========================================

U4 depending on PMTSim/PMTFastSim should be 
reduced if possible. 

Perhaps using the new labelling `U4TrackInfo<spho>` 
to communicate between the FastSim and U4Recorder
can reduce the coupling. 



::

    215 /**
    216 U4VolumeMaker::PVF_
    217 ---------------------
    218 
    219 HMM: Coupling between U4 and PMTSim/PMTFastSim 
    220 should be reduced to a minimum. 
    221 
    222 Can it be reduced back to just Geant4 geometry access ? 
    223 
    224 Extending to JUNO specifics like junoPMTOpticalModel seems a step too far. 
    225 
    226 Also, more comfortable with PMTSim/PMTFastSim dependency 
    227 being restricted to test executables, keeping it out 
    228 of the U4 lib. Could do this with header-only U4 impls.  
    229 
    230 **/


    230 const G4VPhysicalVolume* U4VolumeMaker::PVF_(const char* name)
    231 {   
    232     const G4VPhysicalVolume* pv = nullptr ;
    233 #ifdef WITH_PMTFASTSIM
    234     bool has_manager_prefix = PMTFastSim::HasManagerPrefix(name) ;
    235     LOG(LEVEL) << "[ WITH_PMTFASTSIM name [" << name << "] has_manager_prefix " << has_manager_prefix ;
    236     if(has_manager_prefix)
    237     {   
    238         G4LogicalVolume* lv = PMTFastSim::GetLV(name) ; 
    239         junoPMTOpticalModel* pom = PMTFastSim::GetPMTOpticalModel(name);
    240         if(pom) PVF_POM = pom ; 
    241         LOG(LEVEL) << "PVF_POM  " << PVF_POM  ;
    242         
    243         LOG_IF(fatal, lv == nullptr ) << "PMTFastSim::GetLV returned nullptr for name [" << name << "]" ;
    244         assert( lv );
    245 
    246         pv = WrapRockWater( lv ) ;
    247     }



    epsilon:u4 blyth$ opticks-f junoPMTOpticalModel
    ./sysrap/SFastSimOpticalModel.hh:from the PMTFastSim junoPMTOpticalModel::DoIt into U4StepPoint 

    ./qudarap/qsim.h: based on https://juno.ihep.ac.cn/trac/browser/offline/trunk/Simulation/DetSimV2/PMTSim/src/junoPMTOpticalModel.cc 

    ./u4/tests/U4PMTFastSimTest.sh:   export junoPMTOpticalModel=INFO

    ./u4/tests/U4RecorderTest.h:class junoPMTOpticalModel ; 
    ./u4/tests/U4RecorderTest.h:    junoPMTOpticalModel*  fPOM ;     // just stays nullptr when do not have PMTFASTSIM_STANDALONE
    ./u4/tests/U4RecorderTest.h:    junoPMTOpticalModel* pom = U4VolumeMaker::PVF_POM ; 

    ./u4/U4StepPoint.cc:        // within junoPMTOpticalModel and access it here.

    ./u4/U4VolumeMaker.hh:class junoPMTOpticalModel ;
    ./u4/U4VolumeMaker.hh:    static       junoPMTOpticalModel* PVF_POM ;  // set by last PVF_ call giving non-null pom
    ./u4/U4VolumeMaker.cc:Also : is it really needed for junoPMTOpticalModel to 
    ./u4/U4VolumeMaker.cc:        junoPMTOpticalModel* pom = PMTFastSim::GetPMTOpticalModel(name); 
    ./u4/U4VolumeMaker.cc:junoPMTOpticalModel* U4VolumeMaker::PVF_POM = nullptr ; 
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 



Seems can remove POM from U4RecorderTest without issue::

    130 G4VPhysicalVolume* U4RecorderTest::Construct()
    131 {
    132     G4VPhysicalVolume* pv = const_cast<G4VPhysicalVolume*>(U4VolumeMaker::PV());  // sensitive to GEOM envvar 
    133 
    134     fPV = pv ;
    135 #ifdef PMTFASTSIM_STANDALONE
    136     junoPMTOpticalModel* pom = U4VolumeMaker::PVF_POM ;
    137     fPOM = pom ;
    138 #endif
    139 
    140     std::cout
    141         << "U4RecorderTest::Construct"
    142         << " fPV " << ( fPV ? "Y" : "N" )
    143         << " fPOM " << ( fPOM ? "Y" : "N" )
    144         << std::endl
    145         ;
    146 
    147     return pv ;
    148 }




Unexpected NA in history::

    280 U4Recorder::PostUserTrackingAction_Optical@330:  label.id   729 seq.desc_seqhis     8acbbcaaccccd nib 13 TO BT BT BT BT SR SR BT BR BR BT SR SA
     281 U4Recorder::PostUserTrackingAction_Optical@330:  label.id   728 seq.desc_seqhis              8ccd nib  4 TO BT BT SA
     282 U4Recorder::PostUserTrackingAction_Optical@330:  label.id   727 seq.desc_seqhis      8ccccaaccccd nib 12 TO BT BT BT BT SR SR BT BT BT BT SA
     283 U4Recorder::PostUserTrackingAction_Optical@330:  label.id   726 seq.desc_seqhis  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
     284 U4Recorder::PostUserTrackingAction_Optical@330:  label.id   725 seq.desc_seqhis              8ccd nib  4 TO BT BT SA


    U4Recorder::BeginOfEventAction@93: 
    U4Recorder::PostUserTrackingAction_Optical@330:  label.id   726 seq.desc_seqhis  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    U4Recorder::EndOfEventAction@94: 




Dumping the seqhis point by point with the rerun shows its a wraparound effect::

    epsilon:tests blyth$ grep SEvt::pointPhoton *.log
    SEvt::pointPhoton@1269:  label.id   726 bounce  0 seq.desc_seqhis                 0 nib  0  
    SEvt::pointPhoton@1269:  label.id   726 bounce  1 seq.desc_seqhis                 d nib  1 TO
    SEvt::pointPhoton@1269:  label.id   726 bounce  2 seq.desc_seqhis                cd nib  2 TO BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  3 seq.desc_seqhis               ccd nib  3 TO BT BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  4 seq.desc_seqhis              cccd nib  4 TO BT BT BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  5 seq.desc_seqhis             ccccd nib  5 TO BT BT BT BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  6 seq.desc_seqhis            accccd nib  6 TO BT BT BT BT SR
    SEvt::pointPhoton@1269:  label.id   726 bounce  7 seq.desc_seqhis           aaccccd nib  7 TO BT BT BT BT SR SR
    SEvt::pointPhoton@1269:  label.id   726 bounce  8 seq.desc_seqhis          caaccccd nib  8 TO BT BT BT BT SR SR BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  9 seq.desc_seqhis         bcaaccccd nib  9 TO BT BT BT BT SR SR BT BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 10 seq.desc_seqhis        bbcaaccccd nib 10 TO BT BT BT BT SR SR BT BR BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 11 seq.desc_seqhis       cbbcaaccccd nib 11 TO BT BT BT BT SR SR BT BR BR BT
    SEvt::pointPhoton@1269:  label.id   726 bounce 12 seq.desc_seqhis      acbbcaaccccd nib 12 TO BT BT BT BT SR SR BT BR BR BT SR
    SEvt::pointPhoton@1269:  label.id   726 bounce 13 seq.desc_seqhis     aacbbcaaccccd nib 13 TO BT BT BT BT SR SR BT BR BR BT SR SR
    SEvt::pointPhoton@1269:  label.id   726 bounce 14 seq.desc_seqhis    aaacbbcaaccccd nib 14 TO BT BT BT BT SR SR BT BR BR BT SR SR SR
    SEvt::pointPhoton@1269:  label.id   726 bounce 15 seq.desc_seqhis   caaacbbcaaccccd nib 15 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT
    SEvt::pointPhoton@1269:  label.id   726 bounce 16 seq.desc_seqhis  bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 17 seq.desc_seqhis  bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 18 seq.desc_seqhis  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 19 seq.desc_seqhis  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    epsilon:tests blyth$ 



