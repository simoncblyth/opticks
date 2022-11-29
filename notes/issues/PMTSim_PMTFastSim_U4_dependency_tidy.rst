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
    SEvt::pointPhoton@1269:  label.id   726 bounce  0 ctx.p.flag TO seq.desc_seqhis                 0 nib  0  
    SEvt::pointPhoton@1269:  label.id   726 bounce  1 ctx.p.flag BT seq.desc_seqhis                 d nib  1 TO
    SEvt::pointPhoton@1269:  label.id   726 bounce  2 ctx.p.flag BT seq.desc_seqhis                cd nib  2 TO BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  3 ctx.p.flag BT seq.desc_seqhis               ccd nib  3 TO BT BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  4 ctx.p.flag BT seq.desc_seqhis              cccd nib  4 TO BT BT BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  5 ctx.p.flag SR seq.desc_seqhis             ccccd nib  5 TO BT BT BT BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  6 ctx.p.flag SR seq.desc_seqhis            accccd nib  6 TO BT BT BT BT SR
    SEvt::pointPhoton@1269:  label.id   726 bounce  7 ctx.p.flag BT seq.desc_seqhis           aaccccd nib  7 TO BT BT BT BT SR SR
    SEvt::pointPhoton@1269:  label.id   726 bounce  8 ctx.p.flag BR seq.desc_seqhis          caaccccd nib  8 TO BT BT BT BT SR SR BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  9 ctx.p.flag BR seq.desc_seqhis         bcaaccccd nib  9 TO BT BT BT BT SR SR BT BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 10 ctx.p.flag BT seq.desc_seqhis        bbcaaccccd nib 10 TO BT BT BT BT SR SR BT BR BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 11 ctx.p.flag SR seq.desc_seqhis       cbbcaaccccd nib 11 TO BT BT BT BT SR SR BT BR BR BT
    SEvt::pointPhoton@1269:  label.id   726 bounce 12 ctx.p.flag SR seq.desc_seqhis      acbbcaaccccd nib 12 TO BT BT BT BT SR SR BT BR BR BT SR
    SEvt::pointPhoton@1269:  label.id   726 bounce 13 ctx.p.flag SR seq.desc_seqhis     aacbbcaaccccd nib 13 TO BT BT BT BT SR SR BT BR BR BT SR SR
    SEvt::pointPhoton@1269:  label.id   726 bounce 14 ctx.p.flag BT seq.desc_seqhis    aaacbbcaaccccd nib 14 TO BT BT BT BT SR SR BT BR BR BT SR SR SR
    SEvt::pointPhoton@1269:  label.id   726 bounce 15 ctx.p.flag BR seq.desc_seqhis   caaacbbcaaccccd nib 15 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT
    SEvt::pointPhoton@1269:  label.id   726 bounce 16 ctx.p.flag BT seq.desc_seqhis  bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 17 ctx.p.flag SR seq.desc_seqhis  bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 18 ctx.p.flag BT seq.desc_seqhis  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 19 ctx.p.flag SA seq.desc_seqhis  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    epsilon:tests blyth$ 


Also looks like getting repeated flag at FastSim/SlowSim transitions ? 
NO its not, its just the BT across the fake boundary leading to more. 

Reproduce the misbehavior in sseq_test::

    epsilon:tests blyth$ name=sseq_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name
                   TORCH :                 d nib  1 TO
       BOUNDARY_TRANSMIT :                cd nib  2 TO BT
       BOUNDARY_TRANSMIT :               ccd nib  3 TO BT BT
       BOUNDARY_TRANSMIT :              cccd nib  4 TO BT BT BT
       BOUNDARY_TRANSMIT :             ccccd nib  5 TO BT BT BT BT
        SURFACE_SREFLECT :            accccd nib  6 TO BT BT BT BT SR
        SURFACE_SREFLECT :           aaccccd nib  7 TO BT BT BT BT SR SR
       BOUNDARY_TRANSMIT :          caaccccd nib  8 TO BT BT BT BT SR SR BT
        BOUNDARY_REFLECT :         bcaaccccd nib  9 TO BT BT BT BT SR SR BT BR
        BOUNDARY_REFLECT :        bbcaaccccd nib 10 TO BT BT BT BT SR SR BT BR BR
       BOUNDARY_TRANSMIT :       cbbcaaccccd nib 11 TO BT BT BT BT SR SR BT BR BR BT
        SURFACE_SREFLECT :      acbbcaaccccd nib 12 TO BT BT BT BT SR SR BT BR BR BT SR
        SURFACE_SREFLECT :     aacbbcaaccccd nib 13 TO BT BT BT BT SR SR BT BR BR BT SR SR
        SURFACE_SREFLECT :    aaacbbcaaccccd nib 14 TO BT BT BT BT SR SR BT BR BR BT SR SR SR
       BOUNDARY_TRANSMIT :   caaacbbcaaccccd nib 15 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT
        BOUNDARY_REFLECT :  bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
       BOUNDARY_TRANSMIT :  bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
        SURFACE_SREFLECT :  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
       BOUNDARY_TRANSMIT :  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
          SURFACE_ABSORB :  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    epsilon:tests blyth$ 



* The wraparound is from shifting beyond the width of the type. 
* And getting NA arises from OR-ing of different flags together. 

  
Need to widen sseq storage adopting the techniques used in stag.h 
to write the nibbles. 
   
Writing for both GPU and CPU is done via::

    076 SCTX_METHOD void sctx::point(int bounce)
     77 {
     78     if(evt->record && bounce < evt->max_record) evt->record[evt->max_record*idx+bounce] = p ;
     79     if(evt->rec    && bounce < evt->max_rec)    evt->add_rec( rec, idx, bounce, p );    // this copies into evt->rec array 
     80     if(evt->seq    && bounce < evt->max_seq)    seq.add_nibble( bounce, p.flag(), p.boundary() );
     81 }


    114 SSEQ_METHOD void sseq::add_nibble(unsigned slot, unsigned flag, unsigned boundary )
    115 {
    116     seqhis |=  (( FFS(flag) & 0xfull ) << 4*slot );
    117     seqbnd |=  (( boundary  & 0xfull ) << 4*slot );
    118     // 0xfull is needed to avoid all bits above 32 getting set
    119     // NB: nibble restriction of each "slot" means there is absolute no need for FFSLL
    120 }


Reworked sseq.h to hold NSEQ elements following stag.h example.

This fixes overwriting, increasing sseq recording to not overwrite up to maxbounce 32::

    epsilon:tests blyth$ name=sseq_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name
    test_desc_seqhis_1
                   TORCH :                 0                d nib  1 TO
       BOUNDARY_TRANSMIT :                 0               cd nib  2 TO BT
       BOUNDARY_TRANSMIT :                 0              ccd nib  3 TO BT BT
       BOUNDARY_TRANSMIT :                 0             cccd nib  4 TO BT BT BT
       BOUNDARY_TRANSMIT :                 0            ccccd nib  5 TO BT BT BT BT
        SURFACE_SREFLECT :                 0           accccd nib  6 TO BT BT BT BT SR
        SURFACE_SREFLECT :                 0          aaccccd nib  7 TO BT BT BT BT SR SR
       BOUNDARY_TRANSMIT :                 0         caaccccd nib  8 TO BT BT BT BT SR SR BT
        BOUNDARY_REFLECT :                 0        bcaaccccd nib  9 TO BT BT BT BT SR SR BT BR
        BOUNDARY_REFLECT :                 0       bbcaaccccd nib 10 TO BT BT BT BT SR SR BT BR BR
       BOUNDARY_TRANSMIT :                 0      cbbcaaccccd nib 11 TO BT BT BT BT SR SR BT BR BR BT
        SURFACE_SREFLECT :                 0     acbbcaaccccd nib 12 TO BT BT BT BT SR SR BT BR BR BT SR
        SURFACE_SREFLECT :                 0    aacbbcaaccccd nib 13 TO BT BT BT BT SR SR BT BR BR BT SR SR
        SURFACE_SREFLECT :                 0   aaacbbcaaccccd nib 14 TO BT BT BT BT SR SR BT BR BR BT SR SR SR
       BOUNDARY_TRANSMIT :                 0  caaacbbcaaccccd nib 15 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT
        BOUNDARY_REFLECT :                 0 bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
       BOUNDARY_TRANSMIT :                 c bcaaacbbcaaccccd nib 17 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR BT
        SURFACE_SREFLECT :                ac bcaaacbbcaaccccd nib 18 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR BT SR
       BOUNDARY_TRANSMIT :               cac bcaaacbbcaaccccd nib 19 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR BT SR BT
          SURFACE_ABSORB :              8cac bcaaacbbcaaccccd nib 20 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR BT SR BT SA
    epsilon:tests blyth$ 







