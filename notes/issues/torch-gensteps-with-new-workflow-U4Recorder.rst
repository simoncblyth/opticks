torch-gensteps-with-new-workflow-U4Recorder
==============================================

* from :doc:`reimplement-G4OpticksRecorder-CManager-for-new-workflow`


U4Recorder labelling the unlabelled
-------------------------------------

It would be better to label in PrimaryGeneratorAction, but 
when that is not convenient can use standin labelling

Actually there are no G4Track accessible from PrimaryGeneratorAction 
so no way to label. 


::

    080 void U4Recorder::PreUserTrackingAction_Optical(const G4Track* track)
     81 {
     82     spho label = U4Track::Label(track);  // just label, not sphoton 
     83 
     84     if( label.isDefined() == false )
     85     {
     86         int trackID = track->GetTrackID() - 1 ;
     87         assert( trackID >= 0 );
     88 
     89         spho fab = spho::Fabricate(trackID);
     90 
     91         G4Track* _track = const_cast<G4Track*>(track);
     92         U4PhotonInfo::Set(_track, fab );
     93         label = U4Track::Label(track);
     94 
     95         LOG(info) << " labelling photon " << label.desc() ;
     96     }
     97 
     98     assert( label.isDefined() );
     99     SEvt* sev = SEvt::Get();
    100 
    101     if(label.gn == 0)
    102     {
    103         sev->beginPhoton(label);
    104     }
    105     else if( label.gn > 0 )
    106     {
    107         sev->continuePhoton(label);
    108     }
    109 }



Review old workflow torch gensteps
-------------------------------------

Abstract base::

    class CFG4_API CSource : public G4VPrimaryGenerator

    virtual void GeneratePrimaryVertex(G4Event *evt) = 0 ;  

::

    [blyth@localhost cfg4]$ grep public\ CSource *.hh
    CGenstepSource.hh     :class CFG4_API CGenstepSource: public CSource
    CGunSource.hh         :class CFG4_API CGunSource: public CSource
    CInputPhotonSource.hh :class CFG4_API CInputPhotonSource: public CSource
    CPrimarySource.hh     :class CFG4_API CPrimarySource: public CSource
    CTorchSource.hh       :class CFG4_API CTorchSource: public CSource


CTorchSource
    implements generation of primaries parametrized by the torch genstep 


HMM: there is lots of code for different torch genstep types, 
but only really need one form initially

 
SEvt::AddTorchGenstep as used by CSGOptiX/tests/CXRaindropTest.cc
---------------------------------------------------------------------

::
   
    93 void SEvt::AddTorchGenstep(){   AddGenstep(SEvent::MakeTorchGensteps());   }

    47 NP* SEvent::MakeTorchGensteps(){    return MakeGensteps( OpticksGenstep_TORCH ) ; }

    052 NP* SEvent::MakeGensteps( int gentype )
     53 {
     54     unsigned num_gs = 1 ;
     55     NP* gs = NP::Make<float>(num_gs, 6, 4 );
     56     switch(gentype)
     57     {
     58         case  OpticksGenstep_TORCH:         FillGensteps<storch>(   gs, 100) ; break ;
     59         case  OpticksGenstep_CERENKOV:      FillGensteps<scerenkov>(gs, 100) ; break ;
     60         case  OpticksGenstep_SCINTILLATION: FillGensteps<sscint>(   gs, 100) ; break ;
     61         case  OpticksGenstep_CARRIER:       FillGensteps<scarrier>( gs, 10)  ; break ;
     62     }
     63     return gs ;
     64 }

    066 template<typename T>
     67 void SEvent::FillGensteps( NP* gs, unsigned numphoton_per_genstep )
     68 {
     69     T* tt = (T*)gs->bytes() ;
     70     for(int i=0 ; i < gs->shape[0] ; i++ ) T::FillGenstep( tt[i], i, numphoton_per_genstep ) ;
     71 }

    113 #if defined(__CUDACC__) || defined(__CUDABE__)
    114 #else
    115 inline void storch::FillGenstep( storch& gs, unsigned genstep_id, unsigned numphoton_per_genstep )
    116 {
    117     float3 mom = make_float3( 0.f, 0.f, 1.f );
    118 
    119     gs.gentype = OpticksGenstep_TORCH ;
    120     gs.wavelength = 501.f ;
    121     gs.mom = normalize(mom);
    122     gs.radius = 50.f ;
    123     gs.pos = make_float3( 0.f, 0.f, -90.f );
    124     gs.time = 0.f ;
    125     gs.zenith = make_float2( 0.f, 1.f );
    126     gs.azimuth = make_float2( 0.f, 1.f );
    127     gs.type = storchtype::Type("disc");
    128     gs.mode = 255 ;    //torchmode::Type("...");  
    129     gs.numphoton = numphoton_per_genstep  ;
    130 }
    131 

::

    1347 inline QSIM_METHOD void qsim::generate_photon(sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    1348 {
    1349     quad4& q = (quad4&)p ;
    1350     const int& gencode = gs.q0.i.x ;
    1351 
    1352     switch(gencode)
    1353     {
    1354         case OpticksGenstep_CARRIER:         scarrier::generate(     q, rng, gs, photon_id, genstep_id)  ; break ;
    1355         case OpticksGenstep_TORCH:           storch::generate(       p, rng, gs, photon_id, genstep_id ) ; break ;
    1356         case OpticksGenstep_CERENKOV:        cerenkov->generate(     p, rng, gs, photon_id, genstep_id ) ; break ;
    1357         case OpticksGenstep_SCINTILLATION:   scint->generate(        p, rng, gs, photon_id, genstep_id ) ; break ;
    1358         default:                             generate_photon_dummy(  q, rng, gs, photon_id, genstep_id)  ; break ;
    1359     }
    1360 }




torch generation within Geant4 PrimaryGeneratorAction ?
----------------------------------------------------------

sysrap/tests/storch_test.cc uses MOCK_CURAND to generate torch photons on CPU


Doing similar in SGenerate.h and u4/U4VPrimaryGenerator.h::

    epsilon:sysrap blyth$ o
    On branch master
    Your branch is up-to-date with 'origin/master'.

    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git checkout -- <file>..." to discard changes in working directory)

        modified:   CSGOptiX/CSGOptiX7.cu
        modified:   notes/issues/reimplement-G4OpticksRecorder-CManager-for-new-workflow.rst
        modified:   qudarap/qsim.h
        modified:   sysrap/CMakeLists.txt
        modified:   sysrap/SEvt.cc
        modified:   sysrap/SFrameGenstep.cc
        modified:   sysrap/SGenstep.cc
        modified:   sysrap/SGenstep.hh
        modified:   sysrap/scarrier.h
        modified:   sysrap/sevent.h
        modified:   sysrap/sseq.h
        modified:   sysrap/storch.h
        modified:   u4/CMakeLists.txt
        modified:   u4/U4.cc
        modified:   u4/tests/CMakeLists.txt

    Untracked files:
      (use "git add <file>..." to include in what will be committed)

        notes/issues/torch-gensteps-with-new-workflow-U4Recorder.rst
        sysrap/SGenerate.h
        sysrap/tests/SGenerate_test.cc
        sysrap/tests/SGenerate_test.py
        sysrap/tests/SGenerate_test.sh
        u4/U4VPrimaryGenerator.h
        u4/tests/U4VPrimaryGeneratorTest.cc

    no changes added to commit (use "git add" and/or "git commit -a")
    epsilon:opticks blyth$ 













