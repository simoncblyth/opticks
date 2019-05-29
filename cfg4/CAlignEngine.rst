CAlignEngine
===============


Usage
-------

::

    [blyth@localhost opticks]$ opticks-fl CAlignEngine::SetSequenceIndex | grep -v CAlignEngine
    ./cfg4/CCerenkovGenerator.hh
    ./cfg4/CCerenkovGenerator.cc
    ./examples/Geant4/CerenkovMinimal/ckm.bash
    ./g4ok/G4Opticks.cc   // from G4Opticks::setAlignIndex


G4Opticks::setAlignIndex
--------------------------


::

    blyth@localhost opticks]$ opticks-f setAlignIndex
    ./cfg4/CAlignEngine.hh:    ./g4ok/G4Opticks.cc   // from G4Opticks::setAlignIndex
    ./examples/Geant4/CerenkovMinimal/Ctx.cc:    G4Opticks::GetOpticks()->setAlignIndex(_record_id);
    ./examples/Geant4/CerenkovMinimal/Ctx.cc:    G4Opticks::GetOpticks()->setAlignIndex(-1);
    ./examples/Geant4/CerenkovMinimal/L4Cerenkov.cc:        G4Opticks::GetOpticks()->setAlignIndex(record_id); 
    ./examples/Geant4/CerenkovMinimal/L4Cerenkov.cc:        G4Opticks::GetOpticks()->setAlignIndex(-1); 
    ./g4ok/G4Opticks.hh:        void setAlignIndex(int align_idx) const ; 
    ./g4ok/G4Opticks.cc:void G4Opticks::setAlignIndex(int align_idx) const 



examples/Geant4/CerenkovMinimal/Ctx.cc::

    103 void Ctx::setTrackOptical(const G4Track* track)
    104 {
    105     const_cast<G4Track*>(track)->UseGivenVelocity(true);
    106 
    107 #ifdef WITH_OPTICKS
    108     CTrackInfo* info=dynamic_cast<CTrackInfo*>(track->GetUserInformation());
    109     assert(info) ;
    110     _record_id = info->photon_record_id ;
    111     G4Opticks::GetOpticks()->setAlignIndex(_record_id);
    112 #endif
    113 }
    114 
    115 void Ctx::postTrackOptical(const G4Track* track)
    116 {
    117 #ifdef WITH_OPTICKS
    118     CTrackInfo* info=dynamic_cast<CTrackInfo*>(track->GetUserInformation());
    119     assert(info) ;
    120     assert( _record_id == info->photon_record_id ) ;
    121     G4Opticks::GetOpticks()->setAlignIndex(-1);
    122 #endif
    123 }



