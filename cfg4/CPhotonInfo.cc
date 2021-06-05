#include "PLOG.hh"
#include "G4Track.hh"
#include "CPhotonInfo.hh"

const plog::Severity CPhotonInfo::LEVEL = PLOG::EnvLevel("CPhotonInfo", "DEBUG") ; 

CPhotonInfo::CPhotonInfo(const CGenstep& gs, unsigned ix_, unsigned gn_, int id_ )
    :   
    G4VUserTrackInformation("CPhotonInfo"),
    pho(gs.index, ix_, id_ < 0 ? gs.offset + ix_ : id_ , gn_ )
{   
    LOG(LEVEL) ; 
}   

CPhotonInfo::~CPhotonInfo(){}
G4String* CPhotonInfo::type() const {   return pType ; }
std::string CPhotonInfo::desc() const { return pho.desc(); }

unsigned CPhotonInfo::gs() const { return pho.gs ; } // 0-based genstep index within the event
unsigned CPhotonInfo::ix() const { return pho.ix ; } // 0-based photon index within the genstep
unsigned CPhotonInfo::id() const { return pho.id ; } // 0-based absolute photon identity index within the event 
unsigned CPhotonInfo::gn() const { return pho.gn ; } // 0-based generation index incremented at each reemission  


CPho CPhotonInfo::Get(const G4Track* optical_photon_track)   // static 
{
    G4VUserTrackInformation* ui = optical_photon_track->GetUserInformation() ;
    CPhotonInfo* cpui = ui ? dynamic_cast<CPhotonInfo*>(ui) : nullptr ; 
    CPho pho ; 
    if(cpui)
    {
        pho = cpui->pho ; 
    }
    return pho  ; 
}

