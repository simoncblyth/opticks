#include "PLOG.hh"
#include "CPhotonInfo.hh"

const plog::Severity CPhotonInfo::LEVEL = PLOG::EnvLevel("CPhotonInfo", "DEBUG") ; 

CPhotonInfo::CPhotonInfo(const CGenstep& gs, unsigned ix_, bool re_ )
    :   
    G4VUserTrackInformation("CPhotonInfo"),
    pho(gs.index, ix_, re_ )
{   
    LOG(LEVEL) ; 
}   

CPhotonInfo::~CPhotonInfo(){}
G4String* CPhotonInfo::type() const {   return pType ; }
std::string CPhotonInfo::desc() const { return pho.desc(); }

