#pragma once
/**
U4PhotonInfo.h
================

Carries spho.h photon label, which comprises 4 int indices. 

**/

#include <string>
#include "spho.h"
#include "G4Track.hh"
#include "G4VUserTrackInformation.hh"

struct U4PhotonInfo : public G4VUserTrackInformation
{
    spho pho ; 
    U4PhotonInfo(const spho& _pho); 
    std::string desc() const ; 

    static bool Exists(const G4Track* track); 
    static spho Get(const G4Track* track); 
    static int GetIndex(const  G4Track* track);
    static void Set(G4Track* track, const spho& pho_ ); 
};

inline U4PhotonInfo::U4PhotonInfo(const spho& _pho )
    :   
    G4VUserTrackInformation("U4PhotonInfo"),
    pho(_pho)
{
}
 
inline std::string U4PhotonInfo::desc() const 
{
    std::stringstream ss ; 
    ss << *pType << " " << pho.desc() ; 
    std::string s = ss.str(); 
    return s ; 
}

inline bool U4PhotonInfo::Exists(const G4Track* track)
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    U4PhotonInfo* pin = ui ? dynamic_cast<U4PhotonInfo*>(ui) : nullptr ;
    return pin != nullptr ; 
}
inline spho U4PhotonInfo::Get(const G4Track* track)
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    U4PhotonInfo* pin = ui ? dynamic_cast<U4PhotonInfo*>(ui) : nullptr ;
    return pin ? pin->pho : spho::Placeholder() ; 
}

inline int U4PhotonInfo::GetIndex(const G4Track* track)
{
    spho label = Get(track); 
    return label.id ;  
}

inline void U4PhotonInfo::Set(G4Track* track, const spho& pho_ )
{
    track->SetUserInformation(new U4PhotonInfo(pho_)); 
}

