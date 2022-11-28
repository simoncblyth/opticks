#pragma once
/**
Deprecated_U4PhotonInfo.h
=============================

Carries spho.h photon label, which comprises 4 int indices. 

This is replaced by the templated U4TrackInfo which 
maintains more separation between Geant4 TrackInfo 
mechanics and the details of the info carried in the label. 


**/

#include <string>
#include "spho.h"
#include "G4Track.hh"
#include "G4VUserTrackInformation.hh"

struct Deprecated_U4PhotonInfo : public G4VUserTrackInformation
{
    spho pho ; 

    Deprecated_U4PhotonInfo(const spho& _pho); 
    std::string desc() const ; 

    static bool Exists(const G4Track* track); 
    static spho  Get(   const G4Track* track);   // by value 
    static spho* GetRef(const G4Track* track);   // by reference, allowing inplace changes

    static int GetIndex(const  G4Track* track);
    static void Set(G4Track* track, const spho& pho_ ); 
};

inline Deprecated_U4PhotonInfo::Deprecated_U4PhotonInfo(const spho& _pho )
    :   
    G4VUserTrackInformation("Deprecated_U4PhotonInfo"),
    pho(_pho)
{
}
 
inline std::string Deprecated_U4PhotonInfo::desc() const 
{
    std::stringstream ss ; 
    ss << *pType << " " << pho.desc() ; 
    std::string s = ss.str(); 
    return s ; 
}

inline bool Deprecated_U4PhotonInfo::Exists(const G4Track* track)
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    Deprecated_U4PhotonInfo* pin = ui ? dynamic_cast<Deprecated_U4PhotonInfo*>(ui) : nullptr ;
    return pin != nullptr ; 
}

inline spho Deprecated_U4PhotonInfo::Get(const G4Track* track) // by value 
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    Deprecated_U4PhotonInfo* pin = ui ? dynamic_cast<Deprecated_U4PhotonInfo*>(ui) : nullptr ;
    return pin ? pin->pho : spho::Placeholder() ; 
}

inline spho* Deprecated_U4PhotonInfo::GetRef(const G4Track* track) // by value 
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    Deprecated_U4PhotonInfo* pin = ui ? dynamic_cast<Deprecated_U4PhotonInfo*>(ui) : nullptr ;
    return pin ? &(pin->pho) : nullptr ; 
}

inline int Deprecated_U4PhotonInfo::GetIndex(const G4Track* track)
{
    spho label = Get(track); 
    return label.id ;  
}

inline void Deprecated_U4PhotonInfo::Set(G4Track* track, const spho& pho_ )
{
    track->SetUserInformation(new Deprecated_U4PhotonInfo(pho_)); 
    // hmm: seems expensive if already has the label 
}

