#pragma once

#include <string>
#include "spho.h"
#include "G4VUserTrackInformation.hh"

struct U4PhotonInfo : public G4VUserTrackInformation
{
    spho pho ; 
    U4PhotonInfo(const spho& _pho); 
    std::string desc() const ; 
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



