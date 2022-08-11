#pragma once

#include <string>
#include <sstream>
#include <iomanip>

#include "G4VPhysicalVolume.hh"
#include "G4TouchableHistory.hh"
#include "G4VSolid.hh"

struct U4Touchable
{
    static std::string Desc(const G4VTouchable* touch);
};

inline std::string U4Touchable::Desc(const G4VTouchable* touch)
{
    int depth = touch->GetHistoryDepth();

    std::stringstream ss ; 
    ss << "U4Touchable::Desc depth " << depth << std::endl ; 
    for(int i=0 ; i<depth ; i++) 
    { 
        G4VPhysicalVolume* pv = touch->GetVolume(i); 
        G4VSolid* so = touch->GetSolid(i); 
        G4int cp = touch->GetReplicaNumber(i); 

        ss << " i " << std::setw(2) << i 
           << " cp " << std::setw(6)  << cp
           << " so " << std::setw(40) << so->GetName()
           << " pv " << std::setw(60) << pv->GetName()
           << std::endl ; 
    }   
    std::string s = ss.str(); 
    return s ; 
}

