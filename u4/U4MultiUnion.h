#pragma once

#include <string>
#include <iomanip>
#include <sstream>

#include "G4MultiUnion.hh"

struct U4MultiUnion
{
    static std::string Desc(const G4MultiUnion* solid);
};

inline std::string U4MultiUnion::Desc(const G4MultiUnion* solid) // static
{
    G4String name = solid->GetName() ; // bizarre: G4VSolid::GetName returns by value, not reference
    unsigned sub_num = solid->GetNumberOfSolids() ;

    std::stringstream ss ;
    ss
       << "[U4MultiUnion::Desc\n"
       << " name " << name << "\n"
       << " sub_num " << sub_num << "\n"
       ;

    for( unsigned i=0 ; i < sub_num ; i++)
    {
        const G4VSolid* sub = solid->GetSolid(i);
        G4String sub_name = sub->GetName() ;
        ss << std::setw(4) << i << " : " << sub_name << "\n" ;
    }
    ss << "]U4MultiUnion::Desc\n" ;
    std::string str = ss.str() ;
    return str ;
}



