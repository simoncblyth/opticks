#pragma once

#include <string>
#include <sstream>
#include "G4Version.hh"

struct U4Version
{
    static std::string Desc(); 
}; 

inline std::string U4Version::Desc()
{
    std::stringstream ss ; 
    ss << G4Version << " " << G4Date ; 
    std::string str = ss.str(); 
    return str ; 
}
