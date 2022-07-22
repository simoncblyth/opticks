#pragma once

#include <sstream>
#include <string>
#include "G4ThreeVector.hh"

struct U4ThreeVector
{
    static std::string Desc(const G4ThreeVector& v ); 
}; 

inline std::string U4ThreeVector::Desc(const G4ThreeVector& v )
{
    std::stringstream ss ; 
    ss << "U4ThreeVector::Desc"
       << "(" 
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << v.x() 
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << v.y() 
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << v.z() 
       << ")"
       ;
    std::string s = ss.str(); 
    return s ; 
}
