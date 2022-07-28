#pragma once

#include "scuda.h"

#include <sstream>
#include <string>
#include "G4ThreeVector.hh"


struct U4ThreeVector
{
    static void FromFloat3( G4ThreeVector& v, const float3& f ); 
    static std::string Desc(const G4ThreeVector& v ); 
}; 

inline void U4ThreeVector::FromFloat3(G4ThreeVector& v, const float3& f )
{
    v.set(double(f.x), double(f.y), double(f.z)); 
}

inline std::string U4ThreeVector::Desc(const G4ThreeVector& v )
{
    std::stringstream ss ; 
    ss 
       << "(" 
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << v.x() 
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << v.y() 
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << v.z() 
       << ")"
       ;
    std::string s = ss.str(); 
    return s ; 
}
