#pragma once

#include <string>
#include <cstring>
#include <sstream>
#include <iomanip>

#include "geomdefs.hh"
#include "G4ThreeVector.hh"

struct sfmt
{
    static std::string Format(double v, int w=10) ; 
    static std::string Format(const G4ThreeVector* v); 
}; 


inline std::string sfmt::Format(double v, int w)
{
    std::stringstream ss ;    
    if( v == kInfinity ) 
    {   
        ss << std::setw(w) << "kInfinity" ; 
    }   
    else
    {   
        ss << std::setw(w) << std::fixed << std::setprecision(4) << v ; 
    }   
    std::string str = ss.str(); 
    return str ; 
}

inline std::string sfmt::Format(const G4ThreeVector* v)
{
    std::stringstream ss ;    
    ss << *v ; 
    std::string str = ss.str(); 
    return str ; 
}


