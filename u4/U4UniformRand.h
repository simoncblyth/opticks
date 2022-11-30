#pragma once
/**
U4UniformRand.h
=================

As this is headeronly and can benefit from a static UU 
include this header into a compilation unit with::

    #include "U4UniformRand.h"
    NP* U4UniformRand::UU = nullptr ;

And where appropriate set the UU to a reference array of randoms. 

**/

#include <string>
#include <iomanip>
#include "G4Types.hh"
#include "Randomize.hh"
#include "NP.hh"

struct U4UniformRand
{
    static NP* UU ; 
    static constexpr const double EPSILON = 1e-6 ; 
    static std::string Desc(int n=10); 
    static NP* Get(int n=1000); 
    static int Find(double u, const NP* uu=UU ) ; 
    static std::string Desc(double u, const NP* uu=UU ) ; 
}; 

inline std::string U4UniformRand::Desc(int n )
{
    std::stringstream ss ; 
    ss << "U4UniformRand::Desc" << std::endl ; 
    for(int i=0 ; i < n ; i++)
    {
        G4double u = G4UniformRand(); 
        ss << std::setw(6) << i 
           << " " << std::setw(10) << std::fixed << std::setprecision(5) << u 
           << std::endl 
           ;
    }
    std::string s = ss.str(); 
    return s; 
}


inline NP* U4UniformRand::Get(int n)
{
    NP* u = NP::Make<double>(n); 
    double* uu = u->values<double>(); 
    for(int i=0 ; i < n ; i++) uu[i] = G4UniformRand();
    return u ; 
}

inline int U4UniformRand::Find(double u, const NP* uu)
{
    return uu ? uu->find_value_index(u, EPSILON) : -2 ; 
}

inline std::string U4UniformRand::Desc(double u, const NP* uu)
{
    std::stringstream ss ; 
    ss << "UU[" 
       << std::setw(7) << std::fixed << std::setprecision(5) << u 
       << " " 
       << std::setw(6) << Find(u, uu) 
       << "]"
       ;  
    std::string s = ss.str(); 
    return s; 
}

