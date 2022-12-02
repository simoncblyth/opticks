#pragma once
/**
U4UniformRand.h
=================

As this is headeronly and can benefit from a static UU 
include this header into a compilation unit with::

    #include "U4UniformRand.h"
    NP* U4UniformRand::UU = nullptr ;

And where appropriate set the UU to a reference array of randoms. 


Headers for random setup:

Randomize.hh
    Geant4 level setup that includes Randomize.h and does::

        #define G4UniformRand() CLHEP::HepRandom::getTheEngine()->flat()
Randomize.h
    CLHEP level setup


Tried to avoid duplication between this and SUniformRand.h using::

    #include "Randomize.hh"
    typedef SUniformRand<CLHEP::HepRandom> U4UniformRand ;

It kinda works, see sysrap/tests/SUniformRand_test.cc, but runs 
into duplicate symbol issues.  So are simply duplicating. 

Actually maybe easier to get rid of U4UniformRand::UU and use some 
another symbol like SEvt::UU (or SLOG::UU) to hold the randoms as 
that can always be made accessible : and avoid duplicate symbols
troubles ?
 
**/

#include <string>
#include <iomanip>
#include "Randomize.hh"
#include "NP.hh"

struct U4UniformRand
{
    static NP* UU ; 
    static constexpr const double EPSILON = 1e-6 ; 

    static double One(); 
    static void Burn(int n); 

    static void Get(std::vector<double>& uu); 
    static std::string Desc(int n=10); 

    static NP* Get(int n=1000); 
    static int Find(double u, const NP* uu=UU ) ; 
    static std::string Desc(double u, const NP* uu=UU ) ; 
}; 

inline double U4UniformRand::One() // static
{
    return G4UniformRand();  // ONLY dependency on Geant4/CLHEP is here 
}
inline void U4UniformRand::Burn(int n) // static
{
    for(int i=0 ; i < n ; i++) One();
}
inline void U4UniformRand::Get(std::vector<double>& uu ) // static
{
    for(unsigned i=0 ; i < uu.size() ; i++) uu[i] = One();
}
inline std::string U4UniformRand::Desc(int n )
{
    std::vector<double> uu(n) ; 
    Get(uu); 
    std::stringstream ss ; 
    ss << "U4UniformRand::Desc" << std::endl ; 
    for(int i=0 ; i < n ; i++)
    {
        ss << std::setw(6) << i 
           << " " << std::setw(10) << std::fixed << std::setprecision(5) << uu[i] 
           << std::endl 
           ;
    }
    std::string s = ss.str(); 
    return s; 
}
inline NP* U4UniformRand::Get(int n)
{
    std::vector<double> uu(n) ; 
    Get(uu); 
    return NP::Make<double>(uu) ; 
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

