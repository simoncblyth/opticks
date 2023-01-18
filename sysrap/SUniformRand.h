#pragma once
/**
SUniformRand.h
=================

This adapts u4/U4UniformRand.h to try to get rid of Geant4 dependency. 

As this is headeronly and can benefit from a static UU 
include this header into a compilation unit with::

    #include "SUniformRand.h"
    NP* SUniformRand::UU = nullptr ;

And where appropriate set the UU to a reference array of randoms. 


Headers for random setup:

Randomize.hh
    Geant4 level setup that includes Randomize.h and does::

        #define G4UniformRand() CLHEP::HepRandom::getTheEngine()->flat()

Randomize.h
    CLHEP level setup

Random.h
    CLHEP::HepRandom class with static method CLHEP::HepRandom::getTheEngine()


g4-cls Randomize
g4-cls Random



**/

#include <string>
#include <iomanip>
#include "Randomize.hh"
#include "NPX.h"

template<typename E>
struct SUniformRand
{
    // static NP* UU ;  // often causes symbol issues, maybe easier to just pass SEvt::UU as argument 
    static constexpr const double EPSILON = 1e-6 ; 
    static std::string Desc(int n=10); 
    static void Get(std::vector<double>& uu); 
    static NP* Get(int n=1000); 
    static int Find(double u, const NP* uu ) ; 
    static std::string Desc(double u, const NP* uu ) ; 
}; 


template<typename E>
inline void SUniformRand<E>::Get(std::vector<double>& uu ) // static
{
    unsigned n = uu.size(); 
    for(unsigned i=0 ; i < n ; i++) uu[i] = E::getTheEngine()->flat() ; 
}

template<typename E>
inline std::string SUniformRand<E>::Desc(int n )
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


template<typename E>
inline NP* SUniformRand<E>::Get(int n)
{
    std::vector<double> uu(n) ; 
    Get(uu); 
    return NPX::Make<double>(uu) ; 
}

template<typename E>
inline int SUniformRand<E>::Find(double u, const NP* uu)
{
    return uu ? uu->find_value_index(u, EPSILON) : -2 ; 
}

template<typename E>
inline std::string SUniformRand<E>::Desc(double u, const NP* uu)
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

