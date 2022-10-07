#pragma once
/**
U4Scintillation_Debug.h
===========================

Usage::

   export U4Scintillation_Debug_SaveDir=/tmp
   ntds3

Saves .npy array with first 100 non-reemission records to::

   /tmp/U4Scintillation_Debug.npy 


**/

#include "plog/Severity.h"
#include <vector>
#include "U4_API_EXPORT.hh"

struct U4_API U4Scintillation_Debug
{   
    static const plog::Severity LEVEL ; 
    static std::vector<U4Scintillation_Debug> record ;   
    static constexpr const unsigned NUM_QUAD = 5u ; 
    static constexpr const char* NAME = "U4Scintillation_Debug.npy" ; 
    static constexpr int LIMIT = 10000 ; 
    static void Save(const char* dir); 
    void add(); 
    void fill(double value); 

    double posx ;
    double posy ;
    double posz ;
    double time ;

    double ResolutionScale ;
    double dx ;
    double dE_dx ;
    double birk1 ;

    double TotalEnergyDeposit ;
    double QuenchedTotalEnergyDeposit ;
    double delta ;
    double birk2 ;

    double ScintillationYield ;
    double MeanNumberOfTracks ;
    double NumTracks ;
    double MeanNumberOfPhotons ;

    double Density ; 
    double fPhotonWeight ;  
    double fApplyPreQE ; 
    double fEnableQuenching ; 

};

