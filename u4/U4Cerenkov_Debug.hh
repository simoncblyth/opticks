#pragma once
/**
U4Cerenkov_Debug.h
===========================

Usage::

   export U4Cerenkov_Debug_SaveDir=/tmp
   ntds3

Saves .npy array with first 100 non-reemission records to::

   /tmp/U4Cerenkov_Debug.npy 


**/

#include "plog/Severity.h"
#include <vector>
#include "U4_API_EXPORT.hh"

struct U4_API U4Cerenkov_Debug
{   
    static const plog::Severity LEVEL ; 
    static std::vector<U4Cerenkov_Debug> record ;   
    static constexpr const unsigned NUM_QUAD = 2u ; 
    static constexpr const char* NAME = "U4Cerenkov_Debug.npy" ; 
    static constexpr int LIMIT = 10000 ; 
    static void Save(const char* dir); 
    void add(); 
    void fill(double value); 

    double posx ; 
    double posy ; 
    double posz ;
    double time ; 

    double BetaInverse ; 
    double step_length ; 
    double MeanNumberOfPhotons ; 
    double fNumPhotons ; 
};

