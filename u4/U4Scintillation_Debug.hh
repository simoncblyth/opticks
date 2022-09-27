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

struct U4Scintillation_Debug
{   
    static const plog::Severity LEVEL ; 
    static std::vector<U4Scintillation_Debug> record ;   
    static constexpr const unsigned NUM_QUAD = 1u ; 
    static constexpr const char* NAME = "U4Scintillation_Debug.npy" ; 
    static constexpr int LIMIT = 10000 ; 
    static constexpr const char* EKEY = "U4Scintillation_Debug_SaveDir" ;   
    static const char* SaveDir ; 
    static void Save(const char* dir);
    static void EndOfEvent(int eventID); 


    double ScintillationYield ; 
    double MeanNumberOfTracks ; 
    double NumTracks ; 
    double Spare ; 

    void add(); 
};

