#pragma once
/**
U4Hit_Debug.h
===========================

**/

#include "plog/Severity.h"
#include <vector>
#include "spho.h"
#include "U4_API_EXPORT.hh"

struct U4_API U4Hit_Debug
{   
    static const plog::Severity LEVEL ; 
    static std::vector<U4Hit_Debug> record ;   
    static constexpr const unsigned NUM_QUAD = 1u ; 
    static constexpr const char* NAME = "U4Hit_Debug.npy" ; 
    static constexpr int LIMIT = 10000 ; 
    static void EndOfEvent(int eventID); 
    void add(); 

    spho label ; 

};

