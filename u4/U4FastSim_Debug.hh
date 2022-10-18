#pragma once
/**
U4FastSim_Debug.h
===========================


**/

#include "plog/Severity.h"
#include <vector>
#include "U4_API_EXPORT.hh"

struct U4_API U4FastSim_Debug
{   
    static const plog::Severity LEVEL ; 
    static std::vector<U4FastSim_Debug> record ;   
    static constexpr const unsigned NUM_QUAD = 4u ; 
    static constexpr const char* NAME = "U4FastSim_Debug.npy" ; 
    static constexpr int LIMIT = 10000 ; 
    static void Save(const char* dir); 
    void add(); 
    void fill(double value); 

    double posx ;
    double posy ;
    double posz ;
    double time ;

    double dirx ;
    double diry ;
    double dirz ;
    double dist1 ;

    double polx ;
    double poly ;
    double polz ;
    double dist2 ;

    double ModelTrigger ;
    double whereAmI ;
    double c ;
    double d ;

    // NB for python parsing check line terminations with set list

};

