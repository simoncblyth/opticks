#pragma once
/**
SFastSim_Debug.h
===========================


**/

#include "plog/Severity.h"
#include <vector>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SFastSim_Debug
{   
    static const plog::Severity LEVEL ; 
    static std::vector<SFastSim_Debug> record ;   
    static constexpr const unsigned NUM_QUAD = 4u ; 
    static constexpr const char* NAME = "SFastSim_Debug.npy" ; 
    static constexpr int LIMIT = 100000 ; 
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

