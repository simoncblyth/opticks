#pragma once
/**
SGeoConfig
==============

**/

#include <string>
#include <vector>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SGeoConfig
{
    static std::string Desc(); 
    static std::string DescEMM(); 

    static constexpr const char* kEMM            = "EMM" ; 
    static constexpr const char* kSolidSelection = "OPTICKS_SOLID_SELECTION" ; 
    static constexpr const char* kFlightConfig   = "OPTICKS_FLIGHT_CONFIG" ; 
    static constexpr const char* kArglistPath    = "OPTICKS_ARGLIST_PATH" ; 

    static unsigned long long _EMM ; 
    static const char* _SolidSelection ;   
    static const char* _FlightConfig ;   
    static const char* _ArglistPath ;   

    static unsigned long long EnabledMergedMesh() ; 
    static const char* SolidSelection(); 
    static const char* FlightConfig(); 
    static const char* ArglistPath(); 

    static void SetSolidSelection( const char* ss ); 
    static void SetFlightConfig(   const char* fc ); 
    static void SetArglistPath(    const char* ap ); 


    static bool IsEnabledMergedMesh(unsigned mm); 
    static std::vector<std::string>*  Arglist() ; 


};

 
