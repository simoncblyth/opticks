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
    static constexpr const char* kCXSkipLV       = "OPTICKS_CXSKIPLV" ; 

    static constexpr const char* kCXSkipLV_desc = "non-dynamic LV skipping in CSG_GGeo_Convert, not usually used" ; 


    static unsigned long long _EMM ; 
    static const char* _SolidSelection ;   
    static const char* _FlightConfig ;   
    static const char* _ArglistPath ;   
    static const char* _CXSkipLV ; 

    static unsigned long long EnabledMergedMesh() ; 
    static const char* SolidSelection(); 
    static const char* FlightConfig(); 
    static const char* ArglistPath(); 
    static const char* CXSkipLV(); 

    static void SetSolidSelection( const char* ss ); 
    static void SetFlightConfig(   const char* fc ); 
    static void SetArglistPath(    const char* ap ); 
    static void SetCXSkipLV(       const char* cx ); 

    static bool IsEnabledMergedMesh(unsigned mm); 
    static bool IsCXSkipLV(int lv); 

    static std::vector<std::string>*  Arglist() ; 


};

 
