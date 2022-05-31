#pragma once
/**
SGeoConfig
==============

**/

#include <string>
#include <vector>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

struct SName ; 

struct SYSRAP_API SGeoConfig
{
    static const plog::Severity LEVEL ; 
    static std::string Desc(); 
    static std::string DescEMM(); 

    static constexpr const char* kEMM            = "EMM" ; 
    static constexpr const char* kELVSelection   = "OPTICKS_ELV_SELECTION" ; 
    static constexpr const char* kSolidSelection = "OPTICKS_SOLID_SELECTION" ; 
    static constexpr const char* kFlightConfig   = "OPTICKS_FLIGHT_CONFIG" ; 
    static constexpr const char* kArglistPath    = "OPTICKS_ARGLIST_PATH" ; 
    static constexpr const char* kCXSkipLV       = "OPTICKS_CXSKIPLV" ; 

    static constexpr const char* kEMM_desc = "CSGFoundry enabled merged mesh control" ; 
    static constexpr const char* kELVSelection_desc = "string control of included/excluded meshes" ; 
    static constexpr const char* kSolidSelection_desc = "CSGFoundry debug control" ; 
    static constexpr const char* kFlightConfig_desc = "NEEDS REVIIVING IN NEW WORKFLOW TO WORK WITH SGLM NOT Composition" ; 
    static constexpr const char* kArglistPath_desc = "generic path to a list of arguments used by some times" ; 
    static constexpr const char* kCXSkipLV_desc = "non-dynamic LV skipping in CSG_GGeo_Convert, not usually used" ; 

    static unsigned long long _EMM ; 
    static const char* _ELVSelection ;   
    static const char* _SolidSelection ;   
    static const char* _FlightConfig ;   
    static const char* _ArglistPath ;   
    static const char* _CXSkipLV ; 

    static unsigned long long EnabledMergedMesh() ; 
    static const char* ELVSelection(); 
    static const char* SolidSelection(); 
    static const char* FlightConfig(); 
    static const char* ArglistPath(); 
    static const char* CXSkipLV(); 

    static void SetSolidSelection( const char* ss ); 
    static void SetELVSelection(   const char* es ); 
    static void SetFlightConfig(   const char* fc ); 
    static void SetArglistPath(    const char* ap ); 
    static void SetCXSkipLV(       const char* cx ); 

    static bool IsEnabledMergedMesh(unsigned mm); 
    static bool IsCXSkipLV(int lv); 

    static std::vector<std::string>*  Arglist() ; 
    static void GeometrySpecificSetup(const SName* id); 



};

 
