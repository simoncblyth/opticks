#pragma once
/**
SGeoConfig
==============


**/

#include <string>
#include <vector>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

struct SBitSet ;
struct SName ;

struct SYSRAP_API SGeoConfig
{
    static const plog::Severity LEVEL ;
    static constexpr const char* ELVSelection_VERBOSE = "SGeoConfig__ELVSelection_VERBOSE";
    static constexpr const char* ELVSelection_STARTING = "SGeoConfig__ELVSelection_STARTING";
    static std::string Desc();
    std::string desc() const ;

    static std::string DescEMM();

    static constexpr const char* kGEOM           = "OPTICKS_GEOM,GEOM" ;
    static constexpr const char* kEMM            = "OPTICKS_EMM_SELECTION,EMM" ;
    static constexpr const char* kELVSelection   = "OPTICKS_ELV_SELECTION,ELV" ;
    static constexpr const char* kSolidSelection = "OPTICKS_SOLID_SELECTION" ;
    static constexpr const char* kSolidTrimesh   = "OPTICKS_SOLID_TRIMESH" ;
    static constexpr const char* kFlightConfig   = "OPTICKS_FLIGHT_CONFIG" ;
    static constexpr const char* kArglistPath    = "OPTICKS_ARGLIST_PATH" ;
    static constexpr const char* kCXSkipLV       = "OPTICKS_CXSKIPLV" ;           // names
    static constexpr const char* kCXSkipLV_IDXList = "OPTICKS_CXSKIPLV_IDXLIST" ; // idxs

    static constexpr const char* kEMM_desc = "CSGFoundry enabled merged mesh control" ;
    static constexpr const char* kELVSelection_desc = "string control of included/excluded meshes, DYNAMIC NATURE MAKES PROBLEMATIC FOR LONG LIVED SKIPS  " ;
    static constexpr const char* kSolidSelection_desc = "CSGFoundry debug control" ;
    static constexpr const char* kSolidTrimesh_desc = "CSGFoundry comma delimited list of CSGSolid for Trimesh geometry" ;
    static constexpr const char* kFlightConfig_desc = "NEEDS REVIIVING IN NEW WORKFLOW TO WORK WITH SGLM NOT Composition" ;
    static constexpr const char* kArglistPath_desc = "generic path to a list of arguments used by some tests" ;
    static constexpr const char* kCXSkipLV_desc = "non-dynamic LV skipping in CSG_GGeo_Convert, PREFERABLE FOR LONG-LIVED SKIPS" ;

    static const char* _GEOM ;
    static unsigned long long _EMM ;
    static const char* _ELVSelection ;
    static const char* _SolidSelection ;
    static const char* _SolidTrimesh ;
    static const char* _FlightConfig ;
    static const char* _ArglistPath ;
    static const char* _CXSkipLV ;
    static const char* _CXSkipLV_IDXList ;

    static unsigned long long EnabledMergedMesh() ;
    static const char* GEOM();
    static const char* SolidSelection();
    static const char* SolidTrimesh();
    static const char* FlightConfig();
    static const char* ArglistPath();
    static const char* CXSkipLV();
    static const char* CXSkipLV_IDXList();
    static const char* ELVSelection();
    static const char* ELVSelection(const SName* id);
    static const char* ELVPrefix(const char* elvarg);

    // moved here from CSGFoundry
    static const char* ELVString(const SName* id);
    static const SBitSet* ELV(const SName* id);



    static void SetCXSkipLV_IDXList(const SName* id);

    static void SetGEOM(           const char* gm );
    static void SetSolidSelection( const char* ss );
    static void SetSolidTrimesh(   const char* ss );
    static void SetELVSelection(   const char* es );
    static void SetFlightConfig(   const char* fc );
    static void SetArglistPath(    const char* ap );
    static void SetCXSkipLV(       const char* cx );

    static bool IsEnabledMergedMesh(unsigned mm);
    static bool IsCXSkipLV(int lv);

    static std::vector<std::string>*  Arglist() ;
    static void GeometrySpecificSetup(const SName* id);
    static bool JUNO_Detected(const SName* id);


};


