#pragma once
/**
SOpticksResource.hh : AIMING TO ELIMINATE THIS : MOVING TO SIMPLER spath.h WITH MORE EXPLICIT PATHS 
==============================================================================================================

Currently this straddles old and new workflows.

Is this abstraction layer still needed ? 
------------------------------------------

NOPE : IN PROCESS OF REMOVING  : BUT MAYBE CAN SLIM DOWN DRASTICALLY::

    epsilon:tests blyth$ opticks-fl SOpticksResource.hh 

    ./CSG/tests/CSGIntersectSolidTest.cc
    ./CSG/tests/CSGGeometryFromGeocacheTest.cc
    ./CSG/CSGFoundry.cc

    ./u4/U4VolumeMaker.cc   ## uses GEOMWrap, GDMLPath, ... 
    ./g4cx/G4CXOpticks.cc   ## GDMLPath methods 


    ## DISCOUNTED
    ./sysrap/SOpticksResource.hh
    ./sysrap/CMakeLists.txt
    ./sysrap/tests/SOpticksResourceTest.cc
    ./sysrap/SOpticksResource.cc
    ./sysrap/SPath.cc
    ./boostrap/BOpticksResource.cc

    epsilon:opticks blyth$ 




**/


#include <string>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

struct NP ; 

struct SYSRAP_API SOpticksResource
{
    static const plog::Severity LEVEL ; 

    static const char* MakeUserDir(const char* sub) ; 
    static const char* GEOCACHE_PREFIX_KEY  ; 
    static const char* RNGCACHE_PREFIX_KEY  ; 
    static const char* USERCACHE_PREFIX_KEY  ; 
    static const char* PRECOOKED_PREFIX_KEY  ; 

    static const char* ResolveUserPrefix(const char* envkey, bool envset); 
    static const char* ResolveGeoCachePrefix();
    static const char* ResolveRngCachePrefix();
    static const char* ResolveUserCachePrefix();
    static const char* ResolvePrecookedPrefix();

    static const char* GeocacheDir();
    static const char* GeocacheScriptPath(); 

    static const char* RNGCacheDir();
    static const char* RNGDir();
    static const char* RuncacheDir();
    static const char* PrecookedDir();

    static const char* ExecutableName(); 
    static const char* ExecutableName_GEOM(); 

    static const char* GEOMFromEnv(const char* fallback); 
    static const char* _GEOM ; 
    static const char* GEOM(const char* fallback=nullptr); 
    static void  SetGEOM(const char* geom); 
    
    static const char* DefaultOutputDir();      // eg /tmp/blyth/opticks/GEOM/acyl/ExecutableName
    static const char* DefaultGeometryBase();   // eg /tmp/blyth/opticks/GEOM 
    static const char* DefaultGeometryDir();    // eg /tmp/blyth/opticks/GEOM/acyl
    static const char* UserGEOMDir();           // eg $HOME/.opticks/GEOM/$GEOM 


    static std::string Desc_DefaultOutputDir();  
  
    static const char* CFBASE_ ;
    static const char* CFBase();
    static const char* CFBaseAlt();
    static const char* CFBaseFromGEOM();
    static const char* GDMLPathFromGEOM(); 
    static const char* WrapLVForName(const char* name); 

    static const char* SearchCFBase(const char* dir); 
    static constexpr const char* SearchCFBase_RELF = "CSGFoundry/solid.npy" ; 

    static const char* SomeGDMLPath_ ; 
    static const char* SomeGDMLPath(); 

    static const char* OpticksGDMLPath_ ; 
    static const char* OpticksGDMLPath(); 

    static const char* GDMLPath(); 
    static const char* GDMLPath(const char* geom); 

    static const char* GEOMSub(); 
    static const char* GEOMSub(const char* geom); 

    static const char* GEOMWrap(); 
    static const char* GEOMWrap(const char* geom); 

    static const char* GEOMList(); 
    static const char* GEOMList(const char* geom); 




    static std::string Dump(); 

    static const char* KEYS ; 
    static const char* Get(const char* key); 
    static std::string Desc() ; 

};



