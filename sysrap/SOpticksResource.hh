#pragma once
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

    static const char* ResolveUserPrefix(const char* envkey, bool envset); 
    static const char* ResolveGeoCachePrefix();
    static const char* ResolveRngCachePrefix();
    static const char* ResolveUserCachePrefix();

    static const char* GeocacheDir();
    static const char* GeocacheScriptPath(); 

    static const char* RNGCacheDir();
    static const char* RNGDir();
    static const char* RuncacheDir();

    // setkey:true means OPTICKS_KEY envvar gets used 
    static const char* IDPath(bool setkey=true);
    static const NP* IDLoad(const char* relpath); 

    static const char* CGDir_NAME ;
    static const char* CGDir_(bool setkey=true, const char* rel=CGDir_NAME ); 
    static const char* CGDir(bool setkey=true);  

    static const char* CGDir_NAME_Alt ;
    static const char* CGDirAlt(bool setkey=true);  

    static const char* CFBase();
    static const char* CFBaseAlt();
    static const char* CFBaseFromGEOM();

    static std::string Dump(); 

    static const char* KEYS ; 
    static const char* Get(const char* key); 
    static std::string Desc() ; 

};



