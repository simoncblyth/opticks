#include <cassert>
#include <sstream>

#include "SSys.hh"
#include "SStr.hh"
#include "SPath.hh"
#include "SOpticksResource.hh"
#include "SOpticksKey.hh"
#include "PLOG.hh"

#include "NP.hh"


const plog::Severity SOpticksResource::LEVEL = PLOG::EnvLevel("SOpticksResource", "DEBUG"); 

const char* SOpticksResource::GEOCACHE_PREFIX_KEY = "OPTICKS_GEOCACHE_PREFIX" ; 
const char* SOpticksResource::RNGCACHE_PREFIX_KEY = "OPTICKS_RNGCACHE_PREFIX" ; 
const char* SOpticksResource::USERCACHE_PREFIX_KEY = "OPTICKS_USERCACHE_PREFIX" ; 

const char* SOpticksResource::MakeUserDir(const char* sub) 
{
    int createdirs = 2 ; // 2:dirpath 
    return SPath::Resolve("$HOME", sub, createdirs) ; 
}


/**
SOpticksResource::ResolveUserPrefix
----------------------------------------

1. sensitive to envvars :  OPTICKS_GEOCACHE_PREFIX OPTICKS_RNGCACHE_PREFIX OPTICKS_USERCACHE_PREFIX 
2. if envvar not defined defaults to $HOME/.opticks 
3. the envvar is subsequently internally set for consistency 

NB changes to layout need to be done in several places C++/bash/py::

   ana/geocache.bash
   ana/key.py
   boostrap/BOpticksResource.cc
   sysrap/SOpticksResource.cc

**/

const char* SOpticksResource::ResolveUserPrefix(const char* envkey, bool envset)  // static
{
    const char* evalue = SSys::getenvvar(envkey);    
    const char* prefix = evalue == nullptr ?  MakeUserDir(".opticks") : evalue ; 
    if(envset)
    {
        bool overwrite = true ; 
        int rc = SSys::setenvvar(envkey, prefix, overwrite );
        LOG(LEVEL) 
             << " envkey " << envkey
             << " prefix " << prefix
             << " rc " << rc
             ;    
        assert( rc == 0 );            
    } 
    return prefix ; 
}

const char* SOpticksResource::ResolveGeoCachePrefix() { return ResolveUserPrefix(GEOCACHE_PREFIX_KEY, true) ; }
const char* SOpticksResource::ResolveRngCachePrefix() { return ResolveUserPrefix(RNGCACHE_PREFIX_KEY, true) ; }
const char* SOpticksResource::ResolveUserCachePrefix(){ return ResolveUserPrefix(USERCACHE_PREFIX_KEY, true) ; }

const char* SOpticksResource::GeocacheDir(){        return SPath::Resolve(ResolveGeoCachePrefix(), "geocache", 0); }
const char* SOpticksResource::GeocacheScriptPath(){ return SPath::Resolve(GeocacheDir(), "geocache.sh", 0); }

const char* SOpticksResource::RNGCacheDir(){    return SPath::Resolve(ResolveRngCachePrefix(), "rngcache", 0); }
const char* SOpticksResource::RNGDir(){         return SPath::Resolve(RNGCacheDir(), "RNG", 0); }
const char* SOpticksResource::RuncacheDir(){    return SPath::Resolve(ResolveUserCachePrefix(), "runcache", 0); }



std::string SOpticksResource::Dump()
{

    const char* geocache_prefix = ResolveGeoCachePrefix()  ;
    const char* rngcache_prefix = ResolveRngCachePrefix() ; 
    const char* usercache_prefix = ResolveUserCachePrefix() ;
    const char* geocache_dir = GeocacheDir() ; 
    const char* geocache_scriptpath = GeocacheScriptPath() ; 
    const char* rngcache_dir = RNGCacheDir() ; 
    const char* rng_dir = RNGDir() ; 
    const char* runcache_dir = RuncacheDir() ; 

    bool setkey = true ; 
    const char* idpath = IDPath(setkey) ; 
    const char* cgdir = CGDir(setkey) ; 
    const char* cfbase = CFBase(); 


    std::stringstream ss ; 
    ss 
        << std::endl 
        << "GEOCACHE_PREFIX_KEY                        " << GEOCACHE_PREFIX_KEY  << std::endl 
        << "RNGCACHE_PREFIX_KEY                        " << RNGCACHE_PREFIX_KEY  << std::endl 
        << "USERCACHE_PREFIX_KEY                       " << USERCACHE_PREFIX_KEY  << std::endl 
        << "SOpticksResource::ResolveGeoCachePrefix()  " << ( geocache_prefix ? geocache_prefix : "-" ) << std::endl 
        << "SOpticksResource::ResolveRngCachePrefix()  " << ( rngcache_prefix ? rngcache_prefix : "-" )  << std::endl 
        << "SOpticksResource::ResolveUserCachePrefix() " << ( usercache_prefix ? usercache_prefix : "-" )  << std::endl 
        << "SOpticksResource::GeocacheDir()            " << ( geocache_dir  ? geocache_dir : "-" ) << std::endl 
        << "SOpticksResource::GeocacheScriptPath()     " << ( geocache_scriptpath ? geocache_scriptpath : "-" ) << std::endl 
        << "SOpticksResource::RNGCacheDir()            " << ( rngcache_dir  ? rngcache_dir : "-" )  << std::endl 
        << "SOpticksResource::RNGDir()                 " << ( rng_dir ? rng_dir : "-" )  << std::endl 
        << "SOpticksResource::RuncacheDir()            " << ( runcache_dir ? runcache_dir : "-" )  << std::endl 
        << "SOpticksResource::IDPath(true)             " << ( idpath ? idpath : "-" ) << std::endl  
        << "SOpticksResource::CGDir(true)              " << ( cgdir ? cgdir : "-" )  << std::endl 
        << "SOpticksResource::CFBase()                 " << ( cfbase ? cfbase : "-" ) << std::endl 
        ;

    std::string s = ss.str(); 
    return s ; 
}

/**
SOpticksResource::IDPath
-------------------------

The default setkey:true means that the OPTICKS_KEY from the environment is used. 

**/

const char* SOpticksResource::IDPath(bool setkey) 
{  
    if(setkey)
    {
        SOpticksKey::SetKey(); 
    }
    const char* base = GeocacheDir(); 
    const SOpticksKey* key = SOpticksKey::GetKey() ; 
    return key == nullptr ? nullptr : key->getIdPath(base)  ;  
}

const NP* SOpticksResource::IDLoad(const char* relpath)
{
    const char* idpath = SOpticksResource::IDPath();
    return NP::Load(idpath, relpath) ; 
}



const char* SOpticksResource::CGDir(bool setkey)  // formerally CSG_GGeoDir 
{
    const char* idpath = IDPath(setkey) ; 
    assert( idpath ); 
    int create_dirs = 0 ; 
    return SPath::Resolve( idpath, "CSG_GGeo" , create_dirs ); 
}


/**
SOpticksResource::CFBase
--------------------------

Return the directory path within which the CSGFoundry directory 
will be expected.  The path returned dependes on several 
environment variables. 

Precedence order:

1. GEOM envvar value such as AltXJfixtureConstruction_FirstSuffix_XY leads for CFBASE folder 
   such as /tmp/$USER/opticks/GeoChain_Darwin/AltXJfixtureConstruction_FirstSuffix

2. CFBASE envvar values directly providing CFBASE directory 

3. CFBASE directory derived from OPTICKS_KEY and OPTICKS_GEOCACHE_PREFIX 


When the *ekey* envvar (default CFBASE) is defined its 
value is returned otherwise the CFDir obtained from the 
OPTICKS_KEY is returned.  
**/

const char* SOpticksResource::CFBase(const char* ekey)
{
    const char* cfbase = nullptr ; 
    const char* geom = SSys::getenvvar("GEOM"); 
        
    if( geom != nullptr )
    {
        const char* gcn =  SStr::HeadLast(geom, '_'); 

        int create_dirs = 0 ; 
#ifdef __APPLE__
        const char* rel = "GeoChain_Darwin" ; 
#else
        const char* rel = "GeoChain" ; 
#endif
        cfbase = SPath::Resolve("$TMP", rel, gcn, create_dirs  );    
    }
    else
    {
        cfbase = SSys::getenvvar(ekey) ; 
        if( cfbase == nullptr )
        {
            bool setkey = true ; 
            cfbase = CGDir(setkey); 
        }
    }
    return cfbase ; 
}




