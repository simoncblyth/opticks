#include <cassert>
#include <sstream>
#include <cstdlib>

#include "SSys.hh"
#include "SStr.hh"
#include "SPath.hh"
#include "SProc.hh"
#include "SOpticksResource.hh"
#include "SOpticksKey.hh"
#include "SLOG.hh"

#include "NP.hh"


const plog::Severity SOpticksResource::LEVEL = SLOG::EnvLevel("SOpticksResource", "DEBUG"); 

const char* SOpticksResource::GEOCACHE_PREFIX_KEY = "OPTICKS_GEOCACHE_PREFIX" ; 
const char* SOpticksResource::RNGCACHE_PREFIX_KEY = "OPTICKS_RNGCACHE_PREFIX" ; 
const char* SOpticksResource::USERCACHE_PREFIX_KEY = "OPTICKS_USERCACHE_PREFIX" ; 
const char* SOpticksResource::PRECOOKED_PREFIX_KEY = "OPTICKS_PRECOOKED_PREFIX" ; 

const char* SOpticksResource::MakeUserDir(const char* sub) 
{
    return SPath::Resolve("$HOME", sub, DIRPATH) ; 
}


/**
SOpticksResource::ResolveUserPrefix
----------------------------------------

1. sensitive to envvars :  OPTICKS_GEOCACHE_PREFIX OPTICKS_RNGCACHE_PREFIX OPTICKS_USERCACHE_PREFIX OPTICKS_PRECOOKED_PREFIX
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
const char* SOpticksResource::ResolvePrecookedPrefix(){ return ResolveUserPrefix(PRECOOKED_PREFIX_KEY, true) ; }

const char* SOpticksResource::GeocacheDir(){        return SPath::Resolve(ResolveGeoCachePrefix(), "geocache", NOOP); }
const char* SOpticksResource::GeocacheScriptPath(){ return SPath::Resolve(GeocacheDir(), "geocache.sh", NOOP); }

const char* SOpticksResource::RNGCacheDir(){    return SPath::Resolve(ResolveRngCachePrefix(), "rngcache", NOOP); }
const char* SOpticksResource::RNGDir(){         return SPath::Resolve(RNGCacheDir(), "RNG", NOOP); }
const char* SOpticksResource::RuncacheDir(){    return SPath::Resolve(ResolveUserCachePrefix(), "runcache", NOOP); }
const char* SOpticksResource::PrecookedDir(){   return SPath::Resolve(ResolvePrecookedPrefix(), "precooked", NOOP); }


/**
SOpticksResource::ExecutableName
---------------------------------

In embedded running SProc::ExecutableName returns "python3.8" 
As that is not very informative it is replaced with the value of the 
SCRIPT envvar if it is defined, or otherwise defaulting to "script"

A good practice would be to define and export SCRIPT in the 
invoking bash function, with the line::

   export SCRIPT=$FUNCNAME 

Alternatively the returned name can be controlled directly, not just when embedded, 
using envvar SOpticksResource_ExecutableName

**/

const char* SOpticksResource::ExecutableName()
{  
    const char* exe0 = SProc::ExecutableName() ; 
    bool is_python = SStr::StartsWith(exe0, "python") ;  
    const char* script = SSys::getenvvar("SCRIPT"); 
    const char* exe = ( is_python && script ) ? script : exe0 ; 
    const char* result = SSys::getenvvar("SOpticksResource_ExecutableName", exe ) ; 

    // as this is used before logging is setup cannot use normal logging to check 
    if(SSys::getenvvar("SOpticksResource")) std::cout 
        << "SOpticksResource::ExecutableName" 
        << " exe0 "      << ( exe0 ? exe0 : "-" )
        << " is_python " << is_python 
        << " script "    << ( script ? script : "-" )
        << " exe "       << ( exe  ? exe : "-" )
        << " result "    << ( result ? result : "-" ) 
        ;

    return result ; 
} 

const char* SOpticksResource::ExecutableName_GEOM()
{
    std::stringstream ss ; 
    ss << ExecutableName() << "_GEOM" ;  
    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}

/**
SOpticksResource::GEOMFromEnv
--------------------------------

Precedence order for the GEOM string returned

1. value of envvar ExecutableName_GEOM 
2. value of envvar GEOM 
3. fallback argumnent, which can be nullptr 

**/

const char* SOpticksResource::GEOMFromEnv(const char* fallback)
{
    const char* geom_std = SSys::getenvvar("GEOM", fallback) ; 
    const char* geom_bin = ExecutableName_GEOM() ; 
    const char* geom = SSys::getenvvar(geom_bin, geom_std) ;     
    return geom ; 
}

const char* SOpticksResource::_GEOM = nullptr ; 

/**
SOpticksResource::GEOM
-------------------------

If _GEOM is nullptr an attempt to get it from environment is done, 
otherwise the cached _GEOM is returned.   

**/

const char* SOpticksResource::GEOM(const char* fallback)
{
    if(_GEOM == nullptr) _GEOM = GEOMFromEnv(fallback); 
    return _GEOM ; 
}
void SOpticksResource::SetGEOM( const char* geom )
{
    _GEOM = strdup(geom); 
}


/**
SOpticksResource::DefaultOutputDir
------------------------------------

Default dir used by argumentless SEvt::save is $TMP/GEOM/$GEOM/ExecutableName eg::

   /tmp/blyth/opticks/GEOM/RaindropRockAirWater/G4CXSimulateTest

This allows $TMP/GEOM/$GEOM to be equated with a "temporary CFBASE" for consistent handling in scripts.::

    TMP_CFBASE=/tmp/$USER/opticks/$GEOM

This layout is consistent with geocache output layout CFBASE/ExecutableName eg: 

   /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/G4CXSimulateTest

Previously used the inconsistent flipped layout TMP/ExecutableName/GEOM which complicated scripts. 


Hmm how did this arise::

    /tmp/blyth/opticks/GEOM/ntds3/G4CXOpticks


**/


const char* SOpticksResource::DefaultOutputDir()
{ 
    return SPath::Resolve("$TMP/GEOM", GEOM(), ExecutableName(), NOOP); 
}
const char* SOpticksResource::DefaultGeometryDir()
{ 
    return SPath::Resolve("$TMP/GEOM", GEOM(), NOOP); 
}
const char* SOpticksResource::DefaultGeometryBase()
{ 
    return SPath::Resolve("$TMP/GEOM", NOOP); 
}
const char* SOpticksResource::UserGEOMDir()
{
    return SPath::Resolve("$HOME/.opticks/GEOM", GEOM(), DIRPATH ); 

}



std::string SOpticksResource::Desc_DefaultOutputDir()
{
    const char* geom = GEOM() ; 
    std::stringstream ss ; 
    ss << "SOpticksResource::Desc_DefaultOutputDir" << std::endl 
       << " SPath::Resolve(\"$TMP/GEOM\",NOOP) " << SPath::Resolve("$TMP/GEOM",NOOP) << std::endl 
       << " SOpticksResource::GEOM() " << ( geom ? geom : "-" )  << std::endl 
       << " SOpticksResource::ExecutableName() " << SOpticksResource::ExecutableName() << std::endl 
       << " SOpticksResource::DefaultOutputDir() " << SOpticksResource::DefaultOutputDir() << std::endl 
       ;
    std::string s = ss.str(); 
    return s ; 
}

std::string SOpticksResource::Dump()
{

    const char* geocache_prefix = ResolveGeoCachePrefix()  ;
    const char* rngcache_prefix = ResolveRngCachePrefix() ; 
    const char* usercache_prefix = ResolveUserCachePrefix() ;
    const char* precooked_prefix = ResolvePrecookedPrefix() ;
    const char* geocache_dir = GeocacheDir() ; 
    const char* geocache_scriptpath = GeocacheScriptPath() ; 
    const char* rngcache_dir = RNGCacheDir() ; 
    const char* rng_dir = RNGDir() ; 
    const char* precooked_dir = PrecookedDir() ; 
    const char* outputdir = DefaultOutputDir(); 
    const char* geometrydir = DefaultGeometryDir(); 
    const char* geometrybase = DefaultGeometryBase(); 
    const char* runcache_dir = RuncacheDir() ; 

    bool setkey = true ; 
    const char* idpath = IDPath(setkey) ; 
    //const char* cgdir = CGDir(setkey) ; 
    const char* cfbase = CFBase(); 
    const char* cfbase_alt = CFBaseAlt(); 
    const char* cfbase_fg = CFBaseFromGEOM(); 
    const char* gdmlpath = GDMLPath(); 
    const char* geom = GEOM(); 
    const char* usergeomdir = UserGEOMDir(); 


    std::stringstream ss ; 
    ss 
        << std::endl 
        << "GEOCACHE_PREFIX_KEY                        " << GEOCACHE_PREFIX_KEY  << std::endl 
        << "RNGCACHE_PREFIX_KEY                        " << RNGCACHE_PREFIX_KEY  << std::endl 
        << "USERCACHE_PREFIX_KEY                       " << USERCACHE_PREFIX_KEY  << std::endl 
        << "SOpticksResource::ResolveGeoCachePrefix()  " << ( geocache_prefix ? geocache_prefix : "-" ) << std::endl 
        << "SOpticksResource::ResolveRngCachePrefix()  " << ( rngcache_prefix ? rngcache_prefix : "-" )  << std::endl 
        << "SOpticksResource::ResolveUserCachePrefix() " << ( usercache_prefix ? usercache_prefix : "-" )  << std::endl 
        << "SOpticksResource::ResolvePrecookedPrefix() " << ( precooked_prefix ? precooked_prefix : "-" )  << std::endl 
        << "SOpticksResource::GeocacheDir()            " << ( geocache_dir  ? geocache_dir : "-" ) << std::endl 
        << "SOpticksResource::GeocacheScriptPath()     " << ( geocache_scriptpath ? geocache_scriptpath : "-" ) << std::endl 
        << "SOpticksResource::RNGCacheDir()            " << ( rngcache_dir  ? rngcache_dir : "-" )  << std::endl 
        << "SOpticksResource::RNGDir()                 " << ( rng_dir ? rng_dir : "-" )  << std::endl 
        << "SOpticksResource::PrecookedDir()           " << ( precooked_dir ? precooked_dir : "-" )  << std::endl 
        << "SOpticksResource::DefaultOutputDir()       " << ( outputdir ? outputdir : "-" ) << std::endl 
        << "SOpticksResource::DefaultGeometryBase()    " << ( geometrybase ? geometrybase : "-" ) << std::endl 
        << "SOpticksResource::DefaultGeometryDir()    " << ( geometrydir ? geometrydir : "-" ) << std::endl 
        << "SOpticksResource::RuncacheDir()            " << ( runcache_dir ? runcache_dir : "-" )  << std::endl 
        << "SOpticksResource::IDPath(true)             " << ( idpath ? idpath : "-" ) << std::endl  
       // << "SOpticksResource::CGDir(true)              " << ( cgdir ? cgdir : "-" )  << std::endl 
        << "SOpticksResource::CFBase()                 " << ( cfbase ? cfbase : "-" ) << std::endl 
        << "SOpticksResource::CFBaseAlt()              " << ( cfbase_alt ? cfbase_alt : "-" ) << std::endl 
        << "SOpticksResource::CFBaseFromGEOM()         " << ( cfbase_fg ? cfbase_fg : "-" ) << std::endl 
        << "SOpticksResource::GDMLPath()               " << ( gdmlpath ? gdmlpath : "-" ) << std::endl
        << "SOpticksResource::GEOM()                   " << ( geom ? geom : "-" ) << std::endl
        << "SOpticksResource::UserGEOMDir()            " << ( usergeomdir ? usergeomdir : "-" ) << std::endl
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


/*

const char* SOpticksResource::CGDir_NAME = "CSG_GGeo" ; 
const char* SOpticksResource::CGDir(bool setkey){ return CGDir_(setkey, CGDir_NAME) ; }
const char* SOpticksResource::CGDir_(bool setkey, const char* rel)  
{
    const char* idpath = IDPath(setkey) ; 
    assert( idpath ); 
    return SPath::Resolve( idpath, rel , NOOP ); 
}

const char* SOpticksResource::CGDir_NAME_Alt = "CSG_GGeo_Alt" ; 
const char* SOpticksResource::CGDirAlt(bool setkey){ return CGDir_(setkey, CGDir_NAME_Alt) ; }

*/



/**
SOpticksResource::CFBase
--------------------------

Return the directory path within which the CSGFoundry directory 
will be expected.  The path returned dependes on 
environment variables : CFBASE, OPTICKS_KEY, OPTICKS_GEOCACHE_PREFIX

Precedence order:

1. CFBASE envvar values directly providing CFBASE directory 

2. CFBASE directory derived from OPTICKS_KEY and OPTICKS_GEOCACHE_PREFIX 
   giving "$IDBase/CSG_GGeo"

When the *ekey* envvar (default CFBASE) is defined its 
value is returned otherwise the CFDir obtained from the 
OPTICKS_KEY is returned.  
**/


const char* SOpticksResource::CFBASE_ = "CFBASE" ; 
const char* SOpticksResource::CFBase()
{
    const char* cfbase = SSys::getenvvar(CFBASE_) ; 

    /*
    if( cfbase == nullptr )
    {
        bool setkey = true ; 
        cfbase = CGDir(setkey);    
    }
    */

    return cfbase ; 
}

const char* SOpticksResource::CFBaseAlt()
{
    const char* cfbase = SSys::getenvvar("CFBASE_ALT") ; 

    /*
    if( cfbase == nullptr )
    {
        bool setkey = true ; 
        cfbase = CGDirAlt(setkey); 
    }
    */

    return cfbase ; 
}


/**
SOpticksResource::OldCFBaseFromGEOM
----------------------------------

Construct a CFBASE directory from GEOM envvar.

Have stopped using this automatically from SOpticksResource::CFBase
as GEOM envvar is too commonly used that this can kick in unexpectedly. 

GEOM envvar value such as AltXJfixtureConstruction_FirstSuffix_XY leads for CFBASE folder 
such as /tmp/$USER/opticks/GeoChain_Darwin/AltXJfixtureConstruction_FirstSuffix

**/

const char* SOpticksResource::OldCFBaseFromGEOM()
{
    const char* cfbase = nullptr ; 
    const char* geom = SSys::getenvvar("GEOM"); 
    if( geom != nullptr )
    {
        const char* gcn =  SStr::HeadLast(geom, '_'); 
#ifdef __APPLE__
        const char* rel = "GeoChain_Darwin" ; 
#else
        const char* rel = "GeoChain" ; 
#endif
        cfbase = SPath::Resolve("$TMP", rel, gcn, NOOP  );    
    }
    return cfbase ;  
}


/**
SOpticksResource::CFBaseFromGEOM
----------------------------------

Indirect config of CFBase folder via two envvars.

GEOM
   short name identifying the geometry, eg J001

"$GEOM"_CFBaseFromGEOM
   ie J001_CFBaseFromGEOM

The advantage of the indirect approach is that GEOM provides
a simple name, with the detail of the directory hidden in
the other envvar.   

Bash functions to edit config : geom_, com_. oip

**/

const char* SOpticksResource::CFBaseFromGEOM()
{
    const char* geom = GEOM(); 
    const char* path = geom == nullptr ? nullptr : SSys::getenvvar(SStr::Name(geom, "_CFBaseFromGEOM")) ; 
    LOG(LEVEL) 
        << " geom " << geom 
        << " path " << path 
        ;
    return path ; 
}

const char* SOpticksResource::GDMLPathFromGEOM()
{
    const char* geom = GEOM(); 
    const char* path =  geom == nullptr ? nullptr : SSys::getenvvar(SStr::Name(geom, "_GDMLPathFromGEOM")) ; 
    LOG(LEVEL) 
        << " geom " << geom 
        << " path " << path 
        ;
    return path ; 
}




const char* SOpticksResource::WrapLVForName(const char* name)
{
    assert(name) ; 
    return SSys::getenvvar(SStr::Name(name, "_WrapLVForName")) ; 
}




/**
SOpticksResource::SearchCFBase
-------------------------------

Traverse up the directory provided looking for a directory containing "CSGFoundry/solid.npy"
The first such directory is returned, or nullptr if not found. 

**/

const char* SOpticksResource::SearchCFBase(const char* dir){ return SPath::SearchDirUpTreeWithFile(dir, SearchCFBase_RELF) ; }


const char* SOpticksResource::OpticksGDMLPath_ = "OpticksGDMLPath" ; 
const char* SOpticksResource::OpticksGDMLPath()
{
    return getenv(OpticksGDMLPath_) ;   
}

const char* SOpticksResource::SomeGDMLPath_ = "SomeGDMLPath" ; 
const char* SOpticksResource::SomeGDMLPath()
{
    // TODO: use GDXML instead of the old CGDMLKludge 
    const char* path0 = getenv(SomeGDMLPath_) ;   
    const char* path1 = SPath::Resolve("$IDPath/origin_CGDMLKludge.gdml", NOOP );  
    const char* path2 = SPath::Resolve("$OPTICKS_PREFIX/origin_CGDMLKludge_02mar2022.gdml", NOOP) ; 

    const char* path = SPath::PickFirstExisting(path0, path1, path2); 
    LOG(LEVEL)  << " path " << ( path ? path : "-" ) ;    
    assert(path); 
    return path ; 
}



/**
SOpticksResource::GDMLPath
----------------------------

Converts *geom* name eg "JUNOv101" into a path by reading envvar "JUNOv101_GDMLPath" if it exists, 
returns nullptr when the envvar does not exist or if geom is nullptr. 

For example exercise this with::

    GEOM=hello hello_GDMLPath='$DefaultOutputDir/some/relative/path' SOpticksResourceTest 

* Single quotes are needed to prevent shell expansion of the internal token DefaultOutputDir.
* Notice how the key "hello" provides a shortname with which to refer to the long GDML path. 
* the returned path is expected to be resolved by SPath::Resolve 
* there is no check of the existance of the GDML path 

**/


const char* SOpticksResource::GDMLPath(){ return GDMLPath( GEOM()); }
const char* SOpticksResource::GDMLPath(const char* geom)
{
    return geom == nullptr ? nullptr : SSys::getenvvar(SStr::Name(geom, "_GDMLPath")) ; 
}

const char* SOpticksResource::GEOMSub(){ return GEOMSub( GEOM()); }
const char* SOpticksResource::GEOMSub(const char* geom)
{
    return geom == nullptr ? nullptr : SSys::getenvvar(SStr::Name(geom, "_GEOMSub")) ; 
}

const char* SOpticksResource::GEOMWrap(){ return GEOMWrap( GEOM()); }
const char* SOpticksResource::GEOMWrap(const char* geom)
{
    return geom == nullptr ? nullptr : SSys::getenvvar(SStr::Name(geom, "_GEOMWrap")) ; 
}




const char* SOpticksResource::KEYS = "IDPath CFBase CFBaseAlt GeocacheDir RuncacheDir RNGDir PrecookedDir DefaultOutputDir SomeGDMLPath GDMLPath GEOMSub GEOMWrap CFBaseFromGEOM UserGEOMDir" ; 

/**
SOpticksResource::Get
-----------------------

The below keys have default values derived from the OPTICKS_KEY envvars, however
envvars with the same keys can be used to override these defaults. 

+-------------------------+-----------------------------------------------------+
| key                     |  notes                                              |
+=========================+=====================================================+
|   IDPath                |                                                     |
+-------------------------+-----------------------------------------------------+
|   CFBase                |                                                     |
+-------------------------+-----------------------------------------------------+
|   CFBaseAlt             |                                                     |
+-------------------------+-----------------------------------------------------+
|   GeocacheDir           |                                                     |
+-------------------------+-----------------------------------------------------+
|   RuncacheDir           |                                                     |
+-------------------------+-----------------------------------------------------+
|   RNGDir                |                                                     |
+-------------------------+-----------------------------------------------------+
|   PrecookedDir          |                                                     |
+-------------------------+-----------------------------------------------------+
|   DefaultOutputDir      | eg /tmp/blyth/opticks/GEOM/acyl/ExecutableName      |
+-------------------------+-----------------------------------------------------+
|   DefaultGeometryDir    | eg /tmp/blyth/opticks/GEOM/acyl                     |
+-------------------------+-----------------------------------------------------+
|   DefaultGeometryBase   | eg /tmp/blyth/opticks/GEOM                          |
+-------------------------+-----------------------------------------------------+
|   SomeGDMLPath          |                                                     |
+-------------------------+-----------------------------------------------------+
|   GDMLPath              | GEOM gives Name, Name_GDMLPath gives path           |
+-------------------------+-----------------------------------------------------+
|   CFBaseFromGeom        | GEOM gives Name, Name_CFBaseFromGeom gives dirpath  |
+-------------------------+-----------------------------------------------------+
|   UserGEOMDir           |  eg $HOME/.opticks/GEOM/$GEOM                       |
+-------------------------+-----------------------------------------------------+


**/
const char* SOpticksResource::Get(const char* key) // static
{
    const char* tok = getenv(key) ;   // can override via envvar, but typically below defaults are used
    if(tok) return tok ;  

    if(      strcmp(key, "IDPath")==0)           tok = SOpticksResource::IDPath(); 
    else if( strcmp(key, "CFBase")==0)           tok = SOpticksResource::CFBase(); 
    else if( strcmp(key, "CFBaseAlt")==0)        tok = SOpticksResource::CFBaseAlt(); 
    else if( strcmp(key, "GeocacheDir")==0)      tok = SOpticksResource::GeocacheDir(); 
    else if( strcmp(key, "RuncacheDir")==0)      tok = SOpticksResource::RuncacheDir(); 
    else if( strcmp(key, "RNGDir")==0)           tok = SOpticksResource::RNGDir(); 
    else if( strcmp(key, "PrecookedDir")==0)     tok = SOpticksResource::PrecookedDir(); 
    else if( strcmp(key, "DefaultOutputDir")==0) tok = SOpticksResource::DefaultOutputDir(); 
    else if( strcmp(key, "DefaultGeometryBase")==0) tok = SOpticksResource::DefaultGeometryBase(); 
    else if( strcmp(key, "DefaultGeometryDir")==0) tok = SOpticksResource::DefaultGeometryDir(); 
    else if( strcmp(key, "SomeGDMLPath")==0)     tok = SOpticksResource::SomeGDMLPath(); 
    else if( strcmp(key, "GDMLPath")==0)         tok = SOpticksResource::GDMLPath(); 
    else if( strcmp(key, "CFBaseFromGEOM")==0)   tok = SOpticksResource::CFBaseFromGEOM(); 
    else if( strcmp(key, "UserGEOMDir")==0)      tok = SOpticksResource::UserGEOMDir(); 
    return tok ;  
}


std::string SOpticksResource::Desc() 
{
    std::vector<std::string> keys ; 
    SStr::Split(KEYS, ' ', keys); 

    std::stringstream ss ; 
    ss << "SOpticksResource::Desc" << std::endl ; 
    for(unsigned i=0 ; i < keys.size() ; i++ ) 
    {
        const char* key = keys[i].c_str() ; 
        std::string lab = SStr::Format("SOpticksResource::Get(\"%s\") ", key) ; 
        const char* val = Get(key); 
        ss 
            << std::setw(70) << lab.c_str() 
            << " : "
            << ( val ? val : "-" ) 
            << std::endl 
            ;
    }


    

    const char* gdml_key = "$IDPath/origin_GDMLKludge.gdml" ; 

    for(int pass=0 ; pass < 5 ; pass++)
    {
        switch(pass)
        {
           case 1: SSys::setenvvar("IDPath", "/some/override/IDPath/via/envvar", true );  break ; 
           case 2: SSys::unsetenv("IDPath") ; break ; 
           case 3: SSys::setenvvar("IDPath", "/another/override/IDPath/via/envvar", true );  break ; 
           case 4: SSys::unsetenv("IDPath") ; break ; 
        }
        const char* gdml_path = SPath::Resolve(gdml_key, NOOP );
        ss 
            << std::setw(70) << gdml_key 
            << " : "
            << ( gdml_path ? gdml_path : "-" )
            << std::endl 
            ;
    }



    std::string s = ss.str(); 
    return s ; 
}


