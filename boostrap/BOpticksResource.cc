/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <cassert>
#include <cstring>
#include <csignal>
#include <iostream>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


// for the OKCONF_OPTICKS_INSTALL_PREFIX define from OKConf_Config.hh
#include "OKConf.hh"

#include "SLog.hh"
#include "SSys.hh"
#include "SStr.hh"
#include "SProc.hh"
#include "SAr.hh"

#include "BMeta.hh"
#include "BFile.hh"
#include "BStr.hh"
#include "BPath.hh"
#include "BEnv.hh"
#include "BResource.hh"
#include "BOpticksResource.hh"
#include "BOpticksKey.hh"

#include "PLOG.hh"

const plog::Severity BOpticksResource::LEVEL = PLOG::EnvLevel("BOpticksResource", "DEBUG") ; 


BOpticksResource* BOpticksResource::fInstance = NULL  ;  
BOpticksResource* BOpticksResource::Instance()
{
    return fInstance ; 
}
BOpticksResource* BOpticksResource::Get(const char* spec)  // static 
{
    return fInstance ? fInstance : Create(spec) ; 
}

BOpticksResource* BOpticksResource::Create(const char* spec)  // static 
{
    BOpticksKey::SetKey(spec) ;  //  spec is normally NULL indicating use OPTICKS_KEY envvar 
    BOpticksResource* rsc = new BOpticksResource ; 
    return rsc ; 
}

/**
BOpticksResource::GetCachePath
-------------------------------

If the BOpticksResource instance is not yet created, this will create the
instance and do initViaKey assuming an OPTICKS_KEY envvar.

**/
const char* BOpticksResource::GetCachePath(const char* rela, const char* relb, const char* relc ) // static 
{
    BOpticksResource* rsc = BOpticksResource::Get(NULL) ; 
    return rsc->makeIdPathPath(rela, relb, relc); 
}


const char* BOpticksResource::G4ENV_RELPATH = "externals/config/geant4.ini" ;
const char* BOpticksResource::OKDATA_RELPATH = "opticksdata/config/opticksdata.ini" ; 
const char* BOpticksResource::PREFERENCE_BASE = "$HOME/.opticks" ; 

const char* BOpticksResource::EMPTY  = "" ; 
const char* BOpticksResource::G4LIVE  = "g4live" ; 


/**
BOpticksResource::BOpticksResource
----------------------------------------

OPTICKS_RESOURCE_LAYOUT envvar -> m_layout, which overrides the default of 0

**/


BOpticksResource::BOpticksResource()
    :
    m_testgeo(false), 
    m_log(new SLog("BOpticksResource::BOpticksResource","",debug)),
    m_setup(false),
    m_key(BOpticksKey::GetKey()),   // will be NULL unless BOpticksKey::SetKey has been called 
    m_id(NULL),
    m_res(new BResource),
    m_layout(SSys::getenvint("OPTICKS_RESOURCE_LAYOUT", 0)),   //  gets reset by from the key 
    m_install_prefix(NULL),
    m_geocache_prefix(NULL),
    m_rngcache_prefix(NULL),
    m_usercache_prefix(NULL),
    m_opticksdata_dir(NULL),
    m_opticksaux_dir(NULL),
    m_geocache_dir(NULL),
    m_rngcache_dir(NULL),
    m_runcache_dir(NULL),
    m_resource_dir(NULL),
    m_gensteps_dir(NULL),
    m_export_dir(NULL),
    m_installcache_dir(NULL),
    m_optixcachedefault_dir(NULL),
    m_rng_dir(NULL),
    m_tmpuser_dir(NULL),
    m_srcpath(NULL),
    m_srcfold(NULL),
    m_srcbase(NULL),
    m_srcdigest(NULL),
    m_idfold(NULL),
    m_idfile(NULL),
    m_idgdml(NULL),
    m_idsubd(NULL),
    m_idname(NULL),
    m_idpath(NULL),
    m_idpath_tmp(NULL),
    m_evtbase(NULL),
    m_evtpfx(NULL),
    m_debugging_idpath(NULL),
    m_debugging_idfold(NULL),
    m_daepath(NULL),
    m_gdmlpath(NULL),
    m_srcgdmlpath(NULL),
    m_srcgltfpath(NULL),
    m_idmappath(NULL),
    m_g4codegendir(NULL),
    m_gdmlauxmetapath(NULL),
    m_cachemetapath(NULL),
    m_runcommentpath(NULL),
    m_primariespath(NULL),
    m_directgensteppath(NULL),
    m_directphotonspath(NULL),
    m_gltfpath(NULL),
    m_testcsgpath(NULL),
    m_testconfig(NULL),
    m_gdmlauxmeta(NULL),
    m_gdmlauxmeta_lvmeta(NULL),
    m_gdmlauxmeta_usermeta(NULL),
    m_opticks_geospecific_options(NULL) 
{
    init();
    (*m_log)("DONE"); 
    fInstance = this ; 

}


BOpticksResource::~BOpticksResource()
{
}

void BOpticksResource::init()
{
    LOG(LEVEL) << " layout  : " << m_layout ; 

    initInstallPrefix() ;
    initGeoCachePrefix() ;
    initRngCachePrefix() ;
    initUserCachePrefix() ;
    initTopDownDirs();
    initDebuggingIDPATH();
    initViaKey();    // now the one and only way of setting up 
    initMetadata(); 
}

const char* BOpticksResource::getInstallPrefix() // canonically /usr/local/opticks
{
    return m_install_prefix ; 
}
const char* BOpticksResource::getGeoCachePrefix() // canonically  ~/.opticks with tilde expanded
{
    return m_geocache_prefix ; 
}
const char* BOpticksResource::getRngCachePrefix() // canonically  ~/.opticks with tilde expanded
{
    return m_rngcache_prefix ; 
}
const char* BOpticksResource::getUserCachePrefix() // canonically  ~/.opticks with tilde expanded
{
    return m_usercache_prefix ; 
}




const char* BOpticksResource::InstallPath(const char* relpath) 
{
    std::string path = BFile::FormPath(ResolveInstallPrefix(), relpath) ;
    return strdup(path.c_str()) ;
}

const char* BOpticksResource::InstallPathG4ENV() 
{
    return InstallPath(G4ENV_RELPATH);
}
const char* BOpticksResource::InstallPathOKDATA() 
{
    return InstallPath(OKDATA_RELPATH);
}


std::string BOpticksResource::getInstallPath(const char* relpath) const 
{
    std::string path = BFile::FormPath(m_install_prefix, relpath) ;
    return path ;
}

const char* BOpticksResource::LEGACY_GEOMETRY_ENABLED_KEY = "OPTICKS_LEGACY_GEOMETRY_ENABLED" ; 
bool BOpticksResource::IsLegacyGeometryEnabled()
{
   int lge = SSys::getenvint(LEGACY_GEOMETRY_ENABLED_KEY, -1);
   return lge == 1 ; 
}

const char* BOpticksResource::FOREIGN_GEANT4_ENABLED_KEY = "OPTICKS_FOREIGN_GEANT4_ENABLED" ; 
bool BOpticksResource::IsForeignGeant4Enabled()
{
   int fge = SSys::getenvint(FOREIGN_GEANT4_ENABLED_KEY, -1);
   return fge == 1 ; 
}


const char* BOpticksResource::DEFAULT_TARGET_KEY = "OPTICKS_TARGET" ; 
const char* BOpticksResource::DEFAULT_TARGETPVN_KEY = "OPTICKS_TARGETPVN" ; 
const char* BOpticksResource::DEFAULT_GENSTEP_TARGET_KEY = "OPTICKS_GENSTEP_TARGET" ; 
const char* BOpticksResource::DEFAULT_DOMAIN_TARGET_KEY = "OPTICKS_DOMAIN_TARGET" ; 

int BOpticksResource::DefaultTarget(int fallback) // static
{
   return SSys::getenvint(DEFAULT_TARGET_KEY, fallback);
}

const char* BOpticksResource::DefaultTargetPVN(const char* fallback) // static
{
   return SSys::getenvvar(DEFAULT_TARGETPVN_KEY, fallback ? strdup(fallback) : nullptr );
}



int BOpticksResource::DefaultGenstepTarget(int fallback) // static
{
   return SSys::getenvint(DEFAULT_GENSTEP_TARGET_KEY, fallback);
}
int BOpticksResource::DefaultDomainTarget(int fallback) // static
{
   return SSys::getenvint(DEFAULT_DOMAIN_TARGET_KEY, fallback);
}





const char* BOpticksResource::DEFAULT_KEY_KEY = "OPTICKS_KEY" ; 
std::string BOpticksResource::DefaultKey() // static
{
   const char* key = SSys::getenvvar(DEFAULT_KEY_KEY, "");
   return key  ; 
}







/**
BOpticksResource::IsGeant4EnvironmentDetected
----------------------------------------------

Look for G4...DATA envvars with values that point 
at existing directories. Returns true when 10 
of these are found.

This may change with new Geant4 versions. 

**/

bool BOpticksResource::IsGeant4EnvironmentDetected()
{
    BEnv* e = BEnv::Create("G4"); 
    bool require_existing_dir = true ; 
    unsigned n = e->getNumberOfEnvvars("G4", "DATA", require_existing_dir ) ;     
    bool detect = n >= 10 ;  

    LOG(info) << " n " << n << " detect " << detect ; 

    return detect ; 
}










const char* BOpticksResource::RESULTS_PREFIX_DEFAULT = "$TMP" ; 
const char* BOpticksResource::RESULTS_PREFIX_KEY = "OPTICKS_RESULTS_PREFIX" ; 
const char* BOpticksResource::ResolveResultsPrefix()  // static
{
    return SSys::getenvvar(RESULTS_PREFIX_KEY, RESULTS_PREFIX_DEFAULT);  
}

const char* BOpticksResource::EVENT_BASE_DEFAULT = "$TMP" ; 
const char* BOpticksResource::EVENT_BASE_KEY = "OPTICKS_EVENT_BASE" ; 
const char* BOpticksResource::ResolveEventBase()  // static
{
    return SSys::getenvvar(EVENT_BASE_KEY, EVENT_BASE_DEFAULT);  
}



const char* BOpticksResource::INSTALL_PREFIX_KEY = "OPTICKS_INSTALL_PREFIX" ; 
const char* BOpticksResource::INSTALL_PREFIX_KEY2 = "OPTICKSINSTALLPREFIX" ; 

/**
BOpticksResource::ResolveInstallPrefix
----------------------------------------

1. sensitive to envvar OPTICKS_INSTALL_PREFIX
2. if envvar not defined uses the compiled in OKCONF_OPTICKS_INSTALL_PREFIX from CMake
3. the envvar is subsequently internally set by BOpticksResource::initInstallPrefix

**/

const char* BOpticksResource::ResolveInstallPrefix()  // static
{
    const char* evalue = SSys::getenvvar(INSTALL_PREFIX_KEY);    
    return evalue == NULL ?  strdup(OKCONF_OPTICKS_INSTALL_PREFIX) : evalue ; 
}

void BOpticksResource::initInstallPrefix()
{
    m_install_prefix = ResolveInstallPrefix();
    m_res->addDir("install_prefix", m_install_prefix );

    bool overwrite = true ; 
    int rc = SSys::setenvvar(INSTALL_PREFIX_KEY, m_install_prefix, overwrite );  
    // always set for uniformity 

    LOG(verbose) 
        << " install_prefix " << m_install_prefix  
        << " key " << INSTALL_PREFIX_KEY
        << " rc " << rc
        ;   
 
    assert(rc==0); 

    // for test geometry config underscore has special meaning, so duplicate the envvar without underscore in the key
    int rc2 = SSys::setenvvar(INSTALL_PREFIX_KEY2 , m_install_prefix, true );  
    assert(rc2==0); 


    // The CMAKE_INSTALL_PREFIX from opticks-;opticks-cmake 
    // is set to the result of the opticks-prefix bash function 
    // at configure time.
    // This is recorded into a config file by okc-/CMakeLists.txt 
    // and gets compiled into the OpticksCore library.
    //  
    // Canonically it is :  /usr/local/opticks 

    m_res->addPath("g4env_ini", InstallPathG4ENV() );
    m_res->addPath("okdata_ini", InstallPathOKDATA() );

}



const char* BOpticksResource::GEOCACHE_PREFIX_KEY = "OPTICKS_GEOCACHE_PREFIX" ; 
const char* BOpticksResource::RNGCACHE_PREFIX_KEY = "OPTICKS_RNGCACHE_PREFIX" ; 
const char* BOpticksResource::USERCACHE_PREFIX_KEY = "OPTICKS_USERCACHE_PREFIX" ; 

/**
BOpticksResource::ResolveGeoCachePrefix
----------------------------------------

1. sensitive to envvar OPTICKS_GEOCACHE_PREFIX
2. if envvar not defined defaults to $HOME/.opticks 
3. the envvar is subsequently internally set by BOpticksResource::initCachePrefix

NB changes to layout need to be done in triplicate C++/bash/py::

   ana/geocache.bash
   ana/key.py
   boostrap/BOpticksResource.cc

**/

const char* BOpticksResource::ResolveGeoCachePrefix()  // static
{
    const char* evalue = SSys::getenvvar(GEOCACHE_PREFIX_KEY);    
    return evalue == NULL ?  MakeUserDir(".opticks", NULL) : evalue ; 
}
const char* BOpticksResource::ResolveRngCachePrefix()  // static
{
    const char* evalue = SSys::getenvvar(RNGCACHE_PREFIX_KEY);    
    return evalue == NULL ?  MakeUserDir(".opticks", NULL) : evalue ; 
}
const char* BOpticksResource::ResolveUserCachePrefix()  // static
{
    const char* evalue = SSys::getenvvar(USERCACHE_PREFIX_KEY);    
    return evalue == NULL ?  MakeUserDir(".opticks", NULL) : evalue ; 
}




void BOpticksResource::initGeoCachePrefix()
{
    m_geocache_prefix = ResolveGeoCachePrefix();
    m_res->addDir("geocache_prefix", m_geocache_prefix );

    bool overwrite = true ; 
    int rc = SSys::setenvvar(GEOCACHE_PREFIX_KEY, m_geocache_prefix, overwrite );  
    // always set for uniformity 

    LOG(LEVEL) 
         << " geocache_prefix " << m_geocache_prefix  
         << " key " << GEOCACHE_PREFIX_KEY
         << " rc " << rc
         ;   
 
    assert(rc==0); 
}

void BOpticksResource::initRngCachePrefix()
{
    m_rngcache_prefix = ResolveRngCachePrefix();
    m_res->addDir("rngcache_prefix", m_rngcache_prefix );

    bool overwrite = true ; 
    int rc = SSys::setenvvar(RNGCACHE_PREFIX_KEY, m_rngcache_prefix, overwrite );  
    // always set for uniformity 

    LOG(LEVEL) 
         << " rngcache_prefix " << m_rngcache_prefix  
         << " key " << RNGCACHE_PREFIX_KEY
         << " rc " << rc
         ;   
 
    assert(rc==0); 
}

void BOpticksResource::initUserCachePrefix()
{
    m_usercache_prefix = ResolveUserCachePrefix();
    m_res->addDir("usercache_prefix", m_usercache_prefix );

    bool overwrite = true ; 
    int rc = SSys::setenvvar(USERCACHE_PREFIX_KEY, m_usercache_prefix, overwrite );  
    // always set for uniformity 

    LOG(LEVEL) 
         << " usercache_prefix " << m_usercache_prefix  
         << " key " << USERCACHE_PREFIX_KEY
         << " rc " << rc
         ;   
 
    assert(rc==0); 
}










std::string BOpticksResource::getGeocachePath(const char* rela, const char* relb, const char* relc, const char* reld ) const 
{
    std::string path = BFile::FormPath(m_geocache_dir, rela, relb, relc, reld ) ;
    return path ;
}

std::string BOpticksResource::getResultsPath(const char* rela, const char* relb, const char* relc, const char* reld ) const 
{
    std::string path = BFile::FormPath(m_results_dir, rela, relb, relc, reld ) ;
    return path ;
}





std::string BOpticksResource::getIdPathPath(const char* rela, const char* relb, const char* relc, const char* reld ) const 
{
    const char* idpath = getIdPath(); 
    LOG(debug) << " idpath " << idpath ; 

    std::string path = BFile::FormPath(idpath, rela, relb, relc, reld ) ;
    return path ;
}






void BOpticksResource::initTopDownDirs()
{ 
    m_geocache_dir         = GeocacheDir() ;      // eg ~/.opticks/geocache
    m_runcache_dir         = RuncacheDir() ;      // eg ~/.opticks/runcache
    m_rngcache_dir         = RNGCacheDir() ;      // eg ~/.opticks/rngcache
    m_rng_dir              = RNGDir() ;           // eg ~/.opticks/rngcache/RNG
    m_results_dir          = ResultsDir() ;       // eg /usr/local/opticks/results

    m_opticksdata_dir      = OpticksDataDir() ;   // eg /usr/local/opticks/opticksdata
    m_opticksaux_dir       = OpticksAuxDir() ;    // eg /usr/local/opticks/opticksaux or /usr/local/opticks/opticksdata depending on OPTICKS_LEGACY_GEOMETRY_ENABLED
   
    // TRANSITIONAL : opticksdata IS TO BE REMOVED
 
    m_resource_dir         = ResourceDir() ;      // eg /usr/local/opticks/opticksdata/resource
    m_gensteps_dir         = GenstepsDir() ;      // eg /usr/local/opticks/opticksdata/gensteps
    m_export_dir           = ExportDir() ;        // eg /usr/local/opticks/opticksdata/export

    m_installcache_dir     = InstallCacheDir() ;  // eg /usr/local/opticks/installcache

    m_optixcachedefault_dir  = OptiXCachePathDefault() ;   // eg /var/tmp/simon/OptiXCache



    m_res->addDir("opticksdata_dir", m_opticksdata_dir);
    m_res->addDir("opticksaux_dir",  m_opticksaux_dir);

    m_res->addDir("geocache_dir",    m_geocache_dir );
    m_res->addDir("rngcache_dir",    m_rngcache_dir );
    m_res->addDir("runcache_dir",    m_runcache_dir );

    m_res->addDir("results_dir",     m_results_dir );
    m_res->addDir("resource_dir",    m_resource_dir );
    m_res->addDir("gensteps_dir",    m_gensteps_dir );
    m_res->addDir("export_dir",      m_export_dir);
    m_res->addDir("installcache_dir", m_installcache_dir );
    m_res->addDir("optixcachedefault_dir",  m_optixcachedefault_dir ); 

    m_res->addDir("rng_dir", m_rng_dir );

    //m_res->addDir("okc_installcache_dir", m_okc_installcache_dir );


    m_tmpuser_dir = MakeTmpUserDir("opticks", NULL) ;  // now usurped with $TMP
    m_res->addDir( "tmpuser_dir", m_tmpuser_dir ); 
}

void BOpticksResource::initDebuggingIDPATH()
{
    // directories based on IDPATH envvar ... this is for debugging 
    // and as workaround for npy level tests to access geometry paths 
    // NB should only be used at that level... at higher levels use OpticksResource for this

    
    m_debugging_idpath = SSys::getenvvar("IDPATH") ;

    if(!m_debugging_idpath) return ; 

    std::string idfold = BFile::ParentDir(m_debugging_idpath) ;
    m_debugging_idfold = strdup(idfold.c_str());

}




const char* BOpticksResource::getDebuggingTreedir(int argc, char** argv)
{
    int arg1 = BStr::atoi(argc > 1 ? argv[1] : "-1", -1 );
    const char* idfold = getDebuggingIDFOLD() ;

    std::string treedir ; 

    if(arg1 > -1) 
    {   
        // 1st argument is an integer
        treedir = BFile::FormPath( idfold, "extras", BStr::itoa(arg1) ) ; 
    }   
    else if( argc > 1)
    {
        // otherwise string argument
        treedir = argv[1] ;
    }
    else
    {   
        treedir = BFile::FormPath( idfold, "extras") ;
    }   
    return treedir.empty() ? NULL : strdup(treedir.c_str()) ; 
}


const char* BOpticksResource::GeocacheDir(){    return MakePath(ResolveGeoCachePrefix(), "geocache",  NULL); }
const char* BOpticksResource::RNGCacheDir(){    return MakePath(ResolveRngCachePrefix(), "rngcache",  NULL); }
const char* BOpticksResource::RNGDir(){         return MakePath(RNGCacheDir(), "RNG", NULL); }

const char* BOpticksResource::RuncacheDir(){    return MakePath(ResolveUserCachePrefix(), "runcache",  NULL); }

const char* BOpticksResource::ShaderDir(){      return MakePath(ResolveInstallPrefix(), "gl",  NULL); }
const char* BOpticksResource::InstallCacheDir(){return MakePath(ResolveInstallPrefix(), "installcache",  NULL); }
//const char* BOpticksResource::OKCInstallPath(){ return MakePath(ResolveInstallPrefix(), "installcache", "OKC"); }

const char* BOpticksResource::OpticksDataDir(){ return MakePath(ResolveInstallPrefix(), "opticksdata",  NULL); }
const char* BOpticksResource::OpticksAuxDir(){  return MakePath(ResolveInstallPrefix(),  IsLegacyGeometryEnabled() ? "opticksdata" : "opticksaux" ,  NULL); }

const char* BOpticksResource::ResourceDir(){    return MakePath(OpticksAuxDir() , "resource", NULL ); }
const char* BOpticksResource::GenstepsDir(){    return MakePath(OpticksAuxDir() , "gensteps", NULL ); }
const char* BOpticksResource::ExportDir(){      return MakePath(OpticksAuxDir() , "export",  NULL  ); }


// problematic in readonly installs : because results do not belong with install paths 
const char* BOpticksResource::ResultsDir(){     return MakePath(ResolveResultsPrefix(), "results",  NULL); }



const char* BOpticksResource::getInstallDir() {         return m_install_prefix ; }   
const char* BOpticksResource::getOpticksDataDir() {     return m_opticksdata_dir ; }   
const char* BOpticksResource::getGeocacheDir() {        return m_geocache_dir ; }   
const char* BOpticksResource::getRuncacheDir() {        return m_runcache_dir ; }   
const char* BOpticksResource::getResultsDir() {         return m_results_dir ; }   
const char* BOpticksResource::getResourceDir() {        return m_resource_dir ; } 
const char* BOpticksResource::getExportDir() {          return m_export_dir ; } 

const char* BOpticksResource::getInstallCacheDir() {    return m_installcache_dir ; } 
const char* BOpticksResource::getRNGDir() {             return m_rng_dir ; } 
const char* BOpticksResource::getOptiXCacheDirDefault() const { return m_optixcachedefault_dir ; } 
const char* BOpticksResource::getTmpUserDir() const {   return m_tmpuser_dir ; } 


const char* BOpticksResource::getDebuggingIDPATH() {    return m_debugging_idpath ; } 
const char* BOpticksResource::getDebuggingIDFOLD() {    return m_debugging_idfold ; } 


// note taking from GGeoTest::initCreateCSG for inclusion in evt metadata

void BOpticksResource::setTestCSGPath(const char* testcsgpath)
{
    m_testcsgpath = testcsgpath ? strdup(testcsgpath) : NULL ; 
}
const char* BOpticksResource::getTestCSGPath() const 
{
    return m_testcsgpath  ;
}
void BOpticksResource::setTestConfig(const char* testconfig)
{
    m_testconfig = testconfig ? strdup(testconfig) : NULL ; 
}
const char* BOpticksResource::getTestConfig() const 
{
    return m_testconfig  ;
}


const char* BOpticksResource::MakeSrcPath(const char* srcpath, const char* ext) 
{
    std::string path = BFile::ChangeExt(srcpath, ext ); 
    return strdup(path.c_str());
}
const char* BOpticksResource::MakeSrcDir(const char* srcpath, const char* sub) 
{
    std::string srcdir = BFile::ParentDir(srcpath); 
    std::string path = BFile::FormPath(srcdir.c_str(), sub ); 
    return strdup(path.c_str());
}
const char* BOpticksResource::MakeTmpUserDir_(const char* sub, const char* rel) 
{
    const char* base = "/tmp" ; 
    const char* user = SSys::username(); 
    std::string path = BFile::FormPath(base, user, sub, rel ) ; 
    return strdup(path.c_str());
}

const char* BOpticksResource::MakeTmpUserDir(const char* sub, const char* rel) 
{
    assert( strcmp(sub, "opticks") == 0 ); 
    assert( rel == NULL ); 
    std::string path = BFile::FormPath("$TMP") ; 
    return strdup(path.c_str());
}




const char* BOpticksResource::OptiXCachePathDefault() 
{
    const char* base = "/var/tmp" ; 
    const char* user = SSys::username(); 
    const char* sub = "OptiXCache" ; 
    std::string path = BFile::FormPath(base, user, sub ) ; 
    return strdup(path.c_str());
}






const char* BOpticksResource::MakeUserDir(const char* sub, const char* rel) 
{
    std::string userdir = BFile::FormPath("$HOME", sub, rel) ; 
    return strdup(userdir.c_str());
}


// cannot be static as IDPATH not available statically 
const char* BOpticksResource::makeIdPathPath(const char* rela, const char* relb, const char* relc, const char* reld) 
{
    std::string path = getIdPathPath(rela, relb, relc, reld);  
    return strdup(path.c_str());
}



void BOpticksResource::setSrcDigest(const char* srcdigest)
{
    assert( srcdigest );
    m_srcdigest = strdup( srcdigest );
}




/**
BOpticksResource::setupViaKey
===============================

Invoked from OpticksResource::init only when 
BOpticksKey::SetKey called prior to Opticks instanciation, 
ie when using "--envkey" option.
There is no opticksdata srcpath or associated 
opticksdata paths for running in this direct mode.

Legacy mode alternative to this is setupViaSrc.

This is used for live/direct mode running, 
see for example CerenkovMinimal ckm- 


Formerly
------------

* first geometry + genstep collecting and writing executable is special, 
  it writes its event and genstep into a distinctive "standard" directory (resource "srcevtbase") 
  within the geocache keydir 

* all other executables sharing the same keydir can put their events underneath 
  a relpath named after the executable (resource "evtbase")   


Where to place events with shared geocache ?
-----------------------------------------------

In order to support running from a shared geocache cannot standardly
write events into geocache.

When working with lots of geometries it makes sense for events to
reside in geocache, but thats not the standard mode of operation. 
Hmm could append the geocache digest to the directory path ?

**/

void BOpticksResource::initViaKey()
{
    assert( !m_setup ) ;  
    m_setup = true ; 

    if( m_key == NULL )
    {
        LOG(fatal) << " m_key is NULL : early exit " ; 
        return ;    // temporary whilst debugging geocache creation
    }
    assert( m_key ) ;

    LOG(info) << std::endl << m_key->desc()  ;  

    m_layout = m_key->getLayout(); 


    const char* layout = BStr::itoa(m_layout) ;
    m_res->addName("OPTICKS_RESOURCE_LAYOUT", layout );

    const char* srcdigest = m_key->getDigest(); 
    setSrcDigest( srcdigest );
     
    const char* idname = m_key->getIdname() ;  // eg OpNovice_World_g4live
    assert(idname) ; 
    m_idname = strdup(idname);         
    m_res->addName("idname", m_idname ); 

    const char* idsubd = m_key->getIdsubd() ; //  eg  g4ok_gltf
    assert(idsubd) ; 
    m_idsubd = strdup(idsubd); 
    m_res->addName("idsubd", m_idsubd ); 

    const char* idfile = m_key->getIdfile() ; //  eg  g4ok.gltf
    assert(idfile) ; 
    m_idfile = strdup(idfile); 
    m_res->addName("idfile", m_idfile ); 

    const char* idgdml = m_key->getIdGDML() ; //  eg  g4ok.gdml
    assert(idgdml) ; 
    m_idgdml = strdup(idgdml); 
    m_res->addName("idgdml", m_idgdml ); 


    std::string fold = getGeocachePath(  m_idname ) ; 
    m_idfold = strdup(fold.c_str()) ; 
    m_res->addDir("idfold", m_idfold ); 

    std::string idpath = getGeocachePath( m_idname, m_idsubd, m_srcdigest, layout );
    // HMM IT WOULD BE GOOD TO ARRANGE FOR THIS TO BE STATICALLY AVAILABLE FROM THE KEY ?
    // BUT ITS A LITTLE INVOLVED DUE TO DIFFERENT POSSIBILITES FOR THE OPTICKS_INSTALL_PREFIX


    LOG(LEVEL)
           <<  " idname " << m_idname 
           <<  " idfile " << m_idfile 
           <<  " srcdigest " << m_srcdigest
           <<  " idpath " << idpath 
            ;

    m_idpath = strdup(idpath.c_str()) ; 
    m_res->addDir("idpath", m_idpath ); 

    m_gltfpath = makeIdPathPath(m_idfile) ; // not a srcpath for G4LIVE, but potential cache file 
    m_res->addPath("gltfpath", m_gltfpath ); 

    m_gdmlpath = makeIdPathPath(m_idgdml) ; // output gdml 
    m_res->addPath("gdmlpath", m_gdmlpath ); 


    m_g4codegendir = makeIdPathPath("g4codegen" );
    m_res->addDir("g4codegendir", m_g4codegendir ); 

    m_gdmlauxmetapath = makeIdPathPath("gdmlauxmeta.json");  
    m_res->addPath("gdmlauxmetapath", m_gdmlauxmetapath ); 

   


    m_cachemetapath = makeIdPathPath("cachemeta.json");  
    m_res->addPath("cachemetapath", m_cachemetapath ); 

    m_runcommentpath = makeIdPathPath("runcomment.txt");  
    m_res->addPath("runcommentpath", m_runcommentpath ); 


    m_primariespath = makeIdPathPath("primaries.npy");  
    m_res->addPath("primariespath", m_primariespath ); 

    m_directgensteppath = makeIdPathPath("directgenstep.npy");  
    m_res->addPath("directgensteppath", m_directgensteppath ); 

    m_directphotonspath = makeIdPathPath("directphotons.npy");  
    m_res->addPath("directphotonspath", m_directphotonspath ); 

    const char* exename = SProc::ExecutableName() ;

#ifdef WITH_EXENAME_ALLOWED_ASSERT
    bool exename_allowed = SStr::StartsWith(exename, "Use") || 
                           SStr::StartsWith(exename, "python") || 
                           SStr::StartsWith(exename, "OpticksEmbedded") || 
                           SStr::StartsWith(exename, "x0") || 
                           SStr::EndsWith(exename, "Test") || 
                           SStr::EndsWith(exename, "Minimal") ;  

    if(!exename_allowed)
    {
        LOG(fatal) << "exename " << exename
                   << " is not allowed "  
                   << " (this is to prevent stomping on geocache content). " 
                   << " Names starting with Use or ending with Test or Minimal are permitted" 
                   ; 
    }   
    assert( exename_allowed ); 
#endif

    // see notes/issues/opticks-event-paths.rst 
    // matching python approach to event path addressing 
    // aiming to eliminate srcevtbase and make evtbase mostly constant
    // and equal to idpath normally and OPTICKS_EVENT_BASE eg /tmp for test running
    //
    // KeySource means name of current executable is same as the one that created the geocache
    m_evtpfx = isKeySource() ? "source" : exename ; 
    m_res->addName("evtpfx", m_evtpfx ); 


    bool legacy = BOpticksResource::IsLegacyGeometryEnabled(); 
    if( legacy )
    { 
        if( !m_testgeo ) m_evtbase = m_idpath ; 
        // huh normally NULL for testgeo ?
    }
    else
    {
        m_evtbase = ResolveEventBase(); 
    }

    LOG(LEVEL) 
         << ( legacy ? " OPTICKS_LEGACY_GEOMETRY_ENABLED " : " non-legacy " )
         << " evtbase from idpath in legacy for non-testgeo, from envvar or default in non-legacy " 
         << " evtbase " << m_evtbase 
         << " idpath " << m_idpath
         << " testgeo " << m_testgeo 
         ;    

    m_res->addDir( "evtbase", m_evtbase ); 

}


bool BOpticksResource::idNameContains(const char* s) const 
{
    assert(m_idname); 
    std::string idn(m_idname);
    std::string ss(s);
    return idn.find(ss) != std::string::npos ;
}



/**
BOpticksResource::formCacheRelativePath
---------------------------------------

Shorten an absolute path into a cache relative one for easy reading.

**/

std::string BOpticksResource::formCacheRelativePath(const char* path) const
{
    const char* idpath = getIdPath();
    if(strncmp(idpath, path, strlen(idpath)) == 0)
    {      
        return path + strlen(idpath) + 1 ;
    }      
    else   
    {      
        return path ;
    }  
}

/**
BOpticksResource::getPropertyLibDir
---------------------------------------

**/

std::string BOpticksResource::getPropertyLibDir(const char* name) const 
{
    const char* idpath = getIdPath();    // direct use of m_idpath would fail to honour overrides from m_ipath_tmp 
    return BFile::FormPath( idpath, name ) ;
}


std::string BOpticksResource::getObjectPath(const char* name, unsigned int index) const
{
    const char* idpath = getIdPath();
    assert(idpath && "OpticksResource::getObjectPath idpath not set");
    fs::path dir(idpath);
    dir /= name ;
    dir /= BStr::utoa(index) ;
    return dir.string() ;
}

std::string BOpticksResource::getObjectPath(const char* name) const
{
    const char* idpath = getIdPath();
    assert(idpath && "OpticksResource::getObjectPath idpath not set");
    fs::path dir(idpath);
    dir /= name ;
    return dir.string() ;
}

std::string BOpticksResource::getRelativePath(const char* name, unsigned int index) const
{
    // used eg by GPmt::loadFromCache returning "GPmt/0"
    fs::path reldir(name);
    reldir /= BStr::utoa(index) ;
    return reldir.string() ;
}
std::string BOpticksResource::getRelativePath(const char* name) const
{
    fs::path reldir(name);
    return reldir.string() ;
}








const char* BOpticksResource::getSrcPath() const { return m_srcpath ; }
const char* BOpticksResource::getSrcDigest() const { return m_srcdigest ; }
const char* BOpticksResource::getDAEPath() const { return m_daepath ; }
const char* BOpticksResource::getGDMLPath() const { return m_gdmlpath ; }
const char* BOpticksResource::getSrcGDMLPath() const { return m_srcgdmlpath ; } 
const char* BOpticksResource::getSrcGLTFPath() const { return m_srcgltfpath ; } 

const char* BOpticksResource::getSrcGLTFBase() const
{
    std::string base = BFile::ParentDir(m_srcgltfpath) ;
    return strdup(base.c_str()); 
}
const char* BOpticksResource::getSrcGLTFName() const
{
    std::string name = BFile::Name(m_srcgltfpath) ;
    return strdup(name.c_str()); 
}

const char* BOpticksResource::getG4CodeGenDir() const { return m_g4codegendir ; }
const char* BOpticksResource::getCacheMetaPath() const { return m_cachemetapath ; }
const char* BOpticksResource::getGDMLAuxMetaPath() const { return m_gdmlauxmetapath ; }
const char* BOpticksResource::getRunCommentPath() const { return m_runcommentpath ; }
const char* BOpticksResource::getPrimariesPath() const { return m_primariespath ; } 
const char* BOpticksResource::getGLTFPath() const { return m_gltfpath ; } 



BMeta* BOpticksResource::LoadGDMLAuxMeta() const 
{
    const char* gdmlauxmetapath = getGDMLAuxMetaPath();
    LOG(LEVEL) << " gdmlauxmetapath " << gdmlauxmetapath ; 
    BMeta* gdmlauxmeta = BMeta::Load(gdmlauxmetapath) ;
    return gdmlauxmeta ; 
}

const BMeta* BOpticksResource::getGDMLAuxMeta() const 
{
    return m_gdmlauxmeta ; 
}


const char* BOpticksResource::opticks_geospecific_options = "opticks_geospecific_options" ; 

void BOpticksResource::initMetadata()
{
    m_gdmlauxmeta = LoadGDMLAuxMeta(); 
    if(m_gdmlauxmeta)
    {
        m_gdmlauxmeta_lvmeta = m_gdmlauxmeta->getObj("lvmeta"); 
        m_gdmlauxmeta_usermeta = m_gdmlauxmeta->getObj("usermeta"); 
    }
    if(m_gdmlauxmeta_usermeta)
    {
        std::string opts = m_gdmlauxmeta_usermeta->get<std::string>(opticks_geospecific_options, "");
        m_opticks_geospecific_options = opts.empty() ? NULL : strdup(opts.c_str()) ;  
    }
}

const char* BOpticksResource::getGDMLAuxUserinfoGeospecificOptions() const 
{
    return m_opticks_geospecific_options ; 
}

std::string BOpticksResource::getGDMLAuxUserinfo(const char* k) const 
{
    const BMeta* usermeta = m_gdmlauxmeta_usermeta ;  
    std::string value ;
    if(usermeta) value = usermeta->get<std::string>(k, "") ;
    return value ; 
}





/**
BOpticksResource::findGDMLAuxMetaEntries
------------------------------------------

**/
void BOpticksResource::findGDMLAuxMetaEntries(std::vector<BMeta*>& entries, const char* k, const char* v ) const
{
    const BMeta* lvmeta = m_gdmlauxmeta_lvmeta ; 

    unsigned ni = lvmeta ? lvmeta->getNumKeys() : 0 ;
    bool dump = false ;

    for(unsigned i=0 ; i < ni ; i++) 
    {    
        const char* subKey = lvmeta->getKey(i); 
        BMeta* sub = lvmeta->getObj(subKey); 

        unsigned mode = 0 ;  

        if(k == NULL && v == NULL) // both NULL : match all 
        {    
            mode = 1 ;
            entries.push_back(sub);
        }
        else if( k != NULL && v == NULL)  // key non-NULL, value NULL : match all with that key  
        {
            mode = 2 ;
            bool has_key = sub->hasKey(k);
            if(has_key) entries.push_back(sub) ;
        }
        else if( k != NULL && v != NULL)  // key non-NULL, value non-NULL : match only those with that (key,value) pair
        {
            mode = 3 ;
            bool has_key = sub->hasKey(k);
            std::string value = has_key ? sub->get<std::string>(k) : ""  ;
            if(strcmp(value.c_str(), v) == 0) entries.push_back(sub);
        }

        if(dump)
        std::cout
           << " i " << i
           << " mode " << mode
           << " subKey " << subKey
           << std::endl
           ;

        //sub->dump("Opticks::findGDMLAuxMetaEntries");  
   }
   LOG(LEVEL)
       << " ni " << ni
       << " k " << k
       << " v " << v
       << " entries.size() " << entries.size()
       ;

}

void BOpticksResource::findGDMLAuxValues(std::vector<std::string>& values, const char* k, const char* v, const char* q) const
{
    std::vector<BMeta*> entries ;
    findGDMLAuxMetaEntries(entries, k, v );

    for(unsigned i=0 ; i < entries.size() ; i++)
    {
        BMeta* entry = entries[i];
        std::string qv = entry->get<std::string>(q) ;
        values.push_back(qv);
    }
}

/**
BOpticksResource::getGDMLAuxTargetLVNames
------------------------------------------

Consults the persisted GDMLAux metadata looking for entries with (k,v) pair ("label","target").
For any such entries the "lvname" property is accesses and added to the lvnames vector.

**/

unsigned BOpticksResource::getGDMLAuxTargetLVNames(std::vector<std::string>& lvnames) const
{
    const char* k = "label" ;
    const char* v = "target" ;
    const char* q = "lvname" ;

    findGDMLAuxValues(lvnames, k,v,q);

    LOG(LEVEL)
        << " for entries matching (k,v) : " << "(" << k << "," << v << ")"
        << " collect values of q:" << q
        << " : lvnames.size() " << lvnames.size()
        ;

    return lvnames.size();
}


/**
BOpticksResource::getGDMLAuxTargetLVName
-------------------------------------------

Returns the first lvname or NULL

**/

const char* BOpticksResource::getGDMLAuxTargetLVName() const
{
    std::vector<std::string> lvnames ;
    getGDMLAuxTargetLVNames(lvnames);
    return lvnames.size() > 0 ? strdup(lvnames[0].c_str()) : NULL ;
}




const char* BOpticksResource::getIdMapPath() const { return m_idmappath ; } 

const char* BOpticksResource::getEventBase() const { return m_evtbase ; } 
const char* BOpticksResource::getEventPfx() const {  return m_evtpfx ; } 

/**
BOpticksResource::setEventBase
-------------------------------


**/


void BOpticksResource::setEventBase(const char* rela, const char* relb)
{
    std::string abs = BFile::Absolute( rela, relb );  
    m_evtbase = strdup(abs.c_str()); 
    m_res->setDir( "evtbase", m_evtbase ); 

    LOG(fatal) 
        << " rela " << rela   
        << " relb " << relb
        << " evtbase " <<  m_evtbase 
        ;

    std::raise(SIGINT); 
 
}
void BOpticksResource::setEventPfx(const char* pfx)
{
    m_evtpfx = strdup(pfx); 
    m_res->setName( "evtpfx", m_evtpfx ); 
}


bool  BOpticksResource::hasKey() const
{
    return m_key != NULL ; 
}
BOpticksKey*  BOpticksResource::getKey() const
{
    return m_key ; 
}
const char* BOpticksResource::getKeySpec() const 
{
    return m_key ? m_key->getSpec() : NULL ; 
}
std::string BOpticksResource::export_() const 
{
    assert(m_key); 
    return m_key->export_(); 
}

bool BOpticksResource::isKeySource() const   // name of current executable matches that of the creator of the geocache
{
    return m_key ? m_key->isKeySource() : false ; 
}

/**
BOpticksResource::isKeyLive
-----------------------------

Only true when Opticks::SetKey used with a non-NULL spec. This is the 
case when operating from a live Geant4 geometry, such as when 
creating a geocache with geocache-create.

**/

bool BOpticksResource::isKeyLive() const  
{
    return m_key ? m_key->isLive() : false ; 
}



void BOpticksResource::setIdPathOverride(const char* idpath_tmp)  // used for test saves into non-standard locations
{
   m_idpath_tmp = idpath_tmp ? strdup(idpath_tmp) : NULL ;  
} 
const char* BOpticksResource::getIdPath() const 
{
    LOG(verbose) << "getIdPath"
              << " idpath_tmp " << m_idpath_tmp 
              << " idpath " << m_idpath
              ; 

    return m_idpath_tmp ? m_idpath_tmp : m_idpath  ;
}
const char* BOpticksResource::getIdFold() const 
{
    return m_idfold ;
}



void BOpticksResource::Brief(const char* msg) const 
{
    std::cerr << msg << std::endl ; 
    std::cerr << "install_prefix    : " <<  (m_install_prefix ? m_install_prefix : "NULL" ) << std::endl ; 
    std::cerr << "opticksdata_dir   : " <<  (m_opticksdata_dir ? m_opticksdata_dir : "NULL" ) << std::endl ; 
    std::cerr << "geocache_dir      : " <<  (m_geocache_dir ? m_geocache_dir : "NULL" ) << std::endl ; 
    std::cerr << "resource_dir      : " <<  (m_resource_dir ? m_resource_dir : "NULL" ) << std::endl ; 

    std::cerr << "daepath  : " <<  (m_daepath?m_daepath:"NULL") << std::endl; 
    std::cerr << "gdmlpath : " <<  (m_gdmlpath?m_gdmlpath:"NULL") << std::endl; 
    std::cerr << "gltfpath : " <<  (m_gltfpath?m_gltfpath:"NULL") << std::endl; 
    std::cerr << "digest   : " <<  (m_srcdigest?m_srcdigest:"NULL") << std::endl; 
    std::cerr << "idpath   : " <<  (m_idpath?m_idpath:"NULL") << std::endl; 
    std::cerr << "idpath_tmp " <<  (m_idpath_tmp?m_idpath_tmp:"NULL") << std::endl; 
    std::cerr << "idfold   : " <<  (m_idfold?m_idfold:"NULL") << std::endl; 
    std::cerr << "idname   : " <<  (m_idname?m_idname:"NULL") << std::endl; 

    m_res->dumpPaths("dumpPaths");
    m_res->dumpDirs("dumpDirs");

}

std::string BOpticksResource::desc() const
{
    std::stringstream ss ;

    std::time_t* slwt = BFile::SinceLastWriteTime(m_idpath);
    long seconds = slwt ? *slwt : -1 ;

    float minutes = float(seconds)/float(60) ;
    float hours = float(seconds)/float(60*60) ;
    float days = float(seconds)/float(60*60*24) ;

    ss << "cache.SinceLastWriteTime"
       << " digest " << ( m_srcdigest ? m_srcdigest : "NULL" )
       << " seconds " << std::setw(6) << seconds
       << std::fixed << std::setprecision(3)
       << " minutes " << std::setw(6) << minutes
       << " hours " << std::setw(6) << hours
       << " days " << std::setw(10) << days
       ;

    return ss.str();
}




void BOpticksResource::Summary(const char* msg) const 
{
    LOG(info) << msg << " layout " << m_layout ; 

    const char* prefix = m_install_prefix ; 

    std::cerr << "prefix   : " <<  (prefix ? prefix : "NULL" ) << std::endl ; 

    std::cerr << "debugging_idpath  " << ( m_debugging_idpath ? m_debugging_idpath : "-" )<< std::endl ; 
    std::cerr << "debugging_idfold  " << ( m_debugging_idfold ? m_debugging_idfold : "-" )<< std::endl ; 

    std::string usertmpdir = BFile::FormPath("$TMP") ; 
    std::cerr << "usertmpdir ($TMP) " <<  usertmpdir << std::endl ; 

    std::string usertmptestdir = BFile::FormPath("$TMPTEST") ; 
    std::cerr << "($TMPTEST)        " <<  usertmptestdir << std::endl ; 


    m_res->dumpPaths("dumpPaths");
    m_res->dumpDirs("dumpDirs");
    m_res->dumpNames("dumpNames");

}

const char* BOpticksResource::MakePath( const char* prefix, const char* main, const char* sub )  // static
{
    fs::path ip(prefix);   
    if(main) ip /= main ;        
    if(sub)  ip /= sub  ; 

    std::string path = ip.string();
    return strdup(path.c_str());
}

std::string BOpticksResource::BuildDir(const char* proj)
{
    return BFile::FormPath(ResolveInstallPrefix(), "build", proj );
}
std::string BOpticksResource::BuildProduct(const char* proj, const char* name)
{
    std::string builddir = BOpticksResource::BuildDir(proj);
    return BFile::FormPath(builddir.c_str(), name);
}


