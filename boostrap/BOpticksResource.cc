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

#include "BFile.hh"
#include "BStr.hh"
#include "BPath.hh"
#include "BEnv.hh"
#include "BResource.hh"
#include "BOpticksResource.hh"
#include "BOpticksKey.hh"

#include "PLOG.hh"

const plog::Severity BOpticksResource::LEVEL = PLOG::EnvLevel("BOpticksResource", "DEBUG") ; 

const char* BOpticksResource::G4ENV_RELPATH = "externals/config/geant4.ini" ;
const char* BOpticksResource::OKDATA_RELPATH = "opticksdata/config/opticksdata.ini" ; 


const char* BOpticksResource::EMPTY  = "" ; 

const char* BOpticksResource::G4LIVE  = "g4live" ; 
const char* BOpticksResource::JUNO    = "juno1707" ; 
const char* BOpticksResource::DAYABAY = "dayabay" ; 
const char* BOpticksResource::DPIB    = "PmtInBox" ; 
const char* BOpticksResource::OTHER   = "other" ; 

const char* BOpticksResource::PREFERENCE_BASE = "$HOME/.opticks" ; 


// TODO: having these defaults compiled in is problematic, better to read in from
//       json/ini so they are available at python level

const char* BOpticksResource::DEFAULT_GEOKEY = "OPTICKSDATA_DAEPATH_DYB" ; 
const char* BOpticksResource::DEFAULT_QUERY = "range:3153:12221" ; 
const char* BOpticksResource::DEFAULT_QUERY_LIVE = "all" ;
//const char* BOpticksResource::DEFAULT_QUERY_LIVE = "range:3153:12221" ;  // <-- EXPEDIENT ASSUMPTION THAT G4LIVE GEOMETRY IS DYB  
const char* BOpticksResource::DEFAULT_CTRL = "" ; 
const char* BOpticksResource::DEFAULT_MESHFIX = "iav,oav" ; 
const char* BOpticksResource::DEFAULT_MESHFIX_CFG = "100,100,10,-0.999" ; 

const char* BOpticksResource::DEFAULT_MATERIAL_DYB  = "GdDopedLS" ; 
const char* BOpticksResource::DEFAULT_MATERIAL_JUNO = "LS" ; 
const char* BOpticksResource::DEFAULT_MATERIAL_OTHER = "Water" ; 

const char* BOpticksResource::DEFAULT_MEDIUM_DYB  = "MineralOil" ; 
const char* BOpticksResource::DEFAULT_MEDIUM_JUNO = "Water" ; 
const char* BOpticksResource::DEFAULT_MEDIUM_OTHER = "Water" ; 

const char* BOpticksResource::EXAMPLE_MATNAMES_DYB = "GdDopedLS,Acrylic,LiquidScintillator,MineralOil,Bialkali" ;
const char* BOpticksResource::EXAMPLE_MATNAMES_JUNO = "LS,Acrylic" ; 
const char* BOpticksResource::EXAMPLE_MATNAMES_OTHER = "LS,Acrylic" ; 

const char* BOpticksResource::SENSOR_SURFACE_DYB = "lvPmtHemiCathodeSensorSurface" ;
const char* BOpticksResource::SENSOR_SURFACE_JUNO = "SS-JUNO-UNKNOWN" ; 
const char* BOpticksResource::SENSOR_SURFACE_OTHER = "SS-OTHER-UNKNOWN" ; 

const int BOpticksResource::DEFAULT_FRAME_OTHER = 0 ; 
const int BOpticksResource::DEFAULT_FRAME_DYB = 3153 ; 
const int BOpticksResource::DEFAULT_FRAME_JUNO = 62593 ; 


/**
BOpticksResource::BOpticksResource
----------------------------------------

Instanciated as base class of okc/OpticksResource 

TODO: eliminate testgeo param 

**/


BOpticksResource::BOpticksResource(bool testgeo)
    :
    m_testgeo(testgeo), 
    m_log(new SLog("BOpticksResource::BOpticksResource","",debug)),
    m_setup(false),
    m_key(BOpticksKey::GetKey()),   // will be NULL unless BOpticksKey::SetKey has been called 
    m_id(NULL),
    m_res(new BResource),
    m_layout(SSys::getenvint("OPTICKS_RESOURCE_LAYOUT", 0)),
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
    //m_okc_installcache_dir(NULL),
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
    //m_srcevtbase(NULL),
    m_evtbase(NULL),
    m_evtpfx(NULL),
    m_debugging_idpath(NULL),
    m_debugging_idfold(NULL),
    m_daepath(NULL),
    m_gdmlpath(NULL),
    m_srcgdmlpath(NULL),
    m_srcgltfpath(NULL),
    m_metapath(NULL),
    m_idmappath(NULL),
    m_g4codegendir(NULL),
    m_cachemetapath(NULL),
    m_runcommentpath(NULL),
    m_primariespath(NULL),
    m_directgensteppath(NULL),
    m_directphotonspath(NULL),
    m_gltfpath(NULL),
    m_testcsgpath(NULL),
    m_testconfig(NULL)
{
    init();
    (*m_log)("DONE"); 
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


const char* BOpticksResource::DEFAULT_TARGET_KEY = "OPTICKS_DEFAULT_TARGET" ; 
int BOpticksResource::DefaultTarget() // static
{
   int target = SSys::getenvint(DEFAULT_TARGET_KEY, 0);
   return target  ; 
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


/**
BOpticksResource::setSrcPath  THIS IS SLATED FOR REMOVAL
-----------------------------------------------------------

Invoked by setupViaSrc or setupViaID

example srcpath : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
NB not in geocache, points to actual G4DAE export from opticksdata 

**/

void BOpticksResource::setSrcPath(const char* srcpath)  
{
    assert( srcpath );
    m_srcpath = strdup( srcpath );

    std::string srcfold = BFile::ParentDir(m_srcpath);
    m_srcfold = strdup(srcfold.c_str());

    std::string srcbase = BFile::ParentDir(srcfold.c_str());
    m_srcbase = strdup(srcbase.c_str());

    m_res->addDir("srcfold", m_srcfold ); 
    m_res->addDir("srcbase", m_srcbase ); 

    m_daepath = MakeSrcPath(m_srcpath,".dae"); 
    m_srcgdmlpath = MakeSrcPath(m_srcpath,".gdml"); 
    m_srcgltfpath = MakeSrcPath(m_srcpath,".gltf"); 
    m_metapath = MakeSrcPath(m_srcpath,".ini"); 
    m_idmappath = MakeSrcPath(m_srcpath,".idmap"); 


    m_res->addPath("srcpath", m_srcpath );
    m_res->addPath("daepath", m_daepath );
    m_res->addPath("srcgdmlpath", m_srcgdmlpath );
    m_res->addPath("srcgltfpath", m_srcgltfpath );
    m_res->addPath("metapath", m_metapath );
    m_res->addPath("idmappath", m_idmappath );

    m_g4codegendir = MakeSrcDir(m_srcpath,"g4codegen"); 
    m_res->addDir("g4codegendir", m_g4codegendir ); 

    std::string idname = BFile::ParentName(m_srcpath);
    m_idname = strdup(idname.c_str());   // idname is name of dir containing the srcpath eg DayaBay_VGDX_20140414-1300

    std::string idfile = BFile::Name(m_srcpath);
    m_idfile = strdup(idfile.c_str());    // idfile is name of srcpath geometry file, eg g4_00.dae

    m_res->addName("idname", m_idname ); 
    m_res->addName("idfile", m_idfile ); 
}

void BOpticksResource::setSrcDigest(const char* srcdigest)
{
    assert( srcdigest );
    m_srcdigest = strdup( srcdigest );
}


void BOpticksResource::setupViaID(const char* idpath)
{
    assert( !m_setup );
    m_setup = true ; 

    m_id = new BPath( idpath ); // juicing the IDPATH
    const char* srcpath = m_id->getSrcPath(); 
    const char* srcdigest = m_id->getSrcDigest(); 

    setSrcPath( srcpath );
    setSrcDigest( srcdigest );
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

void BOpticksResource::setupViaKey()
{
    assert( !m_setup ) ;  
    m_setup = true ; 
    assert( m_key ) ; // BOpticksResource::setupViaKey called with a NULL key 

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
    bool exename_allowed = SStr::StartsWith(exename, "Use") || SStr::EndsWith(exename, "Test") || SStr::EndsWith(exename, "Minimal") ;  

    if(!exename_allowed)
    {
        LOG(fatal) << "exename " << exename
                   << " is not allowed "  
                   << " (this is to prevent stomping on geocache content). " 
                   << " Names starting with Use or ending with Test or Minimal are permitted" 
                   ; 
    }   
    assert( exename_allowed ); 

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


/*
    m_srcevtbase = makeIdPathPath("source"); 
    m_res->addDir( "srcevtbase", m_srcevtbase ); 

    // KeySource means name of current executable is same as the one that created the geocache
    m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath(exename ) ;  
    m_res->addDir( "evtbase", m_evtbase ); 
*/

}






/**
BOpticksResource::setupViaSrc  LEGACY approach to resource setup, based on envvars pointing at src dae
------------------------------------------------------------------------------------------------------------

Invoked from OpticksResource::init OpticksResource::readEnvironment

Direct mode equivalent (loosely speaking) is setupViaKey

**/

void BOpticksResource::setupViaSrc(const char* srcpath, const char* srcdigest)
{  
    LOG(LEVEL) 
        << " srcpath " << srcpath 
        << " srcdigest " << srcdigest
        ;

 
    assert( !m_setup );
    m_setup = true ; 

    setSrcPath(srcpath);
    setSrcDigest(srcdigest);
    
    const char* layout = BStr::itoa(m_layout) ;
    m_res->addName("OPTICKS_RESOURCE_LAYOUT", layout );


    if(m_layout == 0)  // geocache co-located with the srcpath typically from opticksdata
    {
        m_idfold = strdup(m_srcfold);
     
        std::string kfn = BStr::insertField( m_srcpath, '.', -1 , m_srcdigest );
        m_idpath = strdup(kfn.c_str());
     
        // IDPATH envvar setup for legacy workflow now done in Opticks::initResource
    } 
    else if(m_layout > 0)  // geocache decoupled from opticksdata
    {
        std::string fold = getGeocachePath(  m_idname ) ; 
        m_idfold = strdup(fold.c_str()) ; 

        std::string idpath = getGeocachePath( m_idname, m_idfile, m_srcdigest, layout );
        m_idpath = strdup(idpath.c_str()) ; 
    }

    m_res->addDir("idfold", m_idfold );
    m_res->addDir("idpath", m_idpath );

    m_res->addDir("idpath_tmp", m_idpath_tmp );


    m_gltfpath = makeIdPathPath("ok.gltf") ;
    m_res->addPath("gltfpath", m_gltfpath ); 

    m_cachemetapath = makeIdPathPath("cachemeta.json");  
    m_res->addPath("cachemetapath", m_cachemetapath ); 

    m_runcommentpath = makeIdPathPath("runcomment.txt");  
    m_res->addPath("runcommentpath", m_runcommentpath ); 





/**
Legacy mode equivalents for resource dirs:

srcevtbase 
    directory within opticksdata with the gensteps ?

evtbase
    user tmp directory for outputting events 

**/


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
const char* BOpticksResource::getRunCommentPath() const { return m_runcommentpath ; }
const char* BOpticksResource::getPrimariesPath() const { return m_primariespath ; } 
const char* BOpticksResource::getGLTFPath() const { return m_gltfpath ; } 
const char* BOpticksResource::getMetaPath() const { return m_metapath ; }
const char* BOpticksResource::getIdMapPath() const { return m_idmappath ; } 

//const char* BOpticksResource::getSrcEventBase() const { return m_srcevtbase ; } 
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

bool BOpticksResource::isKeySource() const   // name of current executable matches that of the creator of the geocache
{
    return m_key ? m_key->isKeySource() : false ; 
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


void BOpticksResource::Summary(const char* msg)
{
    LOG(info) << msg << " layout " << m_layout ; 

    const char* prefix = m_install_prefix ; 

    std::cerr << "prefix   : " <<  (prefix ? prefix : "NULL" ) << std::endl ; 

/*
    const char* name = "generate.cu.ptx" ;
    std::string ptxpath = getPTXPath(name); 
    std::cerr << "getPTXPath(" << name << ") = " << ptxpath << std::endl ;   

    std::string ptxpath_static = PTXPath(name); 
    std::cerr << "PTXPath(" << name << ") = " << ptxpath_static << std::endl ;   
*/


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


