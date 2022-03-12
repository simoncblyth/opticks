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


#include <cstring>
#include <cassert>
#include <algorithm>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


// sysrap
#include "SLog.hh"
#include "SDigest.hh"
#include "SSys.hh"

// brap-
#include "BFile.hh"
#include "BStr.hh"
#include "PLOG.hh"
#include "Map.hh"
#include "BOpticksResource.hh"
#include "BResource.hh"
#include "SOpticksKey.hh"
#include "BEnv.hh"

// npy-
#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "Typ.hpp"
#include "Types.hpp"
#include "Index.hpp"


#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksQuery.hh"
#include "OpticksColors.hh"
#include "OpticksFlags.hh"
#include "OpticksAttrSeq.hh"


const plog::Severity OpticksResource::LEVEL = PLOG::EnvLevel("OpticksResource", "DEBUG") ; 

/*
OpticksResource::OpticksResource
----------------------------------

Instanciated by Opticks::initResource during configure.

TODO
~~~~~

0. arrange for separate BOpticksResource that this picks up, which is instanciated early, prior to Opticks instanciation 

1. migrate everything that does not need the Opticks instance (ie the commandline arguments) 
   down into the base class  BOpticksResource

2. slim this class to almost nothing, leaving only late stage setup post Opticks::configure
   when commandline has been parsed.

3. replace use of boost::filesystem with BFile

*/

OpticksResource::OpticksResource(Opticks* ok) 
    :
    m_log(new SLog("OpticksResource::OpticksResource","",debug)),
    m_rsc(BOpticksResource::Get(NULL)),   // use prior instance or create if not existing
    m_key(m_rsc->getKey()),
    m_ok(ok),
    m_allownokey(ok->isAllowNoKey()),
    m_query(new OpticksQuery("all")),
    m_valid(true),
    m_colors(NULL),
    m_flags(new OpticksFlags),
    m_flagnames(NULL),
    m_types(NULL),
    m_typ(NULL),
    m_g4env(NULL),
    m_okenv(NULL),
    m_runresultsdir(NULL)
{
    init();
    (*m_log)("DONE"); 
}

/**
OpticksResource::init
-----------------------

*/

void OpticksResource::init()
{
   LOG(LEVEL) << "OpticksResource::init" ; 

   BStr::split(m_detector_types, "GScintillatorLib,GMaterialLib,GSurfaceLib,GBndLib,GSourceLib", ',' ); 
   BStr::split(m_resource_types, "GFlags,OpticksColors", ',' ); 


   if( m_allownokey )
   {
       LOG(fatal) << " CAUTION : are allowing no key " ; 
   } 
   else
   {
       assert( m_rsc->hasKey() && "an OPTICKS_KEY is required" );
   }

   initRunResultsDir(); 

   LOG(LEVEL) << "OpticksResource::init DONE" ; 
}

/**
OpticksResource::initRunResultsDir
-----------------------------------

Depends on commandline args, so this needs to happen late (here) 
and cannot be moved down to m_rsc.

**/

void OpticksResource::initRunResultsDir()
{
    const char* runfolder = m_ok->getRunFolder();  // eg geocache-bench 
    const char* runlabel = m_ok->getRunLabel();    // eg OFF_TITAN_V_AND_TITAN_RTX
    const char* rundate = m_ok->getRunDate() ;     // eg 20190422_162401 

    std::string runresultsdir = m_rsc->getResultsPath( runfolder, runlabel, rundate ) ;  // eg /usr/local/opticks/results/geocache-bench/OFF_TITAN_V_AND_TITAN_RTX/20190422_162401

    m_runresultsdir = strdup(runresultsdir.c_str());
    LOG(LEVEL) << runresultsdir ; 
}


/**
OpticksResource::getRunResultsDir
-----------------------------------

Used from OTracer::report 

**/
const char* OpticksResource::getRunResultsDir() const 
{
    return m_runresultsdir ;
}

bool OpticksResource::isDetectorType(const char* type_)
{
    return std::find(m_detector_types.begin(),m_detector_types.end(), type_) != m_detector_types.end()  ; 
}
bool OpticksResource::isResourceType(const char* type_)
{
    return std::find(m_resource_types.begin(),m_resource_types.end(), type_) != m_resource_types.end()  ; 
}



void OpticksResource::setValid(bool valid)
{
    m_valid = valid ; 
}
bool OpticksResource::isValid()
{
   return m_valid ; 
}
OpticksQuery* OpticksResource::getQuery() const 
{
    return m_query ;
}






/**
OpticksResource::getDefaultFrame
---------------------------------

This must come from detector specific config

**/

int OpticksResource::getDefaultFrame() const 
{
    LOG(LEVEL) << " PLACEHOLDER ZERO " ; 
    return 0 ; 
}


void OpticksResource::SetupG4Environment()
{
    // NB this relpath needs to match that in g4-;g4-export-ini
    //    it is relative to the install_prefix which 
    //    is canonically /usr/local/opticks
    //
    const char* inipath = BOpticksResource::InstallPathG4ENV();

    LOG(error) << "inipath " << inipath ; 

    BEnv* g4env = ReadIniEnvironment(inipath);
    if(g4env)
    {
        g4env->setEnvironment();
    }
    else
    {
        LOG(error)
                     << " MISSING inipath " << inipath
                     << " (create it with bash functions: g4-;g4-export-ini ) " 
                     ;
    }
}



/**
OpticksResource::readOpticksEnvironment
-------------------------------------------

NB this relpath needs to match that in opticksdata-;opticksdata-export-ini
   it is relative to the install_prefix which 
   is canonically /usr/local/opticks

Formerly used to setup OPTICKSDATA_DAEPATH envvars::

    [blyth@localhost opticks]$ cat opticksdata/config/opticksdata.ini
    OPTICKSDATA_DAEPATH_DFAR=/home/blyth/local/opticks/opticksdata/export/Far_VGDX_20140414-1256/g4_00.dae
    OPTICKSDATA_DAEPATH_DLIN=/home/blyth/local/opticks/opticksdata/export/Lingao_VGDX_20140414-1247/g4_00.dae
    OPTICKSDATA_DAEPATH_DPIB=/home/blyth/local/opticks/opticksdata/export/dpib/cfg4.dae
    OPTICKSDATA_DAEPATH_DYB=/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    OPTICKSDATA_DAEPATH_J1707=/home/blyth/local/opticks/opticksdata/export/juno1707/g4_00.dae
    OPTICKSDATA_DAEPATH_J1808=/home/blyth/local/opticks/opticksdata/export/juno1808/g4_00.dae
    OPTICKSDATA_DAEPATH_JPMT=/home/blyth/local/opticks/opticksdata/export/juno/test3.dae
    OPTICKSDATA_DAEPATH_LXE=/home/blyth/local/opticks/opticksdata/export/LXe/g4_00.dae
    [blyth@localhost opticks]$ 

**/

void OpticksResource::readOpticksEnvironment()
{
    // assert(0) ; // around 90/412 fails with this in place

    const char* inipath = BOpticksResource::InstallPathOKDATA();
    LOG(LEVEL) << " inipath " << inipath ; 

    m_okenv = ReadIniEnvironment(inipath);
    if(m_okenv)
    {
        m_okenv->setEnvironment();
    }
    else
    {
        LOG(error)
                     << " MISSING inipath " << inipath 
                     << " (create it with bash functions: opticksdata-;opticksdata-export-ini ) " 
                     ;
    }
}

BEnv* OpticksResource::ReadIniEnvironment(const std::string& inipath)
{
    BEnv* env = NULL ; 
    if(BFile::ExistsFile(inipath.c_str()))
    {
        LOG(verbose) 
                  << " from " << inipath
                  ;

         env = BEnv::Load(inipath.c_str()); 
         //env->dump("OpticksResource::ReadIniEnvironment");

    }
    return env ;  
}


void OpticksResource::Dump(const char* msg)
{
    Summary(msg);

    std::string mmsp = getMergedMeshPath(0);
    std::cout 
           << std::setw(40) << " getMergedMeshPath(0) " 
           << " : " 
           << mmsp
           << std::endl ; 


    std::string bndlib = m_rsc->getPropertyLibDir("GBndLib");
    std::cout 
           << std::setw(40) << " getPropertyLibDir(\"GBndLib\") " 
           << " : " 
           << bndlib 
           << std::endl ; 

    const char* testcsgpath = m_rsc->getTestCSGPath() ; 
    std::cout 
           << std::setw(40) << " getTestCSGPath() " 
           << " : " 
           <<  ( testcsgpath ? testcsgpath : "NULL" )
           << std::endl ; 
    
}



void OpticksResource::Summary(const char* msg)
{
    std::cerr << msg << std::endl ; 
    std::cerr << "valid    : " <<  (m_valid ? "valid" : "NOT VALID" ) << std::endl ; 
    typedef std::map<std::string, std::string> SS ;
    for(SS::const_iterator it=m_metadata.begin() ; it != m_metadata.end() ; it++)
        std::cerr <<  std::setw(10) << it->first.c_str() << ":" <<  it->second.c_str() << std::endl  ;

}

std::string OpticksResource::getMergedMeshPath(unsigned int index)
{
    return getObjectPath("GMergedMesh", index);
}

const char* OpticksResource::getIdPath() const
{
    return m_rsc->getIdPath(); 
}
std::string OpticksResource::getPropertyLibDir(const char* name) const
{
    return m_rsc->getPropertyLibDir(name); 
}
std::string OpticksResource::getObjectPath(const char* name, unsigned int index) const 
{
    return m_rsc->getObjectPath(name, index); 
}

std::string OpticksResource::getObjectPath(const char* name) const 
{
    return m_rsc->getObjectPath(name); 
}

std::string OpticksResource::getRelativePath(const char* name, unsigned int index) const 
{
    return m_rsc->getRelativePath(name, index); 
}

std::string OpticksResource::getRelativePath(const char* name) const 
{
    return m_rsc->getRelativePath(name); 
}


/**
OpticksResource::getPreferenceDir
-------------------------------------

::

    find /usr/local/opticks/opticksdata/export/DayaBay -name order.json
    /usr/local/opticks/opticksdata/export/DayaBay/GMaterialLib/order.json
    /usr/local/opticks/opticksdata/export/DayaBay/GSurfaceLib/order.json

    ## important materials arranged to have low indices, 
    ## for 4-bit per step GPU recording of indices

    epsilon:optickscore blyth$ cat /usr/local/opticks/opticksdata/export/DayaBay/GMaterialLib/order.json
    {
        "GdDopedLS": "1",
        "LiquidScintillator": "2",
        "Acrylic": "3",
        "MineralOil": "4",
        "Bialkali": "5",
        "IwsWater": "6",
        "Water": "7",
        ...

**/

std::string OpticksResource::getPreferenceDir(const char* type, const char* udet, const char* subtype ) const 
{
    const char* prefbase = BOpticksResource::PREFERENCE_BASE ;   // $HOME/.opticks

    fs::path prefdir(prefbase) ;
    if(udet) prefdir /= udet ;
    prefdir /= type ; 
    if(subtype) prefdir /= subtype ; 
    std::string pdir = prefdir.string() ;

    LOG(LEVEL)
        << " type " << type 
        << " udet " << udet
        << " subtype " << subtype 
        << " pdir " << pdir 
        ;

    return pdir ; 
}



/*

GPropLib/GAttrSeq prefs such as materials, surfaces, boundaries and flags 
come in threes (abbrev.json, color.json and order.json)
these provided attributes for named items in sequences. 

::

    delta:GMaterialLib blyth$ cat ~/.opticks/GMaterialLib/abbrev.json 
    {
        "ADTableStainlessSteel": "AS",
        "Acrylic": "Ac",
        "Air": "Ai",
        "Aluminium": "Al",


Some such triplets like GMaterialLib, GSurfaceLib belong within "Detector scope" 
as they depend on names used within a particular detector. 

* not within IDPATH as changing geo selection doesnt change names
* not within user scope, as makes sense to standardize 

Moved from ~/.opticks into opticksdata/export/<detname>  m_detector_base
by env-;export-;export-copy-detector-prefs-

::

    delta:GMaterialLib blyth$ l /usr/local/opticks/opticksdata/export/DayaBay/
    drwxr-xr-x  3 blyth  staff  102 Jul  5 10:58 GPmt



*/



bool OpticksResource::loadPreference(std::map<std::string, std::string>& mss, const char* type, const char* name)
{
    std::string prefdir = getPreferenceDir(type);
    bool empty = prefdir.empty() ; 

    LOG(LEVEL)
        << " (MSS) " 
        << " prefdir " << prefdir
        << " name " << name
        << " empty " << ( empty ? "YES" : "NO" )
        ; 


    typedef Map<std::string, std::string> MSS ;  
    MSS* pref = empty ? NULL :  MSS::load(prefdir.c_str(), name ) ; 
    if(pref)
    {
        mss = pref->getMap(); 
    }

    return pref != NULL ; 
}

bool OpticksResource::loadPreference(std::map<std::string, unsigned int>& msu, const char* type, const char* name)
{
    std::string prefdir = getPreferenceDir(type);
    bool empty = prefdir.empty() ; 

    LOG(LEVEL)
        << " (MSU) " 
        << " prefdir " << prefdir
        << " name " << name
        << " empty " << ( empty ? "YES" : "NO" )
        ; 


    typedef Map<std::string, unsigned int> MSU ;  
    MSU* pref = empty ? NULL : MSU::load(prefdir.c_str(), name ) ; 
    if(pref)
    {
        msu = pref->getMap(); 
    }
    return pref != NULL ; 
}

bool OpticksResource::existsFile(const char* path)
{
    fs::path fpath(path);
    return fs::exists(fpath ) && fs::is_regular_file(fpath) ;
}

bool OpticksResource::existsFile(const char* dir, const char* name)
{
    fs::path fpath(dir);
    fpath /= name ; 
    return fs::exists(fpath ) && fs::is_regular_file(fpath) ;
}

bool OpticksResource::existsDir(const char* path)
{
    fs::path fpath(path);
    return fs::exists(fpath ) && fs::is_directory(fpath) ;
}

std::string OpticksResource::makeSidecarPath(const char* path, const char* styp, const char* dtyp)
{
   std::string empty ; 

   fs::path src(path);
   fs::path ext = src.extension();
   bool is_styp = ext.string().compare(styp) == 0  ;

   assert(is_styp && "OpticksResource::makeSidecarPath source file type doesnt match the path file type");

   fs::path dst(path);
   dst.replace_extension(dtyp) ;

   /*
   LOG(debug) << "OpticksResource::makeSidecarPath"
             << " styp " << styp
             << " dtyp " << dtyp
             << " ext "  << ext 
             << " src " << src.string()
             << " dst " << dst.string()
             << " is_styp "  << is_styp 
             ;
   */

   return dst.string() ;
}

bool OpticksResource::loadMetadata(std::map<std::string, std::string>& mdd, const char* path)
{
    typedef Map<std::string, std::string> MSS ;  
    MSS* meta = MSS::load(path) ; 
    if(meta)
    {
        mdd = meta->getMap(); 
    }
    else
    {
        LOG(debug) << "OpticksResource::loadMetadata"
                  << " no path " << path 
                  ;
    } 
    return meta != NULL ; 
}

void OpticksResource::dumpMetadata(std::map<std::string, std::string>& mdd)
{
    typedef std::map<std::string, std::string> SS ;
    for(SS::const_iterator it=mdd.begin() ; it != mdd.end() ; it++)
    {
       std::cout
             << std::setw(20) << it->first 
             << std::setw(20) << it->second
             << std::endl ; 
    }
}


bool OpticksResource::hasMetaKey(const char* key)
{
    return m_metadata.count(key) == 1 ; 
}
const char* OpticksResource::getMetaValue(const char* key)
{
    return m_metadata.count(key) == 1 ? m_metadata[key].c_str() : NULL ;
}


/**
OpticksResource::loadColorMapFromPrefs
-----------------------------------------

The formerly named GCache is exceptionally is not an attribution triplet,
to reflect this manually changed name to OpticksColors

**/

OpticksColors* OpticksResource::loadColorMapFromPrefs()
{

    std::string prefdir = getPreferenceDir("OpticksColors"); 
    bool empty = prefdir.empty(); 
    OpticksColors* colors = empty ? NULL :  OpticksColors::load(prefdir.c_str(),"OpticksColors.json");   // colorname => hexcode
    if(empty)
    {
        LOG(debug) 
                   << " empty PreferenceDir for OpticksColors " 
                   ;
    }
    return colors ; 
}


OpticksColors* OpticksResource::getColors()
{
    if(!m_colors)
    {
        // deferred to avoid output prior to logging setup
        //m_colors = loadColorMapFromPrefs() ;  
        m_colors = OpticksColors::LoadMeta() ;  
    }
    return m_colors ;
}

OpticksFlags* OpticksResource::getFlags() const 
{
    return m_flags ;
}

void OpticksResource::saveFlags(const char* dir)
{
    OpticksFlags* flags = getFlags();
    LOG(info) << " dir " << dir ;
    flags->save(dir);
}

OpticksAttrSeq* OpticksResource::getFlagNames()
{
    if(!m_flagnames)
    {
        OpticksFlags* flags = getFlags();
        BMeta* abbrev = flags->getAbbrevMeta(); 
        BMeta* color = flags->getColorMeta(); 

        Index* index = flags->getIndex();

        m_flagnames = new OpticksAttrSeq(m_ok, "GFlags");
        m_flagnames->setAbbrevMeta(abbrev);   // added flag abbrevs as the abbrev.json missing in direct workflow 
        m_flagnames->setColorMeta(color) ; 


        m_flagnames->loadPrefs(); // color, abbrev and order  <-- missing in direct

        m_flagnames->setSequence(index);
        m_flagnames->setCtrl(OpticksAttrSeq::SEQUENCE_DEFAULTS);    
    }
    return m_flagnames ; 
}

std::map<unsigned int, std::string> OpticksResource::getFlagNamesMap()
{
    OpticksAttrSeq* flagnames = getFlagNames();
    return flagnames->getNamesMap() ; 
}

Typ* OpticksResource::getTyp()
{
    if(m_typ == NULL)
    {   
       m_typ = new Typ ; 
    }   
    return m_typ ; 
}

Types* OpticksResource::getTypes()
{
    if(!m_types)
    {   
        // deferred because idpath not known at init ?
        m_types = new Types ;   
    }   
    return m_types ;
}

void OpticksResource::saveTypes(const char* dir)
{
    LOG(info) << "OpticksResource::saveTypes " << dir ; 
    Types* types = getTypes(); 
    types->saveFlags(dir, ".ini");
}

/**
OpticksResource was formerly a subclass of BOpticksResource ... replaced the
inheritance relationship with constituent m_rsc for better flexibility 
as are aiming to eventually make OpticksResource disappear.
**/

BOpticksResource* OpticksResource::getRsc() const { return m_rsc ; }
const char* OpticksResource::getTestCSGPath() const { return m_rsc->getTestCSGPath() ; }
const char* OpticksResource::getTestConfig()  const { return m_rsc->getTestConfig() ; }

void OpticksResource::setTestCSGPath(const char* path){  m_rsc->setTestCSGPath(path) ; }
void OpticksResource::setTestConfig(const char* config){  m_rsc->setTestConfig(config) ; }



