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



#ifdef _MSC_VER
// object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )
#endif


#include <csignal>
#include <sstream>
#include <iostream>

#include "OKConf.hh"

#include "SLog.hh"
#include "SProc.hh"
#include "SArgs.hh"
#include "STime.hh"
#include "SSys.hh"
// brap-
#include "BTimeKeeper.hh"

#include "NMeta.hpp"

#include "BDynamicDefine.hh"
#include "BOpticksEvent.hh"
#include "BResource.hh"
#include "BOpticksKey.hh"
#include "BFile.hh"
#include "BTxt.hh"
#include "BHex.hh"
#include "BStr.hh"
#include "BPropNames.hh"
#include "BEnv.hh"
#include "PLOG.hh"
#include "Map.hh"


// npy-
#include "TorchStepNPY.hpp"
#include "GLMFormat.hpp"
#include "NState.hpp"
#include "NGLTF.hpp"
#include "NScene.hpp"
#include "NLoad.hpp"
#include "NSlice.hpp"
#include "NSceneConfig.hpp"
#include "NLODConfig.hpp"
#include "NSnapConfig.hpp"


// okc-
#include "OpticksSwitches.h"
#include "OpticksPhoton.h"
#include "OpticksFlags.hh"
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksColors.hh"
#include "OpticksGenstep.hh"
#include "OpticksEvent.hh"
#include "OpticksRun.hh"
#include "OpticksMode.hh"
#include "OpticksEntry.hh"
#include "OpticksProfile.hh"
#include "OpticksAna.hh"
#include "OpticksDbg.hh"

#include "OpticksCfg.hh"

const char*          Opticks::GEOCACHE_CODE_VERSION_KEY = "GEOCACHE_CODE_VERSION" ; 
const int            Opticks::GEOCACHE_CODE_VERSION = 8 ;  // (incremented when code changes invalidate loading old geocache dirs)   

/**
3: starting point 
4: switch off by default addition of extra global_instance GMergedMesh, 
   as GNodeLib now persists the "all volume" info enabling simplification of GMergedMesh 
5: go live with geometry model change mm0 no longer special, just remainder, GNodeLib name changes, start on triplet identity
6: GVolume::getIdentity quad now packing in more info including triplet_identity and sensor_index
7: GNodeLib add all_volume_inverse_transforms.npy
8: GGeo/GNodeLib/NMeta/CGDML/Opticks get G4GDMLAux info thru geocache for default genstep targetting configured 
   from within the GDML, opticksaux-dx1 modified with added auxiliary element for lvADE. Used for example by g4ok/G4OKTest   

**/


const plog::Severity Opticks::LEVEL = PLOG::EnvLevel("Opticks", "DEBUG")  ; 


BPropNames* Opticks::G_MATERIAL_NAMES = NULL ; 



const float Opticks::F_SPEED_OF_LIGHT = 299.792458f ;  // mm/ns

// formerly of GPropertyLib, now booted upstairs
float        Opticks::DOMAIN_LOW  = 60.f ;
float        Opticks::DOMAIN_HIGH = 820.f ;  // has been 810.f for a long time  
float        Opticks::DOMAIN_STEP = 20.f ; 
unsigned int Opticks::DOMAIN_LENGTH = 39  ;

float        Opticks::FINE_DOMAIN_STEP = 1.f ; 
unsigned int Opticks::FINE_DOMAIN_LENGTH = 761  ;






/*

In [12]: np.linspace(60,820,39)
Out[12]: 
array([  60.,   80.,  100.,  120.,  140.,  160.,  180.,  200.,  220.,
        240.,  260.,  280.,  300.,  320.,  340.,  360.,  380.,  400.,
        420.,  440.,  460.,  480.,  500.,  520.,  540.,  560.,  580.,
        600.,  620.,  640.,  660.,  680.,  700.,  720.,  740.,  760.,
        780.,  800.,  820.])

In [13]: np.linspace(60,820,39).shape
Out[13]: (39,)


In [10]: np.arange(60., 820.1, 20. )
Out[10]: 
array([  60.,   80.,  100.,  120.,  140.,  160.,  180.,  200.,  220.,
        240.,  260.,  280.,  300.,  320.,  340.,  360.,  380.,  400.,
        420.,  440.,  460.,  480.,  500.,  520.,  540.,  560.,  580.,
        600.,  620.,  640.,  660.,  680.,  700.,  720.,  740.,  760.,
        780.,  800.,  820.])

In [17]: np.arange(60., 820.1, 20. ).shape
Out[17]: (39,)

In [18]: np.arange(60., 820.1, 1. ).shape
Out[18]: (761,)



*/


glm::vec4 Opticks::getDefaultDomainSpec()
{
    glm::vec4 bd ;

    bd.x = DOMAIN_LOW ;
    bd.y = DOMAIN_HIGH ;
    bd.z = DOMAIN_STEP ;
    bd.w = DOMAIN_HIGH - DOMAIN_LOW ;

    return bd ; 
}

glm::vec4 Opticks::getDomainSpec(bool fine)
{
    glm::vec4 bd ;

    bd.x = DOMAIN_LOW ;
    bd.y = DOMAIN_HIGH ;
    bd.z = fine ? FINE_DOMAIN_STEP : DOMAIN_STEP ;
    bd.w = DOMAIN_HIGH - DOMAIN_LOW ;

    return bd ; 
}




glm::vec4 Opticks::getDefaultDomainReciprocalSpec()
{
    glm::vec4 rd ;
    rd.x = 1.f/DOMAIN_LOW ;
    rd.y = 1.f/DOMAIN_HIGH ;
    rd.z = 0.f ;
    rd.w = 0.f ;
    // not flipping order, only endpoints used for sampling, not the step 

    return rd ; 
}

glm::vec4 Opticks::getDomainReciprocalSpec(bool /*fine*/)
{
    glm::vec4 rd ;
    rd.x = 1.f/DOMAIN_LOW ;
    rd.y = 1.f/DOMAIN_HIGH ;
    rd.z = 0.f ;
    rd.w = 0.f ;
    // not flipping order, only endpoints used for sampling, not the step 

    return rd ; 
}




int Opticks::getRC() const
{
    return m_rc ; 
}
void Opticks::setRC(int rc, const char* rcmsg)
{
    m_rc = rc ; 
    m_rcmsg = rcmsg ? strdup(rcmsg) : NULL ; 
    dumpRC();
}

const char* Opticks::getRCMessage() const 
{
    return m_rcmsg ; 
}

int Opticks::rc() const 
{
    dumpRC();
    return m_rc ; 
}


void Opticks::dumpRC() const 
{
    LOG( m_rc == 0 ? info : fatal) 
           << " rc " << m_rc 
           << " rcmsg : " << ( m_rcmsg ? m_rcmsg : "-" ) 
           ;
}



Opticks*    Opticks::fInstance = NULL ; 


bool Opticks::IsLegacyGeometryEnabled() // static
{
    return BOpticksResource::IsLegacyGeometryEnabled() ;  // returns true when envvar OPTICKS_LEGACY_GEOMETRY_ENABLED is set to 1 
}
bool Opticks::IsForeignGeant4Enabled() // static
{
    return BOpticksResource::IsForeignGeant4Enabled() ;  // returns true when envvar OPTICKS_FOREIGN_GEANT4_ENABLED is set to 1 
}
bool Opticks::IsGeant4EnvironmentDetected() // static
{
    return BOpticksResource::IsGeant4EnvironmentDetected() ;  // returns true when find 10 G4...DATA envvars pointing at existing directories 
}





const char* Opticks::OptiXCachePathDefault()  // static
{
    return BOpticksResource::OptiXCachePathDefault() ; 
}



bool Opticks::HasInstance() 
{
    return fInstance != NULL ; 
}

bool Opticks::HasKey()
{
    assert( fInstance ) ; 
    return fInstance->hasKey() ; 
}

Opticks* Opticks::Instance()
{
    return fInstance ;  
}
Opticks* Opticks::GetInstance()
{
     if(fInstance == NULL )
     {
         const char* argforced = SSys::getenvvar("OPTICKS_INTERNAL_ARGS") ; 
         Opticks* ok = new Opticks(0,0, argforced);  
         ok->setInternal(true);   // internal means was instanciated within Opticks::GetInstance
     }
     assert( fInstance != NULL ) ; // Opticks ctor should have defined THE instance
     return fInstance ; 
}



/**
Opticks::envkey
----------------

Checks if a key is set already. 
If not attempts to set the key obtained from the OPTICKS_KEY envvar.

**/

bool Opticks::envkey()
{
    LOG(LEVEL); 

    bool key_is_set(false) ;
    key_is_set = BOpticksKey::IsSet() ; 
    if(key_is_set) return true ; 

    BOpticksKey::SetKey(NULL) ;  // use keyspec from OPTICKS_KEY envvar 

    key_is_set = BOpticksKey::IsSet() ; 
    assert( key_is_set == true && "valid geocache and key are required" ); 

    return key_is_set ; 
}



Opticks::Opticks(int argc, char** argv, const char* argforced )
    :
    m_log(new SLog("Opticks::Opticks","",debug)),
    m_ok(this),
    m_sargs(new SArgs(argc, argv, argforced)),  
    m_argc(m_sargs->argc),
    m_argv(m_sargs->argv),
    m_lastarg(m_argc > 1 ? strdup(m_argv[m_argc-1]) : NULL),
    m_mode(new OpticksMode(this)),
    m_dumpenv(m_sargs->hasArg("--dumpenv")),
    m_envkey(envkey()),
    m_production(m_sargs->hasArg("--production")),
    m_profile(new OpticksProfile()),
    m_materialprefix(NULL),
    m_photons_per_g4event(0), 

    m_spec(NULL),
    m_nspec(NULL),
    m_resource(NULL),
    m_origin_gdmlpath(NULL),
    m_origin_geocache_code_version(-1),
    m_state(NULL),
    m_apmtslice(NULL),
    m_apmtmedium(NULL),

    m_exit(false),
    m_compute(false),
    m_geocache(false),
    m_instanced(true),
    m_configured(false),

    m_cfg(new OpticksCfg<Opticks>("opticks", this,false)),
    m_parameters(new NMeta), 
    m_runtxt(new BTxt),  
    m_cachemeta(new NMeta), 
    m_origin_cachemeta(NULL), 

    m_scene_config(NULL),
    m_lod_config(NULL),
    m_snap_config(NULL),
    m_detector(NULL),
    m_event_count(0),
    m_domains_configured(false),
    m_time_domain(0.f, 0.f, 0.f, 0.f),
    m_space_domain(0.f, 0.f, 0.f, 0.f),
    m_wavelength_domain(0.f, 0.f, 0.f, 0.f),
    m_settings(0,0,0,0),
    m_run(new OpticksRun(this)),
    //m_evt(NULL),
    m_ana(new OpticksAna(this)),
    m_dbg(new OpticksDbg(this)),
    m_rc(0),
    m_rcmsg(NULL),
    m_tagoffset(0),
    m_verbosity(0),
    m_internal(false),
    m_frame_renderer(NULL)
{

    m_profile->setStamp(m_sargs->hasArg("--stamp"));

    OK_PROFILE("Opticks::Opticks");

    if(fInstance != NULL)
    {
        LOG(fatal) << " SECOND OPTICKS INSTANCE " ;  
    }

    fInstance = this ; 

    init();
    (*m_log)("DONE");
}



/**
Opticks::init
---------------

OKTest without options defaults to writing the below::

    js.py $TMP/default_pfx/evt/dayabay/torch/0/parameters.json

**/

void Opticks::init()
{
    LOG(LEVEL) << "[" ; 
    LOG(info) << m_mode->description() << " hostname " << SSys::hostname() ; 
    if(IsLegacyGeometryEnabled())
    {
        LOG(fatal) << "OPTICKS_LEGACY_GEOMETRY_ENABLED mode is active " 
                   << " : ie dae src access to geometry, opticksdata  "  
                    ;
    }
    else
    {
        LOG(info) << " mandatory keyed access to geometry, opticksaux " ; 
    } 

    m_parameters->add<int>("OptiXVersion",  OKConf::OptiXVersionInteger() );
    m_parameters->add<int>("CUDAVersion",   OKConf::CUDAVersionInteger() );
    m_parameters->add<int>("ComputeVersion", OKConf::ComputeCapabilityInteger() );
    m_parameters->add<int>("Geant4Version",  OKConf::Geant4VersionInteger() );

    m_parameters->addEnvvar("CUDA_VISIBLE_DEVICES");
    m_parameters->add<std::string>("CMDLINE", PLOG::instance ? PLOG::instance->cmdline() : "OpticksEmbedded" ); 
    if(m_envkey) m_parameters->add<int>("--envkey", 1 ); // OPTICKS_KEY envvar is only relevant when use --envkey to switch on sensitivity 
    m_parameters->addEnvvarsWithPrefix("OPTICKS_"); 

    m_parameters->add<std::string>("HOSTNAME", SSys::hostname() ); 
    m_parameters->add<std::string>("USERNAME", SSys::username() ); 

    m_parameters->add<std::string>("OpticksSwitches", OpticksSwitches() ); 

    //std::raise(SIGINT); 

    LOG(LEVEL) << "]" ; 
}



bool Opticks::isDumpEnv() const 
{
    return m_dumpenv ; 
}



bool Opticks::isInternal() const 
{
    return m_internal ; 
}
void Opticks::setInternal(bool internal)  
{
    m_internal = internal ; 
}

void  Opticks::SetFrameRenderer(const char* renderer)  // static 
{ 
    assert(fInstance) ; 
    fInstance->setFrameRenderer(renderer) ;  
}
void Opticks::setFrameRenderer(const char* renderer)  
{
    m_frame_renderer = strdup(renderer) ; 
}
const char* Opticks::getFrameRenderer() const
{
    return m_frame_renderer ; 
}





std::string Opticks::getArgLine()
{
    return m_sargs->getArgLine();
}


/*
template <typename T>
void Opticks::profile(T label)
{
    m_profile->stamp<T>(label, m_tagoffset);
}
*/


void Opticks::profile(const char* label)
{
    m_profile->stamp(label, m_tagoffset);
   // m_tagoffset is set by Opticks::makeEvent
}


unsigned Opticks::accumulateAdd(const char* label)
{
    return m_profile->accumulateAdd(label);
}
void Opticks::accumulateStart(unsigned idx)
{
    m_profile->accumulateStart(idx);  
}
void Opticks::accumulateStop(unsigned idx)
{
    m_profile->accumulateStop(idx);  
}
std::string Opticks::accumulateDesc(unsigned idx)
{
    return m_profile->accumulateDesc(idx);
}


void Opticks::accumulateSet(unsigned idx, float dt)
{
    m_profile->accumulateSet(idx, dt);  
}

unsigned Opticks::lisAdd(const char* label)
{
    return m_profile->lisAdd(label); 
}
void Opticks::lisAppend(unsigned idx, double t)
{
    m_profile->lisAppend(idx, t ); 
}



void Opticks::dumpProfile(const char* msg, const char* startswith, const char* spacewith, double tcut)
{
   m_profile->dump(msg, startswith, spacewith, tcut);
}


const char* Opticks::getProfileDir() const
{
    return m_profile->getDir(); 
}

/**
Opticks::setProfileDir
------------------------

Canonically invoked by Opticks::postgeometry

**/

void Opticks::setProfileDir(const char* dir)
{
    LOG(LEVEL) << " dir " << dir ; 
    bool is_cvmfs = 
           dir[0] == '/' && 
           dir[1] == 'c' && 
           dir[2] == 'v' &&
           dir[3] == 'm' &&
           dir[4] == 'f' &&
           dir[5] == 's' &&
           dir[6] == '/' ;

    assert( !is_cvmfs );   
    m_profile->setDir(dir);
}
void Opticks::saveProfile()
{
   m_profile->save();
}

void Opticks::postgeocache()
{
   dumpProfile("Opticks::postgeocache", NULL  );  
}

void Opticks::postpropagate()
{
   saveProfile();

   //double tcut = 0.0001 ; 
   /*
   double tcut = -1.0 ;  
   dumpProfile("Opticks::postpropagate", NULL, "_OpticksRun::createEvent", tcut  );  // spacwith spacing at start if each evt
   */


   if(isDumpProfile()) 
   {
       LOG(info) << "[ --dumpprofile " ;  
       // startswith filtering 
       dumpProfile("Opticks::postpropagate", "OPropagator::launch");  
       dumpProfile("Opticks::postpropagate", "CG4::propagate");  

       dumpParameters("Opticks::postpropagate");
       LOG(info) << "] --dumpprofile " ;  
   }

   saveParameters(); 


}

void Opticks::ana()
{
   m_ana->run();
}
OpticksAna* Opticks::getAna() const 
{
    return m_ana  ; 
}


NPY<unsigned>* Opticks::getMaskBuffer() const 
{
    return m_dbg->getMaskBuffer() ;  
}
const std::vector<unsigned>&  Opticks::getMask() const 
{
    return m_dbg->getMask();
}
unsigned Opticks::getMaskIndex(unsigned idx) const
{
    bool mask = hasMask();  
    if(!mask)
        LOG(warning) << "BUT there is no mask " ; 

    return mask ? m_dbg->getMaskIndex(idx) : idx ;
}
bool Opticks::hasMask() const 
{
    return m_dbg->getMask().size() > 0 ; 
}
unsigned Opticks::getMaskSize() const 
{
    return m_dbg->getMask().size() ; 
}

unsigned Opticks::getDbgHitMask() const 
{
    const std::string& _dbghitmask = m_cfg->getDbgHitMask(); 
    unsigned dbghitmask = OpticksFlags::AbbrevSequenceToMask( _dbghitmask.c_str(), ',' ); 
    return dbghitmask ; 
}


bool Opticks::isDbgPhoton(unsigned record_id) const 
{
   return m_dbg->isDbgPhoton(record_id);
}
bool Opticks::isOtherPhoton(unsigned photon_id) const 
{
   return m_dbg->isOtherPhoton(photon_id);
}
bool Opticks::isMaskPhoton(unsigned photon_id) const 
{
   return m_dbg->isMaskPhoton(photon_id);
}
bool Opticks::isX4PolySkip(unsigned lvIdx) const 
{
   return m_dbg->isX4PolySkip(lvIdx);
}
bool Opticks::isCSGSkipLV(unsigned lvIdx) const 
{
   return m_dbg->isCSGSkipLV(lvIdx);
}
unsigned Opticks::getNumCSGSkipLV() const 
{
   return m_dbg->getNumCSGSkipLV() ; 
}


bool Opticks::isEnabledMergedMesh(unsigned mm) const 
{
   return m_dbg->isEnabledMergedMesh(mm);
}

unsigned Opticks::getInstanceModulo(unsigned mm) const 
{
   return m_dbg->getInstanceModulo(mm);
}





bool Opticks::isDbgPhoton(int event_id, int track_id)
{
    unsigned record_id = event_id*m_photons_per_g4event + track_id ; 
    return m_dbg->isDbgPhoton(record_id);
}
bool Opticks::isOtherPhoton(int event_id, int track_id)
{
    unsigned record_id = event_id*m_photons_per_g4event + track_id ; 
    return m_dbg->isOtherPhoton(record_id);
}
bool Opticks::isGenPhoton(int gen_id)
{
    return m_dbg->isGenPhoton(gen_id);
}


bool Opticks::isMaskPhoton(int event_id, int track_id)
{
    unsigned record_id = event_id*m_photons_per_g4event + track_id ; 
    return m_dbg->isMaskPhoton(record_id);
}





unsigned Opticks::getNumDbgPhoton() const 
{
    return m_dbg->getNumDbgPhoton();
}
unsigned Opticks::getNumOtherPhoton() const 
{
    return m_dbg->getNumOtherPhoton();
}
unsigned Opticks::getNumGenPhoton() const 
{
    return m_dbg->getNumGenPhoton();
}



unsigned Opticks::getNumMaskPhoton() const 
{
    return m_dbg->getNumMaskPhoton();
}

const std::vector<unsigned>&  Opticks::getDbgIndex()
{
    return m_dbg->getDbgIndex();
}
const std::vector<unsigned>&  Opticks::getOtherIndex()
{
    return m_dbg->getOtherIndex();
}
const std::vector<unsigned>&  Opticks::getGenIndex()
{
    return m_dbg->getGenIndex();
}







int Opticks::getDebugIdx() const 
{
    return m_cfg->getDebugIdx();
}
int Opticks::getDbgNode() const 
{
    return m_cfg->getDbgNode();
}
int Opticks::getDbgMM() const 
{
    return m_cfg->getDbgMM();
}
int Opticks::getDbgLV() const 
{
    return m_cfg->getDbgLV();
}




int Opticks::getStack() const 
{
   return m_cfg->getStack();
}
int Opticks::getMaxCallableProgramDepth() const 
{
   return m_cfg->getMaxCallableProgramDepth();
}
int Opticks::getMaxTraceDepth() const 
{
   return m_cfg->getMaxTraceDepth();
}
int Opticks::getUsageReportLevel() const 
{
   return m_cfg->getUsageReportLevel();
}





int Opticks::getMeshVerbosity() const 
{
   return m_cfg->getMeshVerbosity();
}


const char* Opticks::getAccel() const 
{
   const std::string& accel = m_cfg->getAccel();
   return accel.empty() ? NULL : accel.c_str() ; 
}

const char* Opticks::getFlightPathDir() const 
{
   const std::string& dir = m_cfg->getFlightPathDir();
   return dir.empty() ? NULL : dir.c_str() ;
}


const char* Opticks::getDbgMesh() const 
{
   const std::string& dbgmesh = m_cfg->getDbgMesh();
   return dbgmesh.empty() ? NULL : dbgmesh.c_str() ;
}


/**
Opticks::initResource
-----------------------

Invoked by Opticks::configure.

Instanciates m_resource OpticksResource and its base BOpticksResource
which defines the geocache paths. 

Previously the decision to use legacy and direct geometry workflow 
for python scripts invoked from OpticksAna (and separately) 
was controlled by the setting or not of the IDPATH envvar.  
Now that legacy mode is no longer supported, there is no more need 
for this split.

See notes/issues/test-fails-from-geometry-workflow-interference.rst

**/

void Opticks::initResource()
{
    LOG(LEVEL) << "[ OpticksResource " ;
    m_resource = new OpticksResource(this);
    const char* detector = m_resource->getDetector() ; 
    const char* idpath = m_resource->getIdPath();
    LOG(LEVEL) << "] OpticksResource " << detector ;

    setDetector(detector);
    m_parameters->add<std::string>("idpath", idpath); 

    LOG(LEVEL) << m_resource->desc()  ;
}


int Opticks::getArgc()
{
    return m_argc ; 
}
char** Opticks::getArgv()
{
    return m_argv ; 
}
char* Opticks::getArgv0()
{
    return m_argc > 0 && m_argv ? m_argv[0] : NULL ; 
}


bool Opticks::hasArg(const char* arg)
{
    bool has = false ; 
    for(int i=1 ; i < m_argc ; i++ ) if(strcmp(m_argv[i], arg) == 0) has = true ; 
    return has ; 
}

void Opticks::setCfg(OpticksCfg<Opticks>* cfg)
{
    m_cfg = cfg ; 
}
OpticksCfg<Opticks>* Opticks::getCfg() const 
{
    return m_cfg ; 
}

const char* Opticks::getRenderMode() const 
{
    const std::string& s = m_cfg->getRenderMode();
    return s.empty() ? NULL : s.c_str();
}
const char* Opticks::getRenderCmd() const 
{
    const std::string& s = m_cfg->getRenderCmd();
    return s.empty() ? NULL : s.c_str();
}

const char* Opticks::getCSGSkipLV() const 
{
    const std::string& s = m_cfg->getCSGSkipLV();
    return s.empty() ? NULL : s.c_str();
}






const char* Opticks::getLVSDName() const 
{
    const std::string& s = m_cfg->getLVSDName();
    return s.empty() ? NULL : s.c_str();
}

const char* Opticks::getCathode() const 
{
    const std::string& s = m_cfg->getCathode();
    return s.c_str();
}

const char* Opticks::getCerenkovClass() const 
{
    const std::string& s = m_cfg->getCerenkovClass();
    return s.c_str();
}

const char* Opticks::getScintillationClass() const 
{
    const std::string& s = m_cfg->getScintillationClass();
    return s.c_str();
}



bool Opticks::isGlobalInstanceEnabled() const // --global_instance_enabled
{
    return m_cfg->hasOpt("global_instance_enabled") ; 
}
bool Opticks::isG4CodeGen() const  // --g4codegen
{
    return m_cfg->hasOpt("g4codegen") ;
}
bool Opticks::isNoSavePPM() const  // --nosaveppm
{
    return m_cfg->hasOpt("nosaveppm") ;
}
bool Opticks::isNoG4Propagate() const  // --nog4propagate
{
    return m_cfg->hasOpt("nog4propagate") ;
}



bool Opticks::canDeleteGeoCache() const   // --deletegeocache
{
    return m_cfg->hasOpt("deletegeocache") ;
}
void Opticks::deleteGeoCache() const 
{
    assert( canDeleteGeoCache() ); 
    const char* idpath = getIdPath(); 
    assert( idpath ); 
    LOG(info) << "removing " << idpath << " (as permitted by option : --deletegeocache )" ; 
    BFile::RemoveDir(idpath); 
}

void Opticks::enforceNoGeoCache() const
{
    const Opticks* ok = this ; 
    // used by OKX4Test as that is explicitly intended to write geocaches  
    if(ok->hasGeocache()) 
    {   
        LOG(fatal) << "geocache exists already " << ok->getIdPath() ;
        if(!ok->canDeleteGeoCache())
        {
            LOG(fatal) << "delete this externally OR rerun with --deletegeocache option " ;   
        }
        else
        {
            ok->deleteGeoCache(); 
        }
    }   
    assert(!ok->hasGeocache()); 
}


void Opticks::reportKey(const char* msg) const
{
    const Opticks* ok = this ; 
    LOG(info) << msg ; 

    const char* kspec = ok->getKeySpec() ; 
    const char* espec = SSys::getenvvar("OPTICKS_KEY", "NONE" ); 

    std::cout 
        << std::endl << GEOCACHE_CODE_VERSION_KEY << " " << GEOCACHE_CODE_VERSION
        << std::endl << "KEYDIR  " << ok->getIdPath() 
        << std::endl << "KEY     " << kspec  
        << std::endl << " "    
        << std::endl << "To reuse this geometry include below export in ~/.opticks_config::" 
        << std::endl << " "   
        << std::endl << "    export OPTICKS_KEY=" << kspec 
        << std::endl  
        << std::endl  
        ;   

    if(strcmp(kspec, espec) == 0) 
    {
        LOG(info) << "This key matches that of the current envvar " ; 
    }
    else
    {
        LOG(fatal) << "THE LIVE key DOES NOT MATCH THAT OF THE CURRENT ENVVAR " ; 
        LOG(info) << " (envvar) OPTICKS_KEY=" <<  espec ; 
        LOG(info) << " (live)   OPTICKS_KEY=" <<  kspec ; 
    }
}







bool Opticks::isPrintEnabled() const   // --printenabled
{
    return m_cfg->hasOpt("printenabled") ;
}
bool Opticks::isExceptionEnabled() const  // --exceptionenabled
{
    return m_cfg->hasOpt("exceptionenabled") ;
}

bool Opticks::isPrintIndexLog() const   // --pindexlog
{
    return m_cfg->hasOpt("pindexlog") ;
}

/**
Opticks::isXAnalytic
-----------------------

Attempt to switch this on by default causing 11 test fails, so back off for now.
Need to sort out the testing geometry in opticksdata before can make this move.

See notes/issues/switching-to-xanalytic-as-default-causes-11-test-fails-so-revert.rst

In the mean time, use --xtriangle as a way to switch off --xanalytic in the same commandline

**/

bool Opticks::isXAnalytic() const 
{
    bool is_xanalytic = m_cfg->hasOpt("xanalytic") ; 
    bool is_xtriangle = m_cfg->hasOpt("xtriangle") ; 

    bool ret = is_xanalytic ; 

    if(is_xanalytic && is_xtriangle)
    {
        LOG(error) << " --xanalytic option overridded by --xtriangle  " ;   
        ret = false ;  
    } 
    return ret ; 
}





bool Opticks::isXGeometryTriangles() const 
{
    return m_cfg->hasOpt("xgeometrytriangles") ;
}




int Opticks::getPrintIndex(unsigned dim) const 
{
    glm::ivec3 idx ; 
    int pindex = -1 ; 
    if(getPrintIndex(idx)) 
    {
        switch(dim)
        {
            case 0: pindex = idx.x ; break ; 
            case 1: pindex = idx.y ; break ; 
            case 2: pindex = idx.z ; break ; 
        }
    }
    return pindex ; 
}

bool Opticks::getPrintIndex(glm::ivec3& idx) const 
{
    const char* pindex = getPrintIndexString();
    if(!pindex) return false ; 
    idx = givec3(pindex);
    return true ; 
}


bool Opticks::getAnimTimeRange(glm::vec4& range) const
{
    const std::string& animtimerange = m_cfg->getAnimTimeRange();
    range = gvec4(animtimerange.c_str());
    return true  ; 
}



const char* Opticks::getPrintIndexString() const 
{
    const std::string& printIndex = m_cfg->getPrintIndex();
    return printIndex.empty() ? NULL : printIndex.c_str();
}
const char* Opticks::getDbgIndex() const 
{
    const std::string& dbgIndex = m_cfg->getDbgIndex();
    return dbgIndex.empty() ? NULL : dbgIndex.c_str();
}






const char* Opticks::getDbgCSGPath()
{
    const std::string& dbgcsgpath = m_cfg->getDbgCSGPath();
    return dbgcsgpath.empty() ? NULL : dbgcsgpath.c_str();
}

unsigned Opticks::getSeed() const 
{
    return m_cfg->getSeed();
}

int Opticks::getRTX() const 
{
    return m_cfg->getRTX();
}
int Opticks::getRenderLoopLimit() const 
{
    return m_cfg->getRenderLoopLimit();
}



int Opticks::getLoadVerbosity() const 
{
    return m_cfg->getLoadVerbosity();
}
int Opticks::getImportVerbosity() const 
{
    return m_cfg->getImportVerbosity();
}





OpticksRun* Opticks::getRun()
{
    return m_run ;  
}
OpticksEvent* Opticks::getEvent() const 
{
    return m_run->getEvent()  ; 
}



OpticksProfile* Opticks::getProfile() const 
{
    return m_profile ; 
}


NMeta*       Opticks::getParameters() const 
{
    return m_parameters ; 
}

void Opticks::dumpParameters(const char* msg) const 
{
    m_parameters->dump(msg);
}


/**
Opticks::saveParameters
--------------------------

Metadata parameters.json are saved either into 

1. TagZeroDir following event propagation running 
2. RunResultsDir following non-event running, eg raytrace benchmarks 

**/

void Opticks::saveParameters() const 
{
    const char* name = "parameters.json" ; 
    OpticksEvent* evt = getEvent(); 

    if( evt )
    {
        std::string dir = evt->getTagZeroDir() ; 
        LOG(LEVEL) << " postpropagate save " << name << " into TagZeroDir " << dir ; 
        m_parameters->save( dir.c_str(), name ); 
    }
    else
    {
        const char* dir = getRunResultsDir() ; 
        LOG(LEVEL) << " non-event running (eg raytrace benchmarks) save " << name << " into RunResultsDir " << dir ; 
        m_parameters->save( dir, name ) ; 
    }
}





OpticksResource* Opticks::getResource() const 
{
    return m_resource  ; 
}
void Opticks::dumpResource() const 
{
    return m_resource->Dump()  ; 
}
bool Opticks::isKeySource() const // name of current executable matches that of the creator of the geocache
{
    return m_resource->isKeySource();  
}



NState* Opticks::getState() const 
{
    return m_state  ; 
}

const char* Opticks::getLastArg()
{
   return m_lastarg ; 
}





void Opticks::setModeOverride(unsigned int mode)
{
    m_mode->setOverride(mode) ; 
}
bool Opticks::isRemoteSession()
{
    return SSys::IsRemoteSession();
}
bool Opticks::isCompute()
{
    return m_mode->isCompute() ;
}
bool Opticks::isInterop()
{
    return m_mode->isInterop() ;
}
bool Opticks::isCfG4()
{
    assert(0); 
    return m_mode->isCfG4(); 
}




bool Opticks::isAlign() const  // --align
{
   return m_cfg->hasOpt("align");
}
bool Opticks::isDbgNoJumpZero() const  // --dbgnojumpzero
{
   return m_cfg->hasOpt("dbgnojumpzero");
}
bool Opticks::isDbgFlat() const  // --dbgflat
{
   return m_cfg->hasOpt("dbgflat");
}

bool Opticks::isDbgSkipClearZero() const  // --dbgskipclearzero
{
   return m_cfg->hasOpt("dbgskipclearzero");
}
bool Opticks::isDbgKludgeFlatZero() const  // --dbgkludgeflatzero
{
   return m_cfg->hasOpt("dbgkludgeflatzero");
}
bool Opticks::isDbgTex() const  // --dbgtex
{
   return m_cfg->hasOpt("dbgtex");
}

bool Opticks::isDbgEmit() const  // --dbgemit
{
   return m_cfg->hasOpt("dbgemit");
}

bool Opticks::isDbgDownload() const  // --dbgdownload
{
   return m_cfg->hasOpt("dbgdownload");
}
bool Opticks::isDbgHit() const  // --dbghit
{
   return m_cfg->hasOpt("dbghit");
}
bool Opticks::isDumpHit() const  // --dumphit
{
   return m_cfg->hasOpt("dumphit");
}
bool Opticks::isDumpSensor() const  // --dumpsensor
{
   return m_cfg->hasOpt("dumpsensor");
}
bool Opticks::isSaveSensor() const  // --savesensor
{
   return m_cfg->hasOpt("savesensor");
}
bool Opticks::isDumpProfile() const  // --dumpprofile
{
   return m_cfg->hasOpt("dumpprofile");
}


bool Opticks::isDbgGeoTest() const  // --dbggeotest
{
   return m_cfg->hasOpt("dbggeotest");
}








bool Opticks::isReflectCheat() const  // --reflectcheat
{
   return m_cfg->hasOpt("reflectcheat");
}
bool Opticks::isSave() const   // --save is trumped by --nosave 
{
    bool is_nosave = m_cfg->hasOpt("nosave");  
    bool is_save = m_cfg->hasOpt("save");  
    return is_nosave ? false : is_save  ;   
}
bool Opticks::isLoad() const
{
   // --noload trumps --load
    return m_cfg->hasOpt("load") && !m_cfg->hasOpt("noload"); 
}
bool Opticks::isTracer() const
{
    return m_cfg->hasOpt("tracer") ;
}

bool Opticks::isRayLOD() const
{
    return m_cfg->hasOpt("raylod") ;
}

bool Opticks::isMaterialDbg() const
{
    return m_cfg->hasOpt("materialdbg") ;
}

bool Opticks::isDbgAnalytic() const
{
    return m_cfg->hasOpt("dbganalytic") ;
}

bool Opticks::isDbgSurf() const
{
    return m_cfg->hasOpt("dbgsurf") ;
}
bool Opticks::isDbgBnd() const
{
    return m_cfg->hasOpt("dbgbnd") ;
}

bool Opticks::isDbgRec() const
{
    return m_cfg->hasOpt("dbgrec") ;
}
bool Opticks::isDbgZero() const
{
    return m_cfg->hasOpt("dbgzero") ;
}
bool Opticks::isRecPoi() const  // --recpoi
{
    return m_cfg->hasOpt("recpoi") ;
}
bool Opticks::isRecPoiAlign() const  // --recpoialign
{
    return m_cfg->hasOpt("recpoialign") ;
}


bool Opticks::isRecCf() const     // --reccf
{
    return m_cfg->hasOpt("reccf") ;
}





bool Opticks::isDbgTorch() const
{
    return m_cfg->hasOpt("torchdbg") ;
}
bool Opticks::isDbgSource() const
{
    return m_cfg->hasOpt("sourcedbg") ;
}
bool Opticks::isDbgAim() const  // --dbgaim
{
    return m_cfg->hasOpt("dbgaim") ;
}
bool Opticks::isDbgClose() const
{
    return m_cfg->hasOpt("dbgclose") ;
}







std::string Opticks::brief()
{
    std::stringstream ss ; 
    ss << "OK" ;
    ss << ( isCompute() ? " COMPUTE" : " INTEROP" ) ;
    ss << ( isProduction() ? " PRODUCTION" : " DEVELOPMENT" ) ;
    ss << ( isUTailDebug() ? " UTAILDEBUG " : "" ) ;
    return ss.str();
}



bool Opticks::isProduction() const   // --production
{
   return m_production ; 
}
bool Opticks::isUTailDebug() const   // --utaildebug
{
   return m_cfg->hasOpt("utaildebug") ; 
}





void Opticks::setIntegrated(bool integrated)
{
   m_integrated = integrated ;
}
bool Opticks::isIntegrated()
{
   return m_integrated ; 
}





const glm::vec4& Opticks::getTimeDomain() const 
{
    return m_time_domain ; 
}
const glm::vec4& Opticks::getSpaceDomain() const 
{
    return m_space_domain ; 
}
const glm::vec4& Opticks::getWavelengthDomain() const
{
    return m_wavelength_domain ; 
}
const glm::ivec4& Opticks::getSettings() const 
{
    return m_settings ; 
}


const glm::uvec4& Opticks::getSize()
{
    return m_size ; 
}
const glm::uvec4& Opticks::getPosition()
{
    return m_position ; 
}




void Opticks::setDetector(const char* detector)
{
    m_detector = detector ? strdup(detector) : NULL ; 
}


void Opticks::configureS(const char* , std::vector<std::string> )
{
}

void Opticks::configureI(const char* , std::vector<int> )
{
}

bool Opticks::isExit()
{
    return m_exit ; 
}
void Opticks::setExit(bool exit_)
{
    m_exit = exit_  ;   
    if(m_exit)
    {
        LOG(info) << "EXITING " ; 
        exit(EXIT_SUCCESS) ;
    }
}


unsigned long long Opticks::getDbgSeqmat()
{
    const std::string& seqmat = m_cfg->getDbgSeqmat();
    return BHex<unsigned long long>::hex_lexical_cast( seqmat.c_str() );
}
unsigned long long Opticks::getDbgSeqhis()  // --dbgseqhis
{
    const std::string& seqhis = m_cfg->getDbgSeqhis();
    return BHex<unsigned long long>::hex_lexical_cast( seqhis.c_str() );
}

const std::string& Opticks::getSeqMapString() const 
{
    return m_cfg->getSeqMap() ;
}

void Opticks::setSeqMapString(const char* seqmap)
{
    m_cfg->setSeqMap(seqmap);
}


bool Opticks::getSeqMap(unsigned long long& seqhis, unsigned long long& seqval)
{
    const std::string& seqmap = m_cfg->getSeqMap();
    if(seqmap.empty()) return false ; 
    char edelim = BStr::HasChar(seqmap, ',') ? ',' : ' ' ; 
    OpticksFlags::AbbrevToFlagValSequence(seqhis, seqval, seqmap.c_str(), edelim );
    return true ; 
}



float Opticks::getFxRe()
{
    std::string fxre = m_cfg->getFxReConfig();
    return BStr::atof(fxre.c_str(), 0);
}
float Opticks::getFxAb()
{
    std::string fxab = m_cfg->getFxAbConfig();
    return BStr::atof(fxab.c_str(), 0);
}
float Opticks::getFxSc()
{
    std::string fxsc = m_cfg->getFxScConfig();
    return BStr::atof(fxsc.c_str(), 0);
}


bool Opticks::isAnalyticPMTLoad()
{
    return m_cfg->hasOpt("apmtload");
}




unsigned Opticks::getAnalyticPMTIndex()
{
    return m_cfg->getAnalyticPMTIndex();
}

const char* Opticks::getAnalyticPMTMedium()
{
    if(m_apmtmedium == NULL)
    {
        std::string cmed = m_cfg->getAnalyticPMTMedium() ;
        std::string dmed = m_resource->getDefaultMedium()  ; 
        LOG(verbose) 
            << " cmed " << cmed 
            << " cmed.empty " << cmed.empty()
            << " dmed " << dmed 
            << " dmed.empty " << dmed.empty()
            ;

        m_apmtmedium = !cmed.empty() ? strdup(cmed.c_str()) : strdup(dmed.c_str()) ;
    }
    return m_apmtmedium ;
}

int Opticks::getDefaultFrame() const 
{
    return m_resource->getDefaultFrame() ; 
}

const char* Opticks::getRunResultsDir() const 
{
    return m_resource ? m_resource->getRunResultsDir() : NULL ; 
}
const char* Opticks::getRuncacheDir() const 
{
    return m_resource ? m_resource->getRuncacheDir() : NULL ; 
}
const char* Opticks::getOptiXCacheDirDefault() const 
{
    return m_resource ? m_resource->getOptiXCacheDirDefault() : NULL ; 
}







NSlice* Opticks::getAnalyticPMTSlice()
{
    if(m_apmtslice == 0)
    {
        std::string sli = m_cfg->getAnalyticPMTSlice() ; 
        if(!sli.empty()) m_apmtslice = new NSlice(sli.c_str());
    }
    return m_apmtslice ; 
}


const char* Opticks::getSensorSurface()
{
    return m_resource->getSensorSurface() ;
}







int  Opticks::getGLTF() const 
{
    return m_cfg->getGLTF(); 
}
int  Opticks::getGLTFTarget() const 
{
    return m_cfg->getGLTFTarget(); 
}

bool Opticks::isGLTF() const 
{
    return getGLTF() > 0 ; 
}

const char* Opticks::getGLTFPath() const { return m_resource->getGLTFPath() ; }
const char* Opticks::getG4CodeGenDir() const { return m_resource->getG4CodeGenDir() ; }
const char* Opticks::getCacheMetaPath() const { return m_resource->getCacheMetaPath() ; } 
const char* Opticks::getGDMLAuxMetaPath() const { return m_resource->getGDMLAuxMetaPath() ; } 
const char* Opticks::getRunCommentPath() const { return m_resource->getRunCommentPath() ; } 




/*
const char* Opticks::getSrcGLTFPath() const { return m_resource->getSrcGLTFPath() ; }

const char* Opticks::getSrcGLTFBase() const  // config base and name only used whilst testing with gltf >= 100
{
    int gltf = getGLTF();
    const char* path = getSrcGLTFPath() ;
    if(!path) return NULL ; 
    std::string base = gltf < 100 ? BFile::ParentDir(path) : m_cfg->getSrcGLTFBase() ;
    return strdup(base.c_str()) ; 
}

const char* Opticks::getSrcGLTFName() const 
{
    int gltf = getGLTF();
    const char* path = getSrcGLTFPath() ;
    if(!path) return NULL ; 
    std::string name = gltf < 100 ? BFile::Name(path) : m_cfg->getSrcGLTFName()  ;
    return strdup(name.c_str()) ; 
}

bool Opticks::hasSrcGLTF() const 
{
    // lookahead to what GScene::GScene will do
    return NGLTF::Exists(getSrcGLTFBase(), getSrcGLTFName()) ;
}


void Opticks::configureCheckGeometryFiles() 
{
    if(isGLTF() && !hasSrcGLTF())
    {
        LOG(fatal) << "gltf option is selected but there is no gltf file " ;
        LOG(fatal) << " SrcGLTFBase " << getSrcGLTFBase() ;
        LOG(fatal) << " SrcGLTFName " << getSrcGLTFName() ;
        LOG(fatal) << "Try to create the GLTF from GDML with eg:  op --j1707 --gdml2gltf  "  ;
        
        //setExit(true); 
        //assert(0);
    }
} 
*/



const char* Opticks::getGLTFConfig()
{
    return m_cfg->getGLTFConfig().c_str() ; 
}

/**
Opticks::getSceneConfig
-----------------------

Still needed by GInstancer.

**/

NSceneConfig* Opticks::getSceneConfig()
{
    if(m_scene_config == NULL)
    {
        m_scene_config = new NSceneConfig(getGLTFConfig());
    }
    return m_scene_config ; 
}




int  Opticks::getLayout() const 
{
    return m_cfg->getLayout(); 
}




const char* Opticks::getGPUMonPath() const 
{
    const std::string& gpumonpath = m_cfg->getGPUMonPath() ;
    return gpumonpath.c_str() ;
}



bool Opticks::isGPUMon() const 
{
    return m_cfg->hasOpt("gpumon");
}


const char* Opticks::getRunComment() const 
{
    const std::string& runcomment = m_cfg->getRunComment() ;  
    return runcomment.empty() ? NULL : runcomment.c_str() ; 
}

int Opticks::getRunStamp() const 
{
    return m_cfg->getRunStamp() ;
}
const char* Opticks::getRunDate() const 
{
    int t = getRunStamp();
    std::string s = STime::Format(t); 
    return strdup(s.c_str());
}

void Opticks::appendCacheMeta(const char* key, NMeta* obj)
{
    m_cachemeta->setObj(key, obj); 
}



/**
Opticks::updateCacheMeta
---------------------------

Invoked by Opticks::configure after initResource



**/

void Opticks::updateCacheMeta()  
{
    std::string argline = PLOG::instance ? PLOG::instance->args.argline() : "OpticksEmbedded" ;

    int runstamp = getRunStamp() ;
    const char* rundate = getRunDate() ;
    const char* runcomment = getRunComment() ; 
    const char* runlabel = getRunLabel() ; 
    const char* runfolder = getRunFolder() ; 

    m_runtxt->addLine(GEOCACHE_CODE_VERSION_KEY);
    m_runtxt->addLine(GEOCACHE_CODE_VERSION); 
    m_runtxt->addLine("rundate") ;  
    m_runtxt->addLine( rundate ) ;  
    m_runtxt->addLine("runstamp" ) ;  
    m_runtxt->addValue( runstamp);  
    m_runtxt->addLine("argline" ) ;  
    m_runtxt->addLine( argline) ;  

    m_cachemeta->set<int>(GEOCACHE_CODE_VERSION_KEY, GEOCACHE_CODE_VERSION ); 
    m_cachemeta->set<std::string>("location", "Opticks::updateCacheMeta"); 
    m_cachemeta->set<std::string>("argline",  argline ); 
    m_cachemeta->set<std::string>("rundate", rundate ); 
    m_cachemeta->set<int>("runstamp", runstamp ); 

    if(runcomment)
    {
        m_runtxt->addLine("runcomment" ) ;  
        m_runtxt->addLine( runcomment) ;  
        m_cachemeta->set<std::string>("runcomment", runcomment ); 
    }

    if(runlabel)
    {
        m_runtxt->addLine("runlabel" ) ;  
        m_runtxt->addLine( runlabel) ;  
        m_cachemeta->set<std::string>("runlabel", runlabel ); 
    }

    if(runfolder)
    {
        m_runtxt->addLine("runfolder" ) ;  
        m_runtxt->addLine( runfolder) ;  
        m_cachemeta->set<std::string>("runfolder", runfolder ); 
    }

}


void Opticks::dumpCacheMeta(const char* msg) const 
{
    m_cachemeta->dump(msg) ;
}
void Opticks::saveCacheMeta() const 
{
    const char* path = getRunCommentPath(); 
    assert( m_runtxt) ; 
    m_runtxt->write(path);    
    const char* cachemetapath = getCacheMetaPath();
    m_cachemeta->save(cachemetapath);
}


/**
Opticks::loadOriginCacheMeta
-----------------------

Invoked by Opticks::configure 

* see GGeo::loadCacheMeta

**/

void Opticks::loadOriginCacheMeta() 
{
    const char* cachemetapath = getCacheMetaPath();
    LOG(info) << " cachemetapath " << cachemetapath ; 
    m_origin_cachemeta = NMeta::Load(cachemetapath); 
    m_origin_cachemeta->dump("Opticks::loadOriginCacheMeta"); 
    std::string gdmlpath = ExtractCacheMetaGDMLPath(m_origin_cachemeta); 
    LOG(info) << "ExtractCacheMetaGDMLPath " << gdmlpath ; 

    m_origin_gdmlpath = strdup(gdmlpath.c_str()); 
    m_origin_geocache_code_version = m_origin_cachemeta->get<int>(GEOCACHE_CODE_VERSION_KEY, "0" );  

    bool is_key_source = isKeySource(); 
    bool geocache_code_version_match = m_origin_geocache_code_version == Opticks::GEOCACHE_CODE_VERSION ; 
    bool geocache_code_version_pass = is_key_source || geocache_code_version_match ; 

    if(is_key_source)  // necessary to allow creation of new geocache 
    {
        LOG(info) 
           << " current executable isKeySource : hence immune to code matching requirements "
           << "\n (current) Opticks::GEOCACHE_CODE_VERSION " << Opticks::GEOCACHE_CODE_VERSION
           << "\n (loaded)  m_origin_geocache_code_version " << m_origin_geocache_code_version
           ;           
    }
    else if(!geocache_code_version_pass)
    {
        LOG(fatal) 
           << "\n (current) Opticks::GEOCACHE_CODE_VERSION " << Opticks::GEOCACHE_CODE_VERSION
           << "\n (loaded)  m_origin_geocache_code_version " << m_origin_geocache_code_version
           << "\n GEOCACHE_CODE_VERSION MISMATCH : PERSISTED CACHE VERSION DOES NOT MATCH CURRENT CODE "
           << "\n -> RECREATE THE CACHE EG WITH geocache-create "
           ; 
    }
    else
    {
         LOG(info) << "(pass) " << GEOCACHE_CODE_VERSION_KEY << " " << m_origin_geocache_code_version  ; 
    }
    assert( geocache_code_version_pass ); 
}

NMeta* Opticks::getOriginCacheMeta(const char* obj) const 
{
    return m_origin_cachemeta ? m_origin_cachemeta->getObj(obj) : NULL ; 
}

NMeta* Opticks::getGDMLAuxMeta() const 
{
    const char* gdmlauxmetapath = getGDMLAuxMetaPath();
    NMeta* gdmlauxmeta = NMeta::Load(gdmlauxmetapath) ;
    return gdmlauxmeta ; 
}

void Opticks::findGDMLAuxMetaEntries(std::vector<NMeta*>& entries, const char* k, const char* v ) const 
{
    NMeta* gam = getGDMLAuxMeta() ; 
    unsigned ni = gam ? gam->getNumKeys() : 0 ;
    bool dump = false ; 

    for(unsigned i=0 ; i < ni ; i++)
    {
        const char* subKey = gam->getKey(i); 
        NMeta* sub = gam->getObj(subKey); 

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


void Opticks::findGDMLAuxValues(std::vector<std::string>& values, const char* k, const char* v, const char* q) const 
{
    std::vector<NMeta*> entries ; 
    findGDMLAuxMetaEntries(entries, k, v );

    for(unsigned i=0 ; i < entries.size() ; i++)
    {
        NMeta* entry = entries[i]; 
        std::string qv = entry->get<std::string>(q) ; 
        values.push_back(qv); 
    }
}

/**
Opticks::getGDMLAuxTargetLVNames
----------------------------------

Consults the persisted GDMLAux metadata looking for entries with (k,v) pair ("label","target").
For any such entries the "lvname" property is accesses and added to the lvnames vector.

**/

unsigned Opticks::getGDMLAuxTargetLVNames(std::vector<std::string>& lvnames) const 
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
Opticks::getGDMLAuxTargetLVName
---------------------------------

Returns the first lvname or NULL

**/

const char* Opticks::getGDMLAuxTargetLVName() const 
{
    std::vector<std::string> lvnames ; 
    getGDMLAuxTargetLVNames(lvnames);
    return lvnames.size() > 0 ? strdup(lvnames[0].c_str()) : NULL ; 
}







/**
Opticks::ExtractCacheMetaGDMLPath
------------------------------------

TODO: avoid having to fish around in the geocache argline to get the gdmlpath in direct mode

Going via the tokpath enables sharing of geocaches across different installs. 

**/

std::string Opticks::ExtractCacheMetaGDMLPath(const NMeta* meta)  // static
{
    std::string argline = meta->get<std::string>("argline", "-");  

    LOG(LEVEL) << " argline " << argline ; 

    SArgs sa(0, NULL, argline.c_str()); 


    const char* executable = sa.elem[0].c_str(); 
  
    std::string sprefix = BFile::ParentParentDir(executable) ; 

    const char* prefix = sprefix.empty() ? NULL : sprefix.c_str(); 

    const char* gdmlpath = sa.get_arg_after("--gdmlpath", NULL) ; 

    bool consistent_prefix = prefix && gdmlpath && strncmp( gdmlpath, prefix, strlen(prefix) ) == 0 ;

    const char* relpath = consistent_prefix ? gdmlpath + strlen(prefix) + 1 : NULL ; 

    std::string tokpath ; 

    if(relpath) 
    {
        tokpath = BFile::FormPath("$OPTICKS_INSTALL_PREFIX", relpath ) ;
    } 
    else if( gdmlpath )
    {
        tokpath = gdmlpath ; 
    }   
    

    LOG(LEVEL) 
        << "\n argline " << argline
        << "\n executable " << executable 
        << "\n gdmlpath " << gdmlpath
        << "\n prefix " << prefix
        << "\n relpath " << relpath
        << "\n tokpath " << tokpath
        << "\n consistent_prefix " << consistent_prefix
        ;  

    return tokpath ; 
}





/*
Opticks::AutoRunLabel
-----------------------

TODO: More friendly label with name of the GPU ?

*/

const char* Opticks::AutoRunLabel(int rtx)
{
    const char* cvd = SSys::getenvvar("CUDA_VISIBLE_DEVICES", "") ; 
    std::stringstream ss ; 
    ss 
          << "R" << rtx
          << "_"
          << "cvd_" << cvd 
          ;

    std::string s = ss.str();  
    return strdup(s.c_str()); 
}

const char* Opticks::getRunLabel() const 
{
    const std::string& runlabel = m_cfg->getRunLabel() ;
    return runlabel.empty() ? AutoRunLabel(getRTX()) : runlabel.c_str() ;

}
const char* Opticks::getRunFolder() const 
{
    const std::string& runfolder = m_cfg->getRunFolder() ;
    return runfolder.c_str() ;
}

const char* Opticks::getDbgGDMLPath() const 
{
    const std::string& dbggdmlpath = m_cfg->getDbgGDMLPath() ;
    return dbggdmlpath.empty() ? NULL : dbggdmlpath.c_str() ;
}

const char* Opticks::getDebugGenstepPath(unsigned idx) const  
{
    const std::string& dbggsdir = m_cfg->getDbgGSDir() ;  // --dbggsdir
    const char* name = BStr::concat<unsigned>("",idx+1,".npy") ;  // 1-based to match signed tags  
    std::string path = BFile::FormPath(dbggsdir.c_str(), name ); 
    return strdup(path.c_str()); 
}


bool Opticks::isDbgGSImport() const  // --dbggsimport
{
    return m_cfg->hasOpt("dbggsimport") ;
} 
bool Opticks::isDbgGSSave() const   // --dbggssave
{
    return m_cfg->hasOpt("dbggssave");
}
bool Opticks::isDbgGSLoad() const   // --dbggsload
{
    return m_cfg->hasOpt("dbggsload");
}






bool Opticks::isTest() const   // --test
{
    return m_cfg->hasOpt("test");
}
bool Opticks::isTestAuto() const  // --testauto
{
    return m_cfg->hasOpt("testauto");
}

const char* Opticks::getTestConfig() const 
{
    const std::string& tc = m_cfg->getTestConfig() ;
    return tc.empty() ? NULL : tc.c_str() ; 
}




bool Opticks::isG4Snap() const   // --g4snap
{
    return m_cfg->hasOpt("g4snap");
}


const char* Opticks::getG4SnapConfigString()  const 
{
    return m_cfg->getG4SnapConfig().c_str() ; 
}

const char* Opticks::getSnapConfigString()
{
    return m_cfg->getSnapConfig().c_str() ; 
}

const char* Opticks::getLODConfigString()
{
    return m_cfg->getLODConfig().c_str() ; 
}
int  Opticks::getLOD()
{
    return m_cfg->getLOD(); 
}

NLODConfig* Opticks::getLODConfig()
{
    if(m_lod_config == NULL)
    {
        m_lod_config = new NLODConfig(getLODConfigString());
    }
    return m_lod_config ; 
}


NSnapConfig* Opticks::getSnapConfig()
{
    if(m_snap_config == NULL)
    {
        m_snap_config = new NSnapConfig(getSnapConfigString());
    }
    return m_snap_config ; 
}




int  Opticks::getDomainTarget() const  // --domaintarget 
{
    return m_cfg->getDomainTarget(); 
}
int  Opticks::getGenstepTarget() const  // --gensteptarget
{
    return m_cfg->getGenstepTarget(); 
}
int  Opticks::getTarget() const   // --target 
{
    return m_cfg->getTarget(); 
}



int  Opticks::getAlignLevel() const 
{
    return m_cfg->getAlignLevel(); 
}




 
unsigned Opticks::getVerbosity() const 
{
    return m_verbosity ; 
}
void  Opticks::setVerbosity(unsigned verbosity)
{
    m_verbosity = verbosity ; 
}



/**
Opticks::getInputUDet
-------------------------



**/

const char* Opticks::getInputUDet() const 
{
    const char* det = m_detector ;           // set by initResource
    const char* cat = m_cfg->getEventCat() ;  
    return cat && strlen(cat) > 0 ? cat : det ;  
}


/**
Opticks::defineEventSpec
-------------------------

Invoked from Opticks::configure after commandline parse and initResource.
The components of the spec determine file system paths of event files.


OpticksCfg::m_event_pfx "--pfx"
   event prefix for organization of event files, typically "source" or the name of 
   the creating executable or the testname 

OpticksCfg::m_event_cat "--cat" 
   event category for organization of event files, typically used instead of detector 
   for test geometries such as prism and lens, default ""

OpticksCfg::m_event_tag "--tag"
   event tag, non zero positive integer string identifying an event 



**/

const char* Opticks::DEFAULT_PFX = "default_pfx" ; 

void Opticks::defineEventSpec()
{
    const char* cat = m_cfg->getEventCat(); // expected to be defined for tests and equal to the TESTNAME from bash functions like tboolean-
    const char* udet = getInputUDet(); 
    const char* tag = m_cfg->getEventTag();
    const char* ntag = BStr::negate(tag) ; 
    const char* typ = getSourceType(); 

    const char* resource_pfx = m_resource->getEventPfx() ; 
    const char* config_pfx = m_cfg->getEventPfx() ; 
    const char* pfx = config_pfx ? config_pfx : resource_pfx ;  
    if( !pfx )
    { 
        pfx = DEFAULT_PFX ; 
        LOG(fatal) 
            << " resource_pfx " << resource_pfx 
            << " config_pfx " << config_pfx 
            << " pfx " << pfx 
            << " cat " << cat
            << " udet " << udet
            << " typ " << typ
            << " tag " << tag
            ;
    }
    //assert( pfx ); 


    m_spec  = new OpticksEventSpec(pfx, typ,  tag, udet, cat );
    m_nspec = new OpticksEventSpec(pfx, typ, ntag, udet, cat );

    LOG(LEVEL) 
         << " pfx " << pfx
         << " typ " << typ
         << " tag " << tag
         << " ntag " << ntag
         << " udet " << udet 
         << " cat " << cat 
         ;

}

void Opticks::dumpArgs(const char* msg)
{
    LOG(LEVEL) << msg << " argc " << m_argc ;
    for(int i=0 ; i < m_argc ; i++) 
        LOG(LEVEL) << std::setw(3) << i << " : " << m_argv[i]  ;

   // PLOG by default writes to stdout so for easy splitting write 
   // mostly to stdout and just messages to stderr
}


void Opticks::checkOptionValidity()
{
    if(isInterop() && getMultiEvent() > 1)
    {
        LOG(fatal) << "INTEROP mode with --multievent greater than 1 is not supported " ;  
        setExit(true);
    }

   if(isXAnalytic() && getRTX() == 2)
   {
        LOG(fatal) << " --xanalytic --rtx 2 : no point doing that as rtx 2 and 1 are the same when not dealing with triangles " ;  
        setExit(true);
   }
}

bool Opticks::isConfigured() const
{
    return m_configured ; 
}
void Opticks::configure()
{
    if(m_configured) return ; 
    m_configured = true ; 

    dumpArgs("Opticks::configure");  


    m_cfg->commandline(m_argc, m_argv);

    checkOptionValidity();


    const std::string& cvdcfg = m_cfg->getCVD();
    const char* cvd = cvdcfg.empty() ? NULL : cvdcfg.c_str() ; 

    // Instead of just failing in interop mode with no cvd argument offer a reprieve :  
    // set envvar OPTICKS_DEFAULT_INTEROP_CVD when using multi-gpu workstations, 
    // it should point to the GPU driving the monitor.

    const char* dk = "OPTICKS_DEFAULT_INTEROP_CVD" ; 
    const char* dcvd = SSys::getenvvar(dk) ;  
    if( cvd == NULL && isInterop() && dcvd != NULL )
    {
        LOG(LEVEL) << " --interop mode with no cvd specified, adopting OPTICKS_DEFAULT_INTEROP_CVD hinted by envvar [" << dcvd << "]" ;   
        cvd = strdup(dcvd);   
    }

    if(cvd)
    { 
        const char* ek = "CUDA_VISIBLE_DEVICES" ; 
        LOG(info) << " setting " << ek << " envvar internally to " << cvd ; 
        SSys::setenvvar(ek, cvd, true );    // Opticks::configure setting CUDA_VISIBLE_DEVICES
    }


    initResource();  

    updateCacheMeta(); 

    if(isDirect())
    {
        loadOriginCacheMeta();  // sets m_origin_gdmlpath
    }


    defineEventSpec();  


    const std::string& ssize = m_cfg->getSize();

    if(!ssize.empty()) 
    {
        m_size = guvec4(ssize);
    }
    else if(m_cfg->hasOpt("fullscreen"))
    {
#ifdef __APPLE__
        //m_size = glm::uvec4(2880,1800,2,0) ;
        m_size = glm::uvec4(1440,900,2,0) ;
#else
        m_size = glm::uvec4(2560,1440,1,0) ;
#endif
    } 
    else
    {
#ifdef __APPLE__
        m_size = glm::uvec4(2880,1704,2,0) ;  // 1800-44-44px native height of menubar  
#else
        m_size = glm::uvec4(1920,1080,1,0) ;
#endif
    }


    const std::string& sposition = m_cfg->getPosition();
    if(!sposition.empty()) 
    {
        m_position = guvec4(sposition);
    }
    else
    {
#ifdef __APPLE__
        m_position = glm::uvec4(200,200,0,0) ;  // top left
#else
        m_position = glm::uvec4(100,100,0,0) ;  // top left
#endif
    }



    const char* type = "State" ; 
    const std::string& stag = m_cfg->getStateTag();
    const char* subtype = stag.empty() ? NULL : stag.c_str() ; 

    std::string prefdir = getPreferenceDir(type, subtype);  

    LOG(LEVEL) 
          << " m_size " << gformat(m_size)
          << " m_position " << gformat(m_position)
          << " prefdir " << prefdir
          ;
 

    // Below "state" is a placeholder name of the current state that never gets persisted, 
    // names like 001 002 are used for persisted states : ie the .ini files within the prefdir

    m_state = new NState(prefdir.c_str(), "state")  ;


    const std::string& mpfx = m_cfg->getMaterialPrefix();
    m_materialprefix = ( mpfx.empty() || isJuno()) ? NULL : strdup(mpfx.c_str()) ;


    m_photons_per_g4event = m_cfg->getNumPhotonsPerG4Event();
    m_dbg->postconfigure();




    m_verbosity = m_cfg->getVerbosity(); 

    //configureCheckGeometryFiles();

    configureGeometryHandling();


    if(hasOpt("dumpenv")) 
         BEnv::dumpEnvironment("Opticks::configure --dumpenv", "G4,OPTICKS,DAE,IDPATH") ; 


    LOG(debug) << "Opticks::configure DONE " 
              << " verbosity " << m_verbosity 
              ;

}




void Opticks::configureGeometryHandling()
{
    bool geocache = !m_cfg->hasOpt("nogeocache") ;
    bool instanced = !m_cfg->hasOpt("noinstanced") ; // find repeated geometry 

    LOG(debug) << "Opticks::configureGeometryHandling"
              << " geocache " << geocache 
              << " instanced " << instanced
              ;   

    setGeocacheEnabled(geocache);
    setInstanced(instanced); // find repeated geometry 
}

void Opticks::setGeocacheEnabled(bool geocache)
{
    m_geocache = geocache ; 
}
bool Opticks::isGeocacheEnabled() const 
{
    return m_geocache ;
}

bool Opticks::hasGeocache() const 
{
    const char* idpath = getIdPath(); 
    return BFile::ExistsDir(idpath); 
} 

bool Opticks::isGeocacheAvailable() const 
{
    bool cache_exists = hasGeocache(); 
    bool cache_enabled = isGeocacheEnabled() ; 
    return cache_exists && cache_enabled ;
}










void Opticks::setInstanced(bool instanced)
{
   m_instanced = instanced ;
}
bool Opticks::isInstanced()
{
   return m_instanced ; 
}





bool Opticks::isEnabledLegacyG4DAE() const 
{
    return m_cfg->hasOpt("enabled_legacy_g4dae") ;  
}


//bool Opticks::isLocalG4() const { return m_cfg->hasOpt("localg4") ;  }






void Opticks::dump(const char* msg) 
{
    LOG(info) << msg  ;

    const char* dbgmesh = getDbgMesh();

    std::cout
         << " argline " << std::setw(30) << getArgLine() << std::endl 
         << " dbgnode " << std::setw(30) << getDbgNode() << std::endl 
         << " dbgmesh " << std::setw(30) << ( dbgmesh ? dbgmesh : "-" ) << std::endl
         ;

}


void Opticks::Summary(const char* msg)
{
    LOG(info) << msg 
              << " sourceCode " << getSourceCode() 
              << " sourceType " << getSourceType() 
              << " mode " << m_mode->description()
              ; 

    m_resource->Summary(msg);

    std::cout
        << std::setw(40) << " isInternal "
        << std::setw(40) << isInternal()
        << std::endl
        << std::setw(40) << " Verbosity "
        << std::setw(40) << getVerbosity()
        << std::endl
        << std::setw(40) << " AnalyticPMTMedium "
        << std::setw(40) << getAnalyticPMTMedium()
        << std::endl
        ;

    LOG(info) << msg << "DONE" ; 
}





int Opticks::getLastArgInt()
{
    return BStr::atoi(m_lastarg, -1 );
}

int Opticks::getInteractivityLevel() const 
{
    return m_mode->getInteractivityLevel() ; 

}


/**
Opticks::setSpaceDomain
-----------------------

Invoked by OpticksAim::registerGeometry

**/

void Opticks::setSpaceDomain(const glm::vec4& sd)
{
    setSpaceDomain(sd.x, sd.y, sd.z, sd.w )  ; 
}

void Opticks::setSpaceDomain(float x, float y, float z, float w)
{
    if( m_space_domain.x != 0.f && m_space_domain.x != x ) LOG(fatal) << " changing x " << m_space_domain.x << " -> " << x ;  
    if( m_space_domain.y != 0.f && m_space_domain.y != y ) LOG(fatal) << " changing y " << m_space_domain.y << " -> " << y ;  
    if( m_space_domain.z != 0.f && m_space_domain.z != z ) LOG(fatal) << " changing z " << m_space_domain.z << " -> " << z ;  
    if( m_space_domain.w != 0.f && m_space_domain.w != w ) LOG(fatal) << " changing w " << m_space_domain.w << " -> " << w ;  

    m_space_domain.x = x  ; 
    m_space_domain.y = y  ; 
    m_space_domain.z = z  ; 
    m_space_domain.w = w  ; 

    if( isDbgAim() ) 
    {  
        LOG(info) << " --dbgaim : m_space_domain " << gformat(m_space_domain) ;  
    }

    setupTimeDomain(w); 

    postgeometry();  
}


/**
Opticks::setupTimeDomain
-------------------------

Invoked by setSpaceDomain

When configured values of "--timemax" and "--animtimemax" are 
negative (this is default) a rule of thumb is used to setup a timedomain 
suitable for the extent of space domain.

When the propagation yields times exeeding timemax, the 
domain compression will return SHRT_MIN for them 
which will get translated back as -timemax.  

Initial rule of thumb  2.f*extent/speed_of_light results in 
some times trying to go over domain, so upped the factor to 3.f
Now the factor can be changed with --timemaxthumb option
and are upping the default to 6.f

NB its better not to change this frequently, as it effects 
event records 

For geometries like DYB with a very large world volume which 
is not relevant to optical photon propagation the rule of thumb 
yields an overlarge time domain resulting in difficult to 
control animated propagation visualizations.



**/

void Opticks::setupTimeDomain(float extent)
{
    float timemaxthumb = m_cfg->getTimeMaxThumb();  // factor
    float timemax = m_cfg->getTimeMax();  // ns
    float animtimemax = m_cfg->getAnimTimeMax() ; 

    glm::vec4 animtimerange(0., -1.f, 0.f, 0.f ); 
    getAnimTimeRange( animtimerange ); 
    float speed_of_light = 300.f ;        // mm/ns 
    float rule_of_thumb_timemax = timemaxthumb*extent/speed_of_light ;
   

    if(isDbgAim()) // --dbgaim
    {
        LOG(info)
            << "\n [--dbgaim] output "
            << "\n extent (mm) " << extent 
            << "\n cfg.timemax (ns) [--timemax]" << timemax    
            << "\n animtimerange " << gformat(animtimerange) 
            << "\n cfg.getTimeMaxThumb [--timemaxthumb] " << timemaxthumb 
            << "\n cfg.getAnimTimeMax [--animtimemax] " << animtimemax 
            << "\n cfg.getAnimTimeMax [--animtimemax] " << animtimemax 
            << "\n speed_of_light (mm/ns) " << speed_of_light
            << "\n rule_of_thumb_timemax (ns) " << rule_of_thumb_timemax 
            ;  
    }


    float u_timemin = 0.f ;  // ns
    float u_timemax = timemax < 0.f ? rule_of_thumb_timemax : timemax ;  
    float u_animtimemax = animtimemax < 0.f ? u_timemax : animtimemax ; 

    m_time_domain.x = u_timemin ;
    m_time_domain.y = u_timemax ;
    m_time_domain.z = u_animtimemax ;
    m_time_domain.w = 0.f  ;

    if(isDbgAim())
    {
        LOG(info)
            << "\n [--dbgaim] output "
            << "\n u_timemin " << u_timemin
            << "\n u_timemax " << u_timemax
            << "\n u_animtimemax " << u_animtimemax
            << "\n m_time_domain " << gformat(m_time_domain) 
            ;  
    }


}








int Opticks::getMultiEvent() const 
{    
    return m_cfg->getMultiEvent();
}

int Opticks::getCameraType() const 
{    
    return m_cfg->getCameraType();
}


/**
Opticks::getGenerateOverride
------------------------------

Used by m_emitter for generated input photons, 
see opticksgeo/OpticksGen.cc  and NEmitPhotonsNPY.
When a value greater than zero is returned the emitcfg 
number of photons is overridden.

**/

int Opticks::getGenerateOverride() const 
{
    return m_cfg->getGenerateOverride();
}

/**
Opticks::getPropagateOverride
---------------------------------

Used by OPropagator::prelaunch to override the size of the 
OptiX launch see optixrap/OPropagator.cc when a value greater that 
zero is returned.

**/

int Opticks::getPropagateOverride() const 
{
    return m_cfg->getPropagateOverride();
}





/**
Opticks::postgeometry
------------------------

Invoked by Opticks::setSpaceDomain

**/

void Opticks::postgeometry()
{
    configureDomains();
    setProfileDir(getEventFold());
}


/**
Opticks::configureDomains
--------------------------

This is triggered by setSpaceDomain which is 
invoked when geometry is loaded, canonically by OpticksAim::registerGeometry 


**/

void Opticks::configureDomains()
{
   m_domains_configured = true ; 

   m_wavelength_domain = getDefaultDomainSpec() ;  

   int e_rng_max = SSys::getenvint("CUDAWRAP_RNG_MAX",-1); 

   int x_rng_max = getRngMax() ;

   if(e_rng_max != x_rng_max)
       LOG(verbose) << "Opticks::configureDomains"
                  << " CUDAWRAP_RNG_MAX " << e_rng_max 
                  << " x_rng_max " << x_rng_max 
                  ;

   //assert(e_rng_max == x_rng_max && "Configured RngMax must match envvar CUDAWRAP_RNG_MAX and corresponding files, see cudawrap- ");    
}

float Opticks::getTimeMin() const 
{
    return m_time_domain.x ; 
}
float Opticks::getTimeMax() const 
{
    return m_time_domain.y ; 
}
float Opticks::getAnimTimeMax() const 
{
    return m_time_domain.z ; 
}











std::string Opticks::description() const 
{
    std::stringstream ss ; 
    ss << "Opticks"
       << " time " << gformat(m_time_domain)  
       << " space " << gformat(m_space_domain) 
       << " wavelength " << gformat(m_wavelength_domain) 
       ;
    return ss.str();
}

std::string Opticks::desc() const 
{
    std::stringstream ss ; 
    BOpticksKey* key = getKey() ;
    ss << "Opticks.desc"
       << std::endl 
       << ( key ? key->desc() : "NULL-key?" )
       << std::endl
       << "IdPath : " << getIdPath() 
       << std::endl
       ; 
    return ss.str();
}


std::string Opticks::export_() const 
{
    BOpticksKey* key = getKey() ;
    std::stringstream ss ; 
    ss 
        << "# Opticks::export_ " 
        << "\n"
        << "export OPTICKS_KEY=" << key->getSpec() 
        << "\n"
        << "export OPTICKS_IDPATH=" << getIdPath()
        << "\n"
        ;
    return ss.str();
}


/**
Opticks::getSourceCode
-------------------------

This is not the final word, see OpticksGen 

*live-gensteps* 
    G4GUN: collected from a live CG4 instance  

*loaded-from-file*
    CERENKOV SCINTILLATION NATURAL

*fabricated-from-config*
    TORCH  

**/

unsigned int Opticks::getSourceCode() const
{
    unsigned int code ;
    if(     m_cfg->hasOpt("natural"))       code = OpticksGenstep_NATURAL ;     // doing (CERENKOV | SCINTILLATION) would entail too many changes 
    else if(m_cfg->hasOpt("cerenkov"))      code = OpticksGenstep_G4Cerenkov_1042  ;
    else if(m_cfg->hasOpt("scintillation")) code = OpticksGenstep_DsG4Scintillation_r3971 ;
    else if(m_cfg->hasOpt("torch"))         code = OpticksGenstep_TORCH ;
    else if(m_cfg->hasOpt("machinery"))     code = OpticksGenstep_MACHINERY ;
    else if(m_cfg->hasOpt("g4gun"))         code = OpticksGenstep_G4GUN ;           // <-- dynamic : photon count not known ahead of time
    else if(m_cfg->hasOpt("emitsource"))    code = OpticksGenstep_EMITSOURCE ;      
    else if(m_cfg->hasOpt("primarysource")) code = OpticksGenstep_PRIMARYSOURCE ;   // <-- dynamic : photon count not known ahead of time
    else                                    code = OpticksGenstep_TORCH ;             
    return code ;
}

// not-definitive see OpticksGen CGenerator
const char* Opticks::getSourceType() const
{
    unsigned int code = getSourceCode();
    //return OpticksFlags::SourceTypeLowercase(code) ; 
    return OpticksFlags::SourceType(code) ; 
}

bool Opticks::isFabricatedGensteps() const
{
    unsigned int code = getSourceCode() ;
    return code == OpticksGenstep_TORCH || code == OpticksGenstep_MACHINERY ;  
}
bool Opticks::isG4GUNGensteps() const
{
    unsigned int code = getSourceCode() ;
    return code == OpticksGenstep_G4GUN ;  
}






bool Opticks::isEmbedded() const { return hasOpt("embedded"); }   
// HMM CAN isEmbedded BE EQUATED WITH hasKey ? no the initial run which creates the geocache is
// not run with "--envkey" although it does create a key : at what juncture ? If it 
// is early enough (or can be moved early) could equate them.



bool Opticks::hasKey() const { return m_resource->hasKey() ; }
bool Opticks::isDirect() const { return isEmbedded() || hasKey() ; }
bool Opticks::isLegacy() const { return !isDirect() ; } 

std::string Opticks::getLegacyDesc() const 
{
    std::stringstream ss ; 
    ss 
        << " hasKey " << hasKey() 
        << " isEmbedded " << isEmbedded()
        << " isDirect " << isDirect()
        << " isLegacy " << isLegacy()
        ;  
  
    return ss.str(); 
}





bool Opticks::isLiveGensteps() const {  return hasOpt("live"); }
bool Opticks::isNoInputGensteps() const { return hasOpt("load|nopropagate") ; } 




char Opticks::getEntryCode() const  // debug switching of OptiX launch program  
{
   return OpticksEntry::CodeFromConfig(m_cfg);
}
const char* Opticks::getEntryName() const
{  
    char code = getEntryCode();
    return OpticksEntry::Name(code);
}
bool Opticks::isTrivial() const
{
   char code = getEntryCode();
   return  code == 'T' ; 
}
bool Opticks::isSeedtest() const
{
   char code = getEntryCode();
   return  code == 'S' ; 
}




const char* Opticks::getEventFold() const
{
   return m_spec ? m_spec->getFold() : NULL ;
}

const char* Opticks::getEventDir() const 
{
    return m_spec ? m_spec->getDir() : NULL ;
}


const char* Opticks::getEventPfx() const
{
    return m_spec->getPfx();
}
const char* Opticks::getEventTag() const
{
    return m_spec->getTag();
}
int Opticks::getEventITag() const
{
    return m_spec->getITag() ; 
}
const char* Opticks::getEventCat() const
{
    return m_spec->getCat();
}
const char* Opticks::getEventDet() const
{
    return m_spec->getDet();
}







Index* Opticks::loadHistoryIndex()
{
    const char* pfx = getEventPfx();
    const char* typ = getSourceType();
    const char* tag = getEventTag();
    const char* udet = getEventDet();

    Index* index = OpticksEvent::loadHistoryIndex(pfx, typ, tag, udet) ;

    return index ; 
}
Index* Opticks::loadMaterialIndex()
{
    const char* pfx = getEventPfx();
    const char* typ = getSourceType();
    const char* tag = getEventTag();
    const char* udet = getEventDet();

    return OpticksEvent::loadMaterialIndex(pfx, typ, tag, udet ) ;
}
Index* Opticks::loadBoundaryIndex()
{
    const char* pfx = getEventPfx();
    const char* typ = getSourceType();
    const char* tag = getEventTag();
    const char* udet = getEventDet();

    return OpticksEvent::loadBoundaryIndex(pfx, typ, tag, udet ) ;
}

/**
Opticks::makeDynamicDefine
----------------------------

MATERIAL_COLOR_OFFSET, FLAG_COLOR_OFFSET
    used by oglrap/gl/fcolor.h

MAXREC
    used by oglrap/gl/{,alt,dev}rec/geom.glsl for photon picking interface 
    (not active for many years)

MAXTIME
    appears unused

**/

BDynamicDefine* Opticks::makeDynamicDefine()
{
    BDynamicDefine* dd = new BDynamicDefine();   // configuration used in oglrap- shaders
    dd->add("MAXREC",m_cfg->getRecordMax());    
    dd->add("MAXTIME",m_cfg->getTimeMax());    
    dd->add("PNUMQUAD", 4);  // quads per photon
    dd->add("RNUMQUAD", 2);  // quads per record 
    dd->add("MATERIAL_COLOR_OFFSET", (unsigned int)OpticksColors::MATERIAL_COLOR_OFFSET );
    dd->add("FLAG_COLOR_OFFSET", (unsigned int)OpticksColors::FLAG_COLOR_OFFSET );
    dd->add("PSYCHEDELIC_COLOR_OFFSET", (unsigned int)OpticksColors::PSYCHEDELIC_COLOR_OFFSET );
    dd->add("SPECTRAL_COLOR_OFFSET", (unsigned int)OpticksColors::SPECTRAL_COLOR_OFFSET );

    return dd ; 
}


OpticksEventSpec* Opticks::getEventSpec()
{
    return m_spec ; 
}




OpticksEvent* Opticks::loadEvent(bool ok, unsigned tagoffset)
{
    OpticksEvent* evt = OpticksEvent::make(ok ? m_spec : m_nspec, tagoffset);

    evt->setOpticks(this);

    bool verbose = false ; 
    evt->loadBuffers(verbose);


    LOG(info) << "Opticks::loadEvent"
              << " tagdir " << evt->getTagDir() 
              << " " << ( evt->isNoLoad() ? "FAILED" : "SUCEEDED" )
              ; 


    return evt ; 
}

void Opticks::setTagOffset(unsigned tagoffset)
{
    m_tagoffset = tagoffset ; 
}
unsigned Opticks::getTagOffset()
{
    return m_tagoffset ; 
}

/**
Opticks::makeEvent
---------------------




**/
OpticksEvent* Opticks::makeEvent(bool ok, unsigned tagoffset)
{
    setTagOffset(tagoffset) ; 

    OpticksEvent* evt = OpticksEvent::make(ok ? m_spec : m_nspec, tagoffset);

    evt->setId(m_event_count) ;   // starts from id 0 
    evt->setOpticks(this);
    evt->setEntryCode(getEntryCode());

    LOG(LEVEL) 
        << ( ok ? " OK " : " G4 " )
        << " tagoffset " << tagoffset 
        << " id " << evt->getId() 
        ;

    m_event_count += 1 ; 


    const char* x_udet = getEventDet();
    const char* e_udet = evt->getUDet();

    bool match = strcmp(e_udet, x_udet) == 0 ;
    if(!match)
    {
        LOG(fatal) 
                   << " MISMATCH "
                   << " x_udet " << x_udet 
                   << " e_udet " << e_udet 
                   ;
    }
    assert(match);

    evt->setMode(m_mode);

    if(!m_domains_configured)
         LOG(fatal) 
             << " domains MUST be configured by calling setSpaceDomain "
             << " prior to makeEvent being possible "
             << " description " << description()
             ;

    assert(m_domains_configured);


    unsigned rng_max = getRngMax() ;
    unsigned bounce_max = getBounceMax() ;
    unsigned record_max = getRecordMax() ;
    
    evt->setTimeDomain(getTimeDomain());
    evt->setSpaceDomain(getSpaceDomain());  
    evt->setWavelengthDomain(getWavelengthDomain());

    evt->setMaxRng(rng_max);
    evt->setMaxRec(record_max);
    evt->setMaxBounce(bounce_max);

    evt->createSpec();   
    evt->createBuffers();  // not-allocated and with itemcount 0 
 
    // ctor args define the identity of the Evt, coming in from config
    // other params are best keep in m_parameters where they get saved/loaded  
    // with the evt 

    NMeta*       parameters = evt->getParameters();
    parameters->add<unsigned int>("RngMax",    rng_max );
    parameters->add<unsigned int>("BounceMax", bounce_max );
    parameters->add<unsigned int>("RecordMax", record_max );

    parameters->add<std::string>("mode", m_mode->description());
    parameters->add<std::string>("cmdline", m_cfg->getCommandLine() );

    parameters->add<std::string>("EntryCode", BStr::ctoa(getEntryCode()) );
    parameters->add<std::string>("EntryName", getEntryName() );

    parameters->add<std::string>("KEY",  getKeySpec() ); 
    parameters->add<std::string>("GEOCACHE",  getIdPath() ); 
    // formerly would have called this IDPATH, now using GEOCACHE to indicate new approach 

    evt->setCreator(SProc::ExecutablePath()) ; // no access to argv[0] for embedded running 

    assert( parameters->get<unsigned int>("RngMax") == rng_max );
    assert( parameters->get<unsigned int>("BounceMax") == bounce_max );
    assert( parameters->get<unsigned int>("RecordMax") == record_max );

    // TODO: use these parameters from here, instead of from config again ?

    m_settings.x = bounce_max ;   
    m_settings.y = rng_max ;   
    m_settings.z = 0 ;   
    m_settings.w = record_max ;   

    return evt ; 
}


void Opticks::setOptiXVersion(unsigned version)
{
    m_parameters->add<unsigned>("OptiXVersion",version);
}
void Opticks::setGeant4Version(unsigned version)
{
    m_parameters->add<unsigned>("Geant4Version",version);
}

unsigned Opticks::getOptiXVersion()
{
    return m_parameters->get<unsigned>("OptiXVersion",0);
}
unsigned Opticks::getGeant4Version()
{
    return m_parameters->get<unsigned>("Geant4Version",0);
}



const char* Opticks::getDirectGenstepPath(unsigned tagoffset) const 
{
    const char* det = m_spec->getDet();
    const char* typ = m_spec->getTyp();
    const char* tag = m_spec->getOffsetTag(tagoffset);

    const char* srctagdir = BOpticksEvent::srctagdir(det, typ, tag ); 

    LOG(debug) 
              << " det " << det 
              << " typ " << typ 
              << " tag " << tag
              << " srctagdir " << srctagdir
              ; 

    std::string path = BFile::FormPath( srctagdir, "gs.npy" ); 
    return strdup(path.c_str())  ; 
}


const char* Opticks::getLegacyGenstepPath() const 
{
    const char* det = m_spec->getDet();
    const char* typ = m_spec->getTyp();
    const char* tag = m_spec->getTag();

    std::string path = NLoad::GenstepsPath(det, typ, tag);

    LOG(debug) 
              << " det " << det 
              << " typ " << typ 
              << " tag " << tag
              << " path " << path
              ; 

    return strdup(path.c_str()) ; 
}





/**
Opticks::getGenstepPath
-------------------------

Legacy genstep paths carry the tag in their stems::

    /usr/local/opticks/opticksdata/gensteps/dayabay/scintillation/./1.npy 


const char* Opticks::getGenstepPath() const 
{
    return hasKey() ? getDirectGenstepPath() : getLegacyGenstepPath() ; 
}

**/


bool Opticks::existsDirectGenstepPath(unsigned tagoffset) const 
{
    const char* path = getDirectGenstepPath(tagoffset);
    bool exists = path ? BFile::ExistsFile(path) : false ;
    LOG(LEVEL) 
       << " path " << path 
       << " exists " << exists 
       ;

    return exists ; 
}

bool Opticks::existsLegacyGenstepPath() const 
{
    const char* path = getLegacyGenstepPath();
    bool exists = path ? BFile::ExistsFile(path) : false ;
    LOG(error) 
       << " path " << path 
       << " exists " << exists 
       ;

    return exists ; 
}

bool Opticks::existsDebugGenstepPath(unsigned tagoffset) const 
{
    const char* path = getDebugGenstepPath(tagoffset);
    bool exists = path ? BFile::ExistsFile(path) : false ;
    LOG(LEVEL) 
       << " path " << path 
       << " exists " << exists 
       ;

    return exists ; 
}

NPY<float>* Opticks::findGensteps( unsigned tagoffset ) const 
{
    LOG(LEVEL) << "[ tagoffset " ;  

    NPY<float>* gs = NULL ; 
    if( hasKey() && !isTest() )
    {   
        if( isDbgGSLoad() && existsDebugGenstepPath(tagoffset) )
        {
            gs = loadDebugGenstep(tagoffset) ; 
        }
        else if( existsDirectGenstepPath(tagoffset) )
        {   
            gs = loadDirectGenstep(tagoffset) ; 
        }   
    }   
    LOG(LEVEL) << "] gs " << gs ;  
    return gs ; 
}


NPY<float>* Opticks::load(const char* path) const 
{
    NPY<float>* a = NPY<float>::load(path);
    if(a)
    {
        std::string val = path ; 
        a->setMeta<std::string>("loadpath", val ); 
    }

    if(!a)
    {
        LOG(warning) << "Opticks::load"
                     << " FAILED TO LOAD FROM "
                     << " path " << path 
                     ; 
        return NULL ;
    }
    return a ; 
}

NPY<float>* Opticks::loadDirectGenstep(unsigned tagoffset) const 
{
    std::string path = getDirectGenstepPath(tagoffset);
    return load(path.c_str()); 
}
NPY<float>* Opticks::loadDebugGenstep(unsigned tagoffset) const 
{
    std::string path = getDebugGenstepPath(tagoffset);
    return load(path.c_str()); 
}
NPY<float>* Opticks::loadLegacyGenstep() const 
{
    std::string path = getLegacyGenstepPath();
    return load(path.c_str()); 
}




/*
bool Opticks::existsDirectGenstepPath() const 
{
    const char* path = getDirectGenstepPath();
    bool exists = path ? BFile::ExistsFile(path) : false ;
    LOG(error) 
       << " path " << path 
       << " exists " << exists 
       ;

    return exists ; 
}

NPY<float>* Opticks::loadDirectGenstep() const 
{
    std::string path = getDirectGenstepPath();
    return load(path.c_str()); 
}

const char* Opticks::getPrimariesPath() const { return m_resource->getPrimariesPath() ; } 

bool Opticks::existsPrimariesPath() const 
{
    const char* path = getPrimariesPath();
    return path ? BFile::ExistsFile(path) : false ; 
}


NPY<float>* Opticks::loadPrimaries() const 
{
    const char* path = getPrimariesPath();
    return load(path); 
}

*/



const char* Opticks::getMaterialPrefix()
{
    return m_materialprefix ; 
}

const char* Opticks::Material(const unsigned int mat)
{
    if(G_MATERIAL_NAMES == NULL)
    {
        LOG(info) << "Opticks::Material populating global G_MATERIAL_NAMES " ;
        G_MATERIAL_NAMES = new BPropNames("GMaterialLib") ;
    }
    return G_MATERIAL_NAMES ? G_MATERIAL_NAMES->getLine(mat) : "Opticks::Material-ERROR-NO-GMaterialLib" ;
}

std::string Opticks::MaterialSequence(const unsigned long long seqmat)
{
    LOG(info) << "Opticks::MaterialSequence"
              << " seqmat " << std::hex << seqmat << std::dec ; 

    std::stringstream ss ;
    assert(sizeof(unsigned long long)*8 == 16*4);
    for(unsigned int i=0 ; i < 16 ; i++)
    {
        unsigned long long m = (seqmat >> i*4) & 0xF ; 

        const char* mat = Opticks::Material(m)  ; 

        ss << ( mat ? mat : "NULL" ) << " " ;
    }
    return ss.str();
}

/**
Opticks::makeSimpleTorchStep
-----------------------------

TODO: relocate into OpticksGen ?

**/

TorchStepNPY* Opticks::makeSimpleTorchStep(unsigned gencode)
{
    assert( gencode == OpticksGenstep_TORCH ); 

    const std::string& config = m_cfg->getTorchConfig() ;

    const char* cfg = config.empty() ? NULL : config.c_str() ;

    LOG(fatal)
              << " enable : --torch (the default) "
              << " configure : --torchconfig [" << ( cfg ? cfg : "NULL" ) << "]" 
              << " dump details : --torchdbg " 
              ;

    TorchStepNPY* torchstep = new TorchStepNPY(OpticksGenstep_TORCH, 1, cfg ); // see notes/issues/G4StepNPY_gencode_assert.rst

    unsigned int photons_per_g4event = m_cfg->getNumPhotonsPerG4Event() ;  // only used for cfg4-

    torchstep->setNumPhotonsPerG4Event(photons_per_g4event);

    return torchstep ; 
}


unsigned Opticks::getNumPhotonsPerG4Event(){ return m_cfg->getNumPhotonsPerG4Event() ; }
unsigned Opticks::getRngMax(){       return m_cfg->getRngMax(); }
unsigned Opticks::getBounceMax() {   return m_cfg->getBounceMax(); }
unsigned Opticks::getRecordMax() {   return m_cfg->getRecordMax() ; }

float Opticks::getEpsilon() const {            return m_cfg->getEpsilon()  ; }
float Opticks::getPixelTimeScale() const {    return m_cfg->getPixelTimeScale()  ; }
int   Opticks::getCurFlatSigInt() const {     return m_cfg->getCurFlatSigInt()  ; }
int   Opticks::getBoundaryStepSigInt() const {     return m_cfg->getBoundaryStepSigInt()  ; }


bool Opticks::hasOpt(const char* name) const { return m_cfg->hasOpt(name); }

bool Opticks::operator()(const char* name) const 
{
    return m_cfg->hasOpt(name) ;
} 


const char* Opticks::getAnaKey() const 
{
    const std::string& s = m_cfg->getAnaKey();
    return s.empty() ? NULL : s.c_str() ; 
}
const char* Opticks::getAnaKeyArgs() const 
{
    std::string s = m_cfg->getAnaKeyArgs();
    if(s.empty()) return NULL ; 
    BStr::replace_all(s, "_", " ");
    return strdup(s.c_str()) ; 
}
const char* Opticks::getG4GunConfig() const 
{
    const std::string& s = m_cfg->getG4GunConfig();
    return s.empty() ? NULL : s.c_str() ; 
}


const char* Opticks::getExampleMaterialNames() { return m_resource->getExampleMaterialNames(); }
const char* Opticks::getDefaultMaterial() { return m_resource->getDefaultMaterial(); }
const char* Opticks::getDetector() { return m_resource->getDetector(); }
bool Opticks::isJuno() {    return m_resource->isJuno(); }
bool Opticks::isDayabay() { return m_resource->isDayabay(); }
bool Opticks::isPmtInBox(){ return m_resource->isPmtInBox(); }
bool Opticks::isOther() {   return m_resource->isOther(); }
bool Opticks::isValid() {   return m_resource->isValid(); }
bool Opticks::hasCtrlKey(const char* key) const  { return m_resource->hasCtrlKey(key); }
bool Opticks::hasVolnames() const { return !hasCtrlKey("novolnames") ; }

const char* Opticks::getRNGDir() { return m_resource->getRNGDir(); } 

std::string Opticks::getPreferenceDir(const char* type, const char* subtype)
{
    const char* udet = getEventDet();
    return m_resource->getPreferenceDir(type, udet, subtype);
}

std::string Opticks::getObjectPath(const char* name, unsigned int ridx, bool relative) const 
{
   return relative ?
                     m_resource->getRelativePath(name, ridx)
                   :
                     m_resource->getObjectPath(name, ridx)
                   ; 
}

std::string Opticks::getObjectPath(const char* name, bool relative) const  
{
   return relative ?
                     m_resource->getRelativePath(name)
                   :
                     m_resource->getObjectPath(name)
                   ; 
}





std::string Opticks::formCacheRelativePath(const char* path) { return m_resource->formCacheRelativePath(path); }

OpticksQuery*   Opticks::getQuery() {     return m_resource->getQuery(); }
OpticksColors*  Opticks::getColors() {    return m_resource->getColors(); }
OpticksFlags*   Opticks::getFlags() const { return m_resource->getFlags(); }
OpticksAttrSeq* Opticks::getFlagNames() { return m_resource->getFlagNames(); }

std::map<unsigned int, std::string> Opticks::getFlagNamesMap()
{   
    return m_resource->getFlagNamesMap() ;
}


Types*          Opticks::getTypes() {     return m_resource->getTypes(); }
Typ*            Opticks::getTyp() {       return m_resource->getTyp(); }



NSensorList*    Opticks::getSensorList(){ return m_resource ? m_resource->getSensorList() : NULL ; }
const char*     Opticks::getIdPath() const { return m_resource ? m_resource->getIdPath() : NULL ; }


const char*     Opticks::getIdFold() const { return m_resource ? m_resource->getIdFold() : NULL ; }
const char*     Opticks::getDetectorBase() {    return m_resource ? m_resource->getDetectorBase() : NULL ; }
const char*     Opticks::getMaterialMap() {  return m_resource ? m_resource->getMaterialMap() : NULL ; }
const char*     Opticks::getDAEPath() {   return m_resource ? m_resource->getDAEPath() : NULL ; }
const char*     Opticks::getInstallPrefix() { return m_resource ? m_resource->getInstallPrefix() : NULL ; }

bool             Opticks::SetKey(const char* spec) { return BOpticksKey::SetKey(spec) ; }
BOpticksKey*     Opticks::GetKey() {                 return BOpticksKey::GetKey() ; }
BOpticksKey*     Opticks::getKey() const {           return m_resource->getKey() ; }
const char*      Opticks::getKeySpec() const {       BOpticksKey* key = getKey(); return key ? key->getSpec() : "no-key-spec" ; }

const char*     Opticks::getSrcGDMLPath() const {  return m_resource ? m_resource->getSrcGDMLPath() : NULL ; }
const char*     Opticks::getGDMLPath()    const {  return m_resource ? m_resource->getGDMLPath() : NULL ; }

const char*     Opticks::getOriginGDMLPath() const { return m_origin_gdmlpath ; }

const char*     Opticks::getCurrentGDMLPath() const 
{
    bool is_direct   = isDirect() ;   
    //bool is_embedded = isEmbedded() ;   
    return is_direct ? getOriginGDMLPath() : getSrcGDMLPath() ;
    // GDML path for embedded Opticks (ie direct from Geant4) is within the geocache directory 
}



void Opticks::setIdPathOverride(const char* idpath_tmp) // used for saves into non-standard locations whilst testing
{
    m_resource->setIdPathOverride(idpath_tmp);
}


void Opticks::cleanup()
{
    LOG(LEVEL) << desc() ; 
}


void Opticks::configureF(const char* name, std::vector<float> values)
{
     if(values.empty())
     {   
         printf("Opticks::parameter_set %s no values \n", name);
     }   
     else    
     {   
         float vlast = values.back() ;

         LOG(info) << "Opticks::configureF"
                   << " name " << name 
                   << " vals " << values.size()
                   ;

         for(size_t i=0 ; i < values.size() ; i++ ) printf("%10.3f ", values[i]);
         printf(" : vlast %10.3f \n", vlast );

         //configure(name, vlast);  
     }   
}
 


template <typename T>
void Opticks::set(const char* name, T value)
{
    m_parameters->set<T>(name, value); 
}



/*
template OKCORE_API void Opticks::profile<unsigned>(unsigned);
template OKCORE_API void Opticks::profile<int>(int);
template OKCORE_API void Opticks::profile<char*>(char*);
template OKCORE_API void Opticks::profile<const char*>(const char*);
*/

template OKCORE_API void Opticks::set(const char* name, bool value);
template OKCORE_API void Opticks::set(const char* name, int value);
template OKCORE_API void Opticks::set(const char* name, unsigned int value);
template OKCORE_API void Opticks::set(const char* name, std::string value);
template OKCORE_API void Opticks::set(const char* name, float value);
template OKCORE_API void Opticks::set(const char* name, double  value);
template OKCORE_API void Opticks::set(const char* name, char value);




