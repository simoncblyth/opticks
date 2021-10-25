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
#include "SStr.hh"
#include "SPath.hh"
#include "SRngSpec.hh"


// brap-
#include "BTimeKeeper.hh"
#include "BMeta.hh"
#include "BDynamicDefine.hh"
#include "BOpticksEvent.hh"
#include "BOpticksResource.hh"
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
#include "NPY.hpp"
#include "TorchStepNPY.hpp"
#include "GLMFormat.hpp"
#include "NState.hpp"
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
#include "SensorLib.hh"

#include "FlightPath.hh"
#include "Snap.hh"
#include "Composition.hh"



const char*          Opticks::GEOCACHE_CODE_VERSION_KEY = "GEOCACHE_CODE_VERSION" ; 
const int            Opticks::GEOCACHE_CODE_VERSION = 15 ;  // (incremented when code changes invalidate loading old geocache dirs)   

/**
3: starting point 
4: switch off by default addition of extra global_instance GMergedMesh, 
   as GNodeLib now persists the "all volume" info enabling simplification of GMergedMesh 
5: go live with geometry model change mm0 no longer special, just remainder, GNodeLib name changes, start on triplet identity
6: GVolume::getIdentity quad now packing in more info including triplet_identity and sensorIndex
7: GNodeLib add all_volume_inverse_transforms.npy
8: GGeo/GNodeLib/BMeta/CGDML/Opticks get G4GDMLAux info thru geocache for default genstep targetting configured 
   from within the GDML, opticksaux-dx1 modified with added auxiliary element for lvADE. Used for example by g4ok/G4OKTest   
9: GDMLAux metadata now arranged with lvmeta and usermeta top objects  
10: moved to double precision material and surface properties, with narrowing to float only done at last moment 
    prior to creation of GPU textures 

11: following fix of the inadvertent standardization of raw materials causing GScintillatorLib generated wavelength binning artifacts

12: switch to fine 1nm domain steps as the default for all material/surface properties using Geant4 G4PhysicsVector::Value interpolation 
    to populate all the GProperty in this fine domain
 
13: rejig of GScintillatorLib persisting, now with _ori energy domain properties to facilitate postcache Geant4 testing
    such as for scintillator ICDF creation + plus move to effective multi resolution scintillator/reemission texture

14: following bug fix (some ggeo/GPropertyMap ctor with non-initialized original_domain) old geocache 
    have some probability of missing/corrupted scintillation materials and should be rebuilt  

15: switch to allways positive-izing CSG trees no matter what the height in NTreeProcess::init

**/


const plog::Severity Opticks::LEVEL = PLOG::EnvLevel("Opticks", "DEBUG")  ; 


BPropNames* Opticks::G_MATERIAL_NAMES = NULL ; 



const float Opticks::F_SPEED_OF_LIGHT = 299.792458f ;  // mm/ns

// formerly of GPropertyLib, now booted upstairs
float        Opticks::DOMAIN_LOW  = 60.f ;
float        Opticks::DOMAIN_HIGH = 820.f ;  // has been 810.f for a long time  
float        Opticks::DOMAIN_STEP = 20.f ; 
unsigned     Opticks::DOMAIN_LENGTH = 39  ;


//const char   Opticks::DOMAIN_TYPE = 'C' ; 
const char   Opticks::DOMAIN_TYPE = 'F' ; 

float        Opticks::FINE_DOMAIN_STEP = 1.f ; 
unsigned     Opticks::FINE_DOMAIN_LENGTH = 761  ;
unsigned     Opticks::DomainLength(){ return DOMAIN_TYPE == 'F' ? FINE_DOMAIN_LENGTH : DOMAIN_LENGTH ; }


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
const char* Opticks::OriginGDMLPath()  // static 
{
    return BOpticksResource::GetCachePath("origin.gdml");       
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
Opticks* Opticks::Get()
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
    bool allownokey = isAllowNoKey(); 
    if(allownokey)
    {
        LOG(fatal) << " --allownokey option prevents key checking : this is for debugging of geocache creation " ; 
        return false ; 
    }

    bool key_is_set(false) ;
    key_is_set = BOpticksKey::IsSet() ; 
    if(key_is_set) return true ; 

    BOpticksKey::SetKey(NULL) ;  // use keyspec from OPTICKS_KEY envvar 

    key_is_set = BOpticksKey::IsSet() ; 
    assert( key_is_set == true && "valid geocache and key are required" ); 

    return key_is_set ; 
}

std::string Opticks::getArgLine() const
{
    return m_sargs->getArgLine(); 
}

Opticks::Opticks(int argc, char** argv, const char* argforced )
    :
    m_log(new SLog("Opticks::Opticks","",debug)),
    m_ok(this),
    m_sargs(new SArgs(argc, argv, argforced)),  
    m_geo(nullptr),
    m_argc(m_sargs->argc),
    m_argv(m_sargs->argv),
    m_lastarg(m_argc > 1 ? strdup(m_argv[m_argc-1]) : NULL),
    m_mode(new OpticksMode(this)),
    m_composition(new Composition(this)),
    m_dumpenv(m_sargs->hasArg("--dumpenv")),
    m_allownokey(m_sargs->hasArg("--allownokey")),
    m_envkey(envkey()),
    m_production(m_sargs->hasArg("--production")),
    m_profile(new OpticksProfile()),
    m_profile_enabled(m_sargs->hasArg("--profile")),
    m_photons_per_g4event(0), 

    m_spec(NULL),
    m_nspec(NULL),
    m_resource(NULL),
    m_rsc(NULL),
    m_nogdmlpath(m_sargs->hasArg("--nogdmlpath")),
    m_origin_gdmlpath(NULL),
    m_origin_gdmlpath_kludged(NULL),
    m_origin_geocache_code_version(-1),
    m_state(NULL),
    m_apmtslice(NULL),
    m_apmtmedium(NULL),

    m_exit(false),
    m_compute(false),
    m_geocache(false),
    m_instanced(true),
    m_configured(false),
    m_angular_enabled(false),

    m_cfg(new OpticksCfg<Opticks>("opticks", this,false)),
    m_parameters(new BMeta), 
    m_runtxt(new BTxt),  
    m_cachemeta(new BMeta), 
    m_origin_cachemeta(NULL), 

    m_scene_config(NULL),
    m_lod_config(NULL),
    m_snapconfig(NULL),
    m_flightpath(NULL),
    m_snap(NULL),
    m_detector(NULL),
    m_event_count(0),
    m_domains_configured(false),
    m_time_domain(0.f, 0.f, 0.f, 0.f),
    m_space_domain(0.f, 0.f, 0.f, 0.f),
    m_wavelength_domain(0.f, 0.f, 0.f, 0.f),
    m_settings(0,0,0,0),
    m_run(new OpticksRun(this)),
    m_ana(new OpticksAna(this)),
    m_dbg(new OpticksDbg(this)),
    m_rc(0),
    m_rcmsg(NULL),
    m_tagoffset(0),
    m_verbosity(0),
    m_internal(false),
    m_frame_renderer(NULL),
    m_rngspec(NULL),
    m_sensorlib(NULL),
    m_one_gas_ias(-1), 
    m_outdir(nullptr)
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
    LOG(LEVEL) << m_mode->desc() << " hostname " << SSys::hostname() ; 
    if(IsLegacyGeometryEnabled())
    {
        LOG(fatal) << "OPTICKS_LEGACY_GEOMETRY_ENABLED mode is active " 
                   << " : ie dae src access to geometry, opticksdata  "  
                    ;
    }
    else
    {
        LOG(LEVEL) << " mandatory keyed access to geometry, opticksaux " ; 
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
    m_parameters->add<std::string>("ArgLine", getArgLine() ); 

    std::string switches = OpticksSwitches() ; 
    m_parameters->add<std::string>("OpticksSwitches", switches ); 
    LOG(LEVEL) << "OpticksSwitches:" << switches ; 

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

Composition* Opticks::getComposition() const 
{
    return m_composition ; 
}
















void Opticks::profile(const char* label)
{
    if(!m_profile_enabled) return ; 
    m_profile->stamp(label, m_tagoffset);
   // m_tagoffset is set by Opticks::makeEvent
}

const glm::vec4& Opticks::getLastStamp() const 
{
    return m_profile->getLastStamp() ; 
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
    if(!m_profile_enabled) return ; 
    m_profile->save();
}

void Opticks::postgeocache()
{
   dumpProfile("Opticks::postgeocache", NULL  );  
}

void Opticks::postpropagate()
{
   if(isProduction()) return ;  // --production

   saveProfile();

   //double tcut = 0.0001 ; 
   /*
   double tcut = -1.0 ;  
   dumpProfile("Opticks::postpropagate", NULL, "_OpticksRun::createEvent", tcut  );  // spacwith spacing at start if each evt
   */

   if(isDumpProfile() && m_profile_enabled) 
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


/**
Opticks::Finalize
-------------------

Invoked from G4Opticks::Finalize

**/

void Opticks::Finalize()   // static 
{
    LOG(LEVEL); 
    if(!HasInstance()) return ;  

    Opticks* ok = Instance(); 
    if(ok->isSaveProfile()) ok->saveProfile(); 
}


void Opticks::ana()
{
   m_ana->run();
}
OpticksAna* Opticks::getAna() const 
{
    return m_ana  ; 
}
SensorLib* Opticks::getSensorLib() const 
{
    return m_sensorlib  ; 
}

/**
Opticks::initSensorData
------------------------

Relocated from G4Opticks::setGeometry for generality 

**/
void Opticks::initSensorData(unsigned num_sensors)
{
    LOG(LEVEL) << " num_sensors " << num_sensors ; 
    assert( m_sensorlib == NULL && "not expecting Opticks::initSensorData to be called more than once"); 
    m_sensorlib = new SensorLib ; 
    m_sensorlib->initSensorData(num_sensors); 
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
bool Opticks::isX4BalanceSkip(unsigned lvIdx) const 
{
   return m_dbg->isX4BalanceSkip(lvIdx);
}
bool Opticks::isX4NudgeSkip(unsigned lvIdx) const 
{
   return m_dbg->isX4NudgeSkip(lvIdx);
}
bool Opticks::isX4PointSkip(unsigned lvIdx) const 
{
   return m_dbg->isX4PointSkip(lvIdx);
}




bool Opticks::isCSGSkipLV(unsigned lvIdx) const 
{
   return m_dbg->isCSGSkipLV(lvIdx);
}
bool Opticks::isDeferredCSGSkipLV(unsigned lvIdx) const 
{
   return m_dbg->isDeferredCSGSkipLV(lvIdx);
}
bool Opticks::isSkipSolidIdx(unsigned lvIdx) const  // --skipsolidname
{
    return m_dbg->isSkipSolidIdx(lvIdx); 
}


unsigned Opticks::getNumCSGSkipLV() const 
{
   return m_dbg->getNumCSGSkipLV() ; 
}
unsigned Opticks::getNumDeferredCSGSkipLV() const 
{
   return m_dbg->getNumDeferredCSGSkipLV() ; 
}


unsigned long long Opticks::getEMM() const 
{
   return m_dbg->getEMM();  
}
bool Opticks::isEnabledMergedMesh(unsigned mm) const   // --enabledmergedmesh,e
{
   return m_dbg->isEnabledMergedMesh(mm);
}

const char* Opticks::getEnabledMergedMesh() const    // --enabledmergedmesh,e
{
   return m_dbg->getEnabledMergedMesh();  
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

const std::vector<std::string>& Opticks::getArgList() const 
{
    return m_dbg->getArgList() ;  // --arglist
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




const char* Opticks::getDbgMesh() const 
{
   const std::string& dbgmesh = m_cfg->getDbgMesh();
   return dbgmesh.empty() ? NULL : dbgmesh.c_str() ;
}


/**
Opticks::initResource
-----------------------

Invoked by Opticks::configure.

Instanciates m_resource OpticksResource and its constituent BOpticksResource (m_rsc)
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
    m_rsc = m_resource->getRsc(); 

    const char* detector = BOpticksResource::G4LIVE ; 
    setDetector(detector);

    const char* idpath = m_rsc->getIdPath();
    if( idpath )
    {
        m_parameters->add<std::string>("idpath", idpath); 
    }
    else
    {
        LOG(fatal) << " idpath NULL " ; 
    }

    bool assert_readable = false ;  // false: as many tests use Opticks and do not need the RNG  
    const char* curandstatepath = getCURANDStatePath(assert_readable);

    if(curandstatepath)
    {
        m_parameters->add<std::string>("curandstatepath", curandstatepath); 
    }
    else
    {
        LOG(fatal) << " curandstatepath NULL " ; 
    }

    LOG(LEVEL) << m_rsc->desc()  ;
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


/**
Opticks::hasArg
-----------------

The old hasArg only looked at the actual argv commandline arguments not the 
combination of commandline and extraline (aka argforced) that SArgs::hasArg checks.
As embedded running such as G4Opticks uses the extraline to configure Opticks
it is vital to check with m_sargs.

**/

bool Opticks::hasArg(const char* arg) const 
{
    return m_sargs->hasArg(arg); 
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
const char* Opticks::getRenderCmd() const    // --rendercmd
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


unsigned Opticks::getWayMask() const // --waymask 3
{
    return m_cfg->getWayMask(); 
}
bool Opticks::isWayEnabled() const  // --way
{
    return m_cfg->hasOpt("way") ;
}

bool Opticks::isSaveGPartsEnabled() const  // --savegparts
{
    return m_cfg->hasOpt("savegparts") ;
}

bool Opticks::isGDMLKludge() const  // --gdmlkludge
{
    return m_cfg->hasOpt("gdmlkludge") ;
}

bool Opticks::isFineDomain() const  // --finedomain
{
    return m_cfg->hasOpt("finedomain") ;
}





bool Opticks::isAngularEnabled() const 
{
    return m_angular_enabled ;  
}

/**
Opticks::setAngularEnabled
----------------------------

Canonically invoked from G4Opticks::setSensorAngularEfficiency
immediately after passing the angular array to okc/SensorLib

**/
void Opticks::setAngularEnabled(bool angular_enabled )
{
    LOG(fatal) << " angular_enabled " << angular_enabled  ; 
    m_angular_enabled = angular_enabled ; 
}

bool Opticks::isG4CodeGen() const  // --g4codegen
{
    return m_cfg->hasOpt("g4codegen") ;
}

bool Opticks::isNoSavePPM() const  // --nosaveppm
{
    return m_cfg->hasOpt("nosaveppm") ;
}
bool Opticks::isNoGPU() const  // --nogpu
{
    return m_cfg->hasOpt("nogpu") ;
}




bool Opticks::isNoG4Propagate() const  // --nog4propagate
{
    return m_cfg->hasOpt("nog4propagate") ;
}
bool Opticks::isSaveProfile() const // --saveprofile
{
    return m_cfg->hasOpt("saveprofile") ;
}
bool Opticks::canDeleteGeoCache() const   // --deletegeocache
{
    return m_cfg->hasOpt("deletegeocache") ;
}

bool Opticks::isNoGDMLPath() const   // --nogdmlpath
{
    return m_cfg->hasOpt("nogdmlpath") ;
}


/**
Opticks::isAllowNoKey
-----------------------

As this is needed prior to configure it directly uses
the bool set early in instanciation.

**/

bool Opticks::isAllowNoKey() const   // --allownokey
{
    return m_allownokey ;
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
unsigned Opticks::getSkipAheadStep() const  // --skipaheadstep 1000
{
    return m_cfg->getSkipAheadStep();
}



int Opticks::getRTX() const 
{
    return m_cfg->getRTX();
}
int Opticks::getOneGASIAS() const    // returns value from commandline option --one_gas_ias but may be overriden by setOneGASIAS
{
    return m_one_gas_ias == -1 ? m_cfg->getOneGASIAS() : m_one_gas_ias ; 
}  
void Opticks::setOneGASIAS(int one_gas_ias)
{
    m_one_gas_ias = one_gas_ias ; 
}

int Opticks::getRaygenMode() const  // returns value from commandline option --raygenmode but may be overriden by setRaygenMode
{
    return m_raygenmode == -1 ? m_cfg->getRaygenMode() : m_raygenmode ;
}
void Opticks::setRaygenMode(int raygenmode)
{
    m_raygenmode = raygenmode ; 
}




std::vector<unsigned>& Opticks::getSolidSelection()
{
    return m_solid_selection ; 
}
const std::vector<unsigned>& Opticks::getSolidSelection() const 
{
    return m_solid_selection ; 
}



const char* Opticks::getSolidLabel() const 
{
    const std::string& solid_label = m_cfg->getSolidLabel(); 
    return solid_label.empty() ? nullptr :  solid_label.c_str() ; 
}



int Opticks::getRenderLoopLimit() const 
{
    return m_cfg->getRenderLoopLimit();
}
int Opticks::getAnnoLineHeight() const 
{
    return m_cfg->getAnnoLineHeight();
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
void Opticks::createEvent(NPY<float>* gensteps, char ctrl)
{
    m_run->createEvent(gensteps, ctrl );
}


// better to phase out this alternative to the above from gensteps 
void Opticks::createEvent(unsigned tagoffset, char ctrl)
{
    m_run->createEvent(tagoffset, ctrl );
}



void Opticks::saveEvent(char ctrl)
{
    m_run->saveEvent(ctrl); 
}
void Opticks::resetEvent(char ctrl)
{
    m_run->resetEvent(ctrl);
}

OpticksEvent* Opticks::getEvent(char ctrl) const 
{
    return m_run->getEvent(ctrl)  ; 
}

OpticksEvent* Opticks::getEvent() const 
{
    return m_run->getEvent()  ; 
}
OpticksEvent* Opticks::getG4Event() const 
{
    return m_run->getG4Event()  ; 
}




OpticksProfile* Opticks::getProfile() const 
{
    return m_profile ; 
}


BMeta*       Opticks::getParameters() const 
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

/**
Opticks::isKeySource
----------------------

Name of current executable matches that of the creator of the geocache.
BUT what about the first run of geocache-create ?

**/

bool Opticks::isKeySource() const 
{
    return m_rsc->isKeySource();  
}

/**
Opticks::isKeyLive() 
---------------------

true when creating geocache from live Geant4 geometry tree

**/

bool Opticks::isKeyLive() const 
{
    return m_rsc->isKeyLive();  
}



NState* Opticks::getState() const 
{
    return m_state  ; 
}

const char* Opticks::getLastArg()
{
   return m_lastarg ; 
}





bool Opticks::isRemoteSession() const 
{
    return SSys::IsRemoteSession();
}
bool Opticks::isCompute() const 
{
    return m_mode->isCompute() ;
}
bool Opticks::isInterop() const 
{
    return m_mode->isInterop() ;
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
bool Opticks::isDumpHiy() const  // --dumphiy
{
   return m_cfg->hasOpt("dumphiy");
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





bool Opticks::getSaveDefault() const  // --save is trumped by --nosave 
{
    bool is_nosave = m_cfg->hasOpt("nosave");  
    bool is_save = m_cfg->hasOpt("save");  
    return is_nosave ? false : is_save  ;   
}
void Opticks::setSave(bool save)  // override the default from config 
{
    m_save = save ; 
}
bool Opticks::isSave() const   
{
    return m_save ; 
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

bool Opticks::isDbgRec() const  // --dbgrec 
{
    return m_cfg->hasOpt("dbgrec") ;
}
bool Opticks::isDbgZero() const  // --dbgzero
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


unsigned long long Opticks::getDbgSeqmat()  // --dbgseqmat 0x...
{
    const std::string& seqmat = m_cfg->getDbgSeqmat();
    return BHex<unsigned long long>::hex_lexical_cast( seqmat.c_str() );
}
unsigned long long Opticks::getDbgSeqhis()  // --dbgseqhis 0x...
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
    return m_rsc ? m_rsc->getRuncacheDir() : NULL ; 
}
const char* Opticks::getOptiXCacheDirDefault() const 
{
    return m_rsc ? m_rsc->getOptiXCacheDirDefault() : NULL ; 
}







const char* Opticks::getGLTFPath() const { return m_rsc->getGLTFPath() ; }
const char* Opticks::getG4CodeGenDir() const { return m_rsc->getG4CodeGenDir() ; }
const char* Opticks::getCacheMetaPath() const { return m_rsc->getCacheMetaPath() ; } 
const char* Opticks::getGDMLAuxMetaPath() const { return m_rsc->getGDMLAuxMetaPath() ; } 
const char* Opticks::getRunCommentPath() const { return m_rsc->getRunCommentPath() ; } 



int  Opticks::getGLTFTarget() const 
{
    return m_cfg->getGLTFTarget(); 
}
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

void Opticks::appendCacheMeta(const char* key, BMeta* obj)
{
    m_cachemeta->setObj(key, obj); 
}

/**
Opticks::has_arg
-----------------

This works prior to Opticks::config via the args collected 
into PLOG::instance by OPTICKS_LOG 

**/
bool Opticks::has_arg(const char* arg) const 
{
    return PLOG::instance ? PLOG::instance->has_arg(arg) : false ;
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
    std::string cwd = BFile::CWD(); 

    m_runtxt->addLine(GEOCACHE_CODE_VERSION_KEY);
    m_runtxt->addLine(GEOCACHE_CODE_VERSION); 
    m_runtxt->addLine("rundate") ;  
    m_runtxt->addLine( rundate ) ;  
    m_runtxt->addLine("runstamp" ) ;  
    m_runtxt->addValue( runstamp);  
    m_runtxt->addLine("cwd" ) ;  
    m_runtxt->addLine( cwd ) ;  
    m_runtxt->addLine("argline" ) ;  
    m_runtxt->addLine( argline) ;  

    m_cachemeta->set<int>(GEOCACHE_CODE_VERSION_KEY, GEOCACHE_CODE_VERSION ); 
    m_cachemeta->set<std::string>("location", "Opticks::updateCacheMeta"); 
    m_cachemeta->set<std::string>("cwd", cwd ); 
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
    bool is_key_live = isKeyLive(); 
    if( is_key_live )
    {
        LOG(LEVEL) << " is_key_live prevents loading of cache metadata and code version matching " ; 
    }
    else 
    {
        loadOriginCacheMeta_();
    }
}

const char* Opticks::getCacheMetaGDMLPath_(const BMeta* origin_cachemeta ) const 
{
    std::string gdmlpath = ExtractCacheMetaGDMLPath(origin_cachemeta); 
    LOG(LEVEL) << "ExtractCacheMetaGDMLPath " << gdmlpath ; 

    const char* cachemeta_gdmlpath = gdmlpath.empty() ? NULL : strdup(gdmlpath.c_str());

    if(cachemeta_gdmlpath == NULL)
    {
        LOG(LEVEL) << "argline that creates cachemetapath does not include \"--gdmlpath /path/to/geometry.gdml\" " ; 
        LOG(LEVEL) << "FAILED to extract gdmlpath from the geocache creating commandline persisted in cachemetapath " ; 
    }
    return cachemeta_gdmlpath ; 
} 



void Opticks::loadOriginCacheMeta_() 
{
    LOG(LEVEL) << "[" ; 

    const char* cachemetapath = getCacheMetaPath();
    LOG(LEVEL) << " cachemetapath " << cachemetapath ; 
    m_origin_cachemeta = BMeta::Load(cachemetapath); 
    //m_origin_cachemeta->dump("Opticks::loadOriginCacheMeta_"); 

    const char* cachemeta_gdmlpath = getCacheMetaGDMLPath_( m_origin_cachemeta ); 
    const char* origin_gdmlpath = OriginGDMLPath() ; 
    m_origin_gdmlpath = cachemeta_gdmlpath ? cachemeta_gdmlpath : origin_gdmlpath ; 
    assert( m_origin_gdmlpath );  // it is null with OPTICKS_KEY for geocache from live running 

    m_origin_gdmlpath_kludged = SStr::ReplaceEnd(m_origin_gdmlpath, ".gdml", "_CGDMLKludge.gdml") ; 

    LOG(LEVEL) << " m_origin_gdmlpath " << m_origin_gdmlpath ;  
    LOG(LEVEL) << " m_origin_gdmlpath_kludged " << m_origin_gdmlpath_kludged ;  
 

    m_origin_geocache_code_version = m_origin_cachemeta->get<int>(GEOCACHE_CODE_VERSION_KEY, "0" );  

    bool geocache_code_version_match = m_origin_geocache_code_version == Opticks::GEOCACHE_CODE_VERSION ; 
    bool geocache_code_version_pass = geocache_code_version_match ; 

    if(!geocache_code_version_pass)
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
        LOG(LEVEL) << "(pass) " << GEOCACHE_CODE_VERSION_KEY << " " << m_origin_geocache_code_version  ; 
    }
    assert( geocache_code_version_pass ); 
    LOG(LEVEL) << "]" ; 
}


BMeta* Opticks::getOriginCacheMeta(const char* obj) const 
{
    return m_origin_cachemeta ? m_origin_cachemeta->getObj(obj) : NULL ; 
}





const BMeta* Opticks::getGDMLAuxMeta() const 
{
    return m_rsc->getGDMLAuxMeta() ; 
}

void Opticks::findGDMLAuxMetaEntries(std::vector<BMeta*>& entries, const char* k, const char* v ) const 
{
    m_rsc->findGDMLAuxMetaEntries(entries, k, v );
}

/**
Opticks::findGDMLAuxValues
----------------------------

For metadata entries matching k,v collect q values into the vector.  
**/

void Opticks::findGDMLAuxValues(std::vector<std::string>& values, const char* k, const char* v, const char* q) const 
{
    m_rsc->findGDMLAuxValues(values, k, v, q ); 
}

/**
Opticks::getGDMLAuxTargetLVNames
----------------------------------

Consults the persisted GDMLAux metadata looking for entries with (k,v) pair ("label","target").
For any such entries the "lvname" property is accesses and added to the lvnames vector.

**/

unsigned Opticks::getGDMLAuxTargetLVNames(std::vector<std::string>& lvnames) const 
{
    return m_rsc->getGDMLAuxTargetLVNames(lvnames); 
}

/**
Opticks::getGDMLAuxTargetLVName
---------------------------------

Returns the first lvname or NULL

**/

const char* Opticks::getGDMLAuxTargetLVName() const 
{
    return m_rsc->getGDMLAuxTargetLVName() ; 
}



/**
Opticks::ExtractCacheMetaGDMLPath
------------------------------------

TODO: avoid having to fish around in the geocache argline to get the gdmlpath in direct mode

Going via the tokpath enables sharing of geocaches across different installs. 

**/

std::string Opticks::ExtractCacheMetaGDMLPath(const BMeta* meta)  // static
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

    if(tokpath.empty())
    {
        LOG(LEVEL)
            << " FAILED TO EXTRACT ORIGIN GDMLPATH FROM METADATA argline "  
            << "\n argline " << argline
            ;
    
        const char* fallback_gdmlpath = sa.get_first_arg_ending_with(".gdml", NULL ) ;
        if( fallback_gdmlpath ) 
        {
            LOG(info) 
                << " HOWEVER found a gdmlpath on argline, try using that "
                << " fallback_gdmlpath " << fallback_gdmlpath 
                ;
               
            tokpath = fallback_gdmlpath ; 
        }
    }
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

const char* Opticks::getPVName() const  
{
    const std::string& pvname = m_cfg->getPVName() ;  // --pvname
    return pvname.empty() ? NULL : pvname.c_str() ;
}
const char* Opticks::getBoundary() const  
{
    const std::string& boundary = m_cfg->getBoundary() ;  // --boundary
    return boundary.empty() ? NULL : boundary.c_str() ;
}
const char* Opticks::getMaterial() const 
{
    const std::string& material = m_cfg->getMaterial() ;  // --material
    return material.empty() ? NULL : material.c_str() ;
}


bool Opticks::isLarge() const 
{
    return m_cfg->hasOpt("large") ;  // --large
}
bool Opticks::isMedium() const 
{
    return m_cfg->hasOpt("medium") ;  // --medium
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

const char* Opticks::getSnapConfigString() const 
{
    return m_cfg->getSnapConfig().c_str() ; 
}


const char* Opticks::getFlightPathDir() const 
{
    const std::string& dir = m_cfg->getFlightPathDir();
    return dir.empty() ? NULL : dir.c_str() ;
}

const char* Opticks::getFlightConfig() const 
{
    const std::string& flight_config = m_cfg->getFlightConfig() ; 
    return flight_config.empty() ? NULL : flight_config.c_str() ;
}

const char* Opticks::getFlightOutDir() const   // --flightoutdir
{
    const std::string& flightoutdir = m_cfg->getFlightOutDir() ; 
    return flightoutdir.empty() ? NULL : flightoutdir.c_str() ;
}

const char* Opticks::getSnapOutDir() const    // --snapoutdir
{
    const std::string& snapoutdir = m_cfg->getSnapOutDir() ; 
    return snapoutdir.empty() ? NULL : snapoutdir.c_str() ;
}


/**
Opticks::getOutDir
--------------------

Outdir from m_cfg OpticksCfg depends on OPTICKS_OUTDIR envvar defaulting to $TMP/outdir 
which can be overridden by "--outdir" commandline option.

Alternatively a setting at code level with Opticks::setOutDir overrides the 
envvars and commandline options. But that is problematic
for executables that are also used from scripts.  

Typically envvars and commandline are used from bash scripts.
Setting the envvar in code prior to configuring Opticks
with overwrite:false is convenient for use from bare executables.
As this avoids the need to use scripts but does not remove the 
flexibility of control from scripts.::

   bool overwrite = false ; 
   SSys::setenvvar("OPTICKS_OUTDIR", "$TMP/CSGOptiX", overwrite ); 

**/

const char* Opticks::getOutDir() const    // --outdir
{
    const std::string& outdir = m_cfg->getOutDir() ; 
    const char* default_outdir = outdir.empty() ? NULL : outdir.c_str() ;
    const char* dir = m_outdir ? m_outdir : default_outdir ; 
    int create_dirs = 0 ;  // 0:nop
    return SPath::Resolve(dir, create_dirs); 
}

void Opticks::setOutDir( const char* outdir_ )  // overrides --outdir 
{
    int create_dirs = 2 ; // 2:dirpath
    const char* outdir = SPath::Resolve(outdir_, create_dirs ); 
    if(outdir) m_outdir = strdup(outdir); 
}

const char* Opticks::getNamePrefix() const    // --nameprefix
{
    const std::string& nameprefix = m_cfg->getNamePrefix() ; 
    return nameprefix.empty() ? NULL : nameprefix.c_str() ;
}


NSnapConfig* Opticks::getSnapConfig() // lazy cannot const 
{
    if(m_snapconfig == NULL)
    {
        m_snapconfig = new NSnapConfig(getSnapConfigString());
    }
    return m_snapconfig ; 
}

unsigned Opticks::getSnapSteps() 
{
    NSnapConfig* snapconfig = getSnapConfig() ;  
    return snapconfig->steps ; 
}

void Opticks::getSnapEyes(std::vector<glm::vec3>& eyes)
{
    NSnapConfig* snapconfig = getSnapConfig() ;  
    m_composition->eye_sequence(eyes, snapconfig );
}

const char* Opticks::getSnapPath(int index) 
{
    const char* snapoutdir = m_ok->getSnapOutDir() ;   // --snapoutdir
    const char* nameprefix = m_ok->getNamePrefix() ;   // --nameprefix
    NSnapConfig* snapconfig = getSnapConfig() ;  
    const char* path = snapconfig->getSnapPath(snapoutdir, index, nameprefix );      
    return path ; 
}

Snap* Opticks::getSnap(SRenderer* renderer)
{
    if(m_snap == nullptr)
    {
        NSnapConfig* config = getSnapConfig() ;  
        m_snap = new Snap(this, renderer, config ); 
    }
    return m_snap ; 
}


/**
Opticks::getOutPath namestem ext index 
------------------------------------------

::
 
    <outdir>/<nameprefix><namestem><index><ext>

outdir 
    default evar:OUTDIR that is overridden by --outdir option 
nameprefix
    --nameprefix option
namestem
    method argument defauting to "output"
index 
    method argument with -1 default
    when the index is negative the field is skipped 
    otherwise the integer is %0.5d formatted, eg 00000, 00001
ext
    file extension eg .jpg .npy 


The returned string is strdup-ed so it should be free((void*)str) 
after use if there are lots of them.

**/

const char* Opticks::getOutPath(const char* namestem, const char* ext, int index) const 
{
    const char* outdir = getOutDir(); 
    const char* nameprefix = getNamePrefix(); 

    std::stringstream ss ; 

    if(outdir)     ss << outdir << "/" ; 
    if(nameprefix) ss << nameprefix ; 
    if(namestem)   ss << namestem ; 
    if(index > -1 ) ss << std::setfill('0') << std::setw(5) << index ; 
    if(ext)        ss << ext ; 

    std::string s = ss.str(); 
    int create_dirs = 0 ; // 0:nop
    const char* outpath = SPath::Resolve(s.c_str(), create_dirs) ;

    LOG(info)
       << " outdir " << outdir
       << " nameprefix " << nameprefix 
       << " namestem " << namestem 
       << " index " << index
       << " ext " << ext 
       << " outpath " << outpath 
       ;

    return outpath ; 
}

int Opticks::ExtractIndex(const char* path)  // static 
{
    return SStr::ExtractInt(path, -9, 5, -1 );
}



/**
Opticks::getFlightPath
------------------------

NB further flightpath hookup is required hooking it into Composition and setting m_ctrl, 
see OpticksHub::::configureFlightPath

**/

FlightPath* Opticks::getFlightPath()   // lazy cannot be const 
{
    if(m_flightpath == NULL)
    {
        const char* dir = getFlightPathDir() ; // huh not used?
        const char* config = getFlightConfig(); 
        const char* outdir = getOutDir() ;
        const char* nameprefix = getNamePrefix(); 

        float scale = m_cfg->getFlightPathScale() ;

        LOG(LEVEL) 
             << " Creating flightpath from file " 
             << " --flightconfig " << config 
             << " --outdir " << outdir
             << " --nameprefix " << nameprefix 
             << " --flightpathdir " << dir  
             << " --flightpathscale " << scale 
             ;   


        FlightPath* fp = new FlightPath(this, config, outdir, nameprefix) ;
        fp->setScale(scale) ; 
        m_composition->setFlightPath(fp);

        m_flightpath = fp ; 
    }
    return m_flightpath ; 
}


/**
Opticks::getContextAnnotation
------------------------------

Context annotation appears at the top line of rendered images.

**/

std::string Opticks::getContextAnnotation() const 
{

    std::stringstream ss ; 

    if(hasArg("--flightconfig"))
    {
        ss << " --flightconfig " << getFlightConfig() ; 
    }
    else
    {
        glm::vec4 eye = m_composition->getModelEye(); 
        std::string s_eye = gformat(eye); 
        ss << " eye: " ; 
        ss << s_eye ;  
    }
    std::string s = ss.str(); 
    return s ; 
}


/**
Opticks::getFrameAnnotation
------------------------------

Frame annotation appears at the bottom line of rendered images.

**/

std::string Opticks::getFrameAnnotation(unsigned frame, unsigned num_frame, double dt ) const 
{
    const char* targetpvn = getTargetPVN(); 
    const char* emm = getEnabledMergedMesh() ;  
    std::stringstream ss ; 
    ss 
        << std::setw(5) << frame << "/" << num_frame
        << " dt " << std::setw(10) << std::fixed << std::setprecision(4) << dt  
        << " | "
        ;

    if(targetpvn) ss << " --targetpvn " << targetpvn ;
    if(emm)       ss << " -e " <<  emm ; 
    std::string s = ss.str(); 
    return s ; 
}




const char* Opticks::getSnapOverridePrefix() const   // --snapoverrideprefix
{
    const std::string& snapoverrideprefix = m_cfg->getSnapOverridePrefix() ; 
    return snapoverrideprefix.empty() ? nullptr : snapoverrideprefix.c_str() ; 
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




int  Opticks::getDomainTarget() const  // --domaintarget, default sensitive to OPTICKS_DOMAIN_TARGET envvar  
{
    return m_cfg->getDomainTarget(); 
}
int  Opticks::getGenstepTarget() const  // --gensteptarget, default sensitive to OPTICKS_GENSTEP_TARGET envvar 
{
    return m_cfg->getGenstepTarget(); 
}
int  Opticks::getTarget() const   // --target,  default sensitive to OPTICKS_TARGET envvar   
{
    return m_cfg->getTarget(); 
}
const char* Opticks::getTargetPVN() const  
{
    const std::string& targetpvn = m_cfg->getTargetPVN() ;  // --targetpvn   OPTICKS_TARGETPVN
    return targetpvn.empty() ? NULL : targetpvn.c_str() ;
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

Invoked from Opticks::postconfigure after commandline parse and initResource.
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

    const char* resource_pfx = m_rsc->getEventPfx() ; 
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

   if(isNoGDMLPath() != m_nogdmlpath)
   {
        LOG(fatal) << " INCONSISTENCY "
                   << " isNoGDMLPath() " << isNoGDMLPath()
                   << " m_nogdmlpath   " << m_nogdmlpath
                   ;
      
        setExit(true);
   }
}

bool Opticks::isConfigured() const
{
    return m_configured ; 
}
void Opticks::configure()
{
    LOG(LEVEL) << "[" ; 

    if(m_configured) return ; 
    m_configured = true ; 

    dumpArgs("Opticks::configure");  

    LOG(LEVEL) <<  "m_argc " << m_argc ; 
    for(int i=0 ; i < m_argc ; i++ ) LOG(LEVEL) << " m_argv[" << i << "] " << m_argv[i] << " " ; 

    m_cfg->commandline(m_argc, m_argv);   

    postconfigure(); 

    LOG(LEVEL) << "]" ; 
}


/**
Opticks::getCVD
-----------------

Default cvdcfg is a blank string

**/

const char* Opticks::getCVD() const 
{
    const std::string& cvdcfg = m_cfg->getCVD();  
    const char* cvd = cvdcfg.empty() ? NULL : cvdcfg.c_str() ; 
    return cvd ; 
}

const char* Opticks::getDefaultCVD() const 
{
    const char* dk = "OPTICKS_DEFAULT_INTEROP_CVD" ; 
    const char* dcvd = SSys::getenvvar(dk) ;  
    return dcvd ; 
}

const char* Opticks::getUsedCVD() const 
{
    const char* cvd = getCVD(); 
    const char* dcvd = getDefaultCVD(); 
    const char* ucvd =  cvd == NULL && isInterop() && dcvd != NULL ? dcvd : cvd ; 

    LOG(LEVEL) 
        << " cvd " << cvd
        << " dcvd " << dcvd
        << " isInterop " << isInterop()
        << " ucvd " << ucvd
        ;

    if( cvd == NULL && isInterop() && dcvd != NULL )
    {
        LOG(error) << " --interop mode with no cvd specified, adopting OPTICKS_DEFAULT_INTEROP_CVD hinted by envvar [" << dcvd << "]" ;   
        ucvd = dcvd ;   
    }
    return ucvd ; 
}



void Opticks::postconfigure()
{
    LOG(LEVEL) << "[" ; 

    checkOptionValidity();

    postconfigureCVD(); 

    postconfigureSave(); 
    postconfigureSize(); 
    postconfigurePosition(); 
    postconfigureComposition(); 


    initResource();  

    updateCacheMeta(); 

    if(isDirect())
    {
        loadOriginCacheMeta();  // sets m_origin_gdmlpath
    }

    defineEventSpec();  

    postconfigureState();   // must be after defineEventSpec

    postconfigureGeometryHandling();


    m_photons_per_g4event = m_cfg->getNumPhotonsPerG4Event();

    m_dbg->postconfigure();

    m_verbosity = m_cfg->getVerbosity(); 

    if(hasOpt("dumpenv")) BEnv::dumpEnvironment("Opticks::postconfigure --dumpenv", "G4,OPTICKS,DAE,IDPATH") ; 
    LOG(LEVEL) << "]" ; 
}





/**
Opticks::postconfigureCVD
---------------------------

When "--cvd" option is on the commandline this internally sets 
the CUDA_VISIBLE_DEVICES envvar to the string argument provided.
For example::
 
   --cvd 0 
   --cvd 1
   --cvd 0,1,2,3

   --cvd -   # '-' is treated as a special token representing an empty string 
             # which easier to handle than an actual empty string 

In interop mode on multi-GPU workstations it is often necessary 
to set the --cvd to match the GPU that is driving the monitor
to avoid failures. To automate that the envvar OPTICKS_DEFAULT_INTEROP_CVD 
is consulted when no --cvd option is provides, acting as a default value.

**/

void Opticks::postconfigureCVD()
{
    const char* ucvd = getUsedCVD() ;  
    if(ucvd)
    { 
        const char* ek = "CUDA_VISIBLE_DEVICES" ; 
        LOG(LEVEL) << " setting " << ek << " envvar internally to " << ucvd ; 
        char special_empty_token = '-' ;   // when ucvd is "-" this will replace it with an empty string
        SSys::setenvvar(ek, ucvd, true, special_empty_token );    // Opticks::configure setting CUDA_VISIBLE_DEVICES

        const char* chk = SSys::getenvvar(ek); 
        LOG(error) << " --cvd [" << ucvd << "] option internally sets " << ek << " [" << chk << "]" ; 
    }
}


void Opticks::postconfigureSave()
{
    bool save_default = getSaveDefault(); 
    setSave(save_default); 
}


/**
Opticks::postconfigureSize
----------------------------

OpticksCfg default is "" so without options m_size becomes::

   Linux: 1920,1080,1,0
   Apple: 2880,1704,2,0    # ,2 relates to apple retina pixels-dots   

**/

void Opticks::postconfigureSize()
{
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
        //m_size = glm::uvec4(2880,1704,2,0) ;  // 1800-44-44px native height of menubar  
        m_size = glm::uvec4(1920,1080,1,0) ;
#else
        m_size = glm::uvec4(1920,1080,1,0) ;
#endif
    }


    


}

const glm::uvec4& Opticks::getSize() const 
{
    return m_size ; 
}
unsigned Opticks::getWidth() const 
{
    return m_size.x ; 
}
unsigned Opticks::getHeight() const 
{
    return m_size.y ; 
}
unsigned Opticks::getDepth() const 
{
    return 1u ; 
}




void Opticks::postconfigurePosition()
{
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
}


void Opticks::postconfigureComposition()
{
    assert( isConfigured() );  

    glm::uvec4 size = getSize();
    glm::uvec4 position = getPosition() ;

    LOG(LEVEL) 
        << " size " << gformat(size)
        << " position " << gformat(position)
        ;   

    m_composition->setSize( size );
    m_composition->setFramePosition( position );

    unsigned cameratype = getCameraType(); 
    m_composition->setCameraType( cameratype );  
}








void Opticks::postconfigureState()
{
    const char* type = "State" ; 
    const std::string& stag = m_cfg->getStateTag();
    const char* subtype = stag.empty() ? NULL : stag.c_str() ; 

    std::string prefdir = getPreferenceDir(type, subtype);  

    LOG(LEVEL) << " prefdir " << prefdir ; 

    // Below "state" is a placeholder name of the current state that never gets persisted, 
    // names like 001 002 are used for persisted states : ie the .ini files within the prefdir

    m_state = new NState(prefdir.c_str(), "state")  ;

}


void Opticks::postconfigureGeometryHandling()
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

bool Opticks::isGPartsTransformOffset() const 
{
    return m_cfg->hasOpt("gparts_transform_offset") ;  
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
              << " mode " << m_mode->desc()
              ; 

    m_resource->Summary(msg);

    std::cout
        << std::setw(40) << " isInternal "
        << std::setw(40) << isInternal()
        << std::endl
        << std::setw(40) << " Verbosity "
        << std::setw(40) << getVerbosity()
        << std::endl
        ;

    LOG(info) << msg << " DONE" ; 
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
    return key->export_(); 
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



bool Opticks::hasKey() const { return m_rsc->hasKey() ; }
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
bool Opticks::isNoPropagate() const { return hasOpt("nopropagate") ; } 


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
    OpticksEvent* evt = OpticksEvent::Make(ok ? m_spec : m_nspec, tagoffset);

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

    OpticksEvent* evt = OpticksEvent::Make(ok ? m_spec : m_nspec, tagoffset);

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

    BMeta*       parameters = evt->getParameters();
    parameters->add<unsigned int>("RngMax",    rng_max );
    parameters->add<unsigned int>("BounceMax", bounce_max );
    parameters->add<unsigned int>("RecordMax", record_max );

    parameters->add<std::string>("mode", m_mode->desc());
    parameters->add<std::string>("cmdline", m_cfg->getCommandLine() );
    parameters->add<std::string>("ArgLine", getArgLine() ); 

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

    LOG(LEVEL)
              << " enable : --torch (the default) "
              << " configure : --torchconfig [" << ( cfg ? cfg : "NULL" ) << "]" 
              << " dump details : --torchdbg " 
              ;

    TorchStepNPY* ts = new TorchStepNPY(OpticksGenstep_TORCH, cfg ); // see notes/issues/G4StepNPY_gencode_assert.rst

    unsigned int photons_per_g4event = m_cfg->getNumPhotonsPerG4Event() ;  // only used for cfg4-

    ts->setNumPhotonsPerG4Event(photons_per_g4event);

    int frameIdx = ts->getFrameIndex(); 
    assert( frameIdx == 0 );  
    
    glm::mat4 identity(1.f); 
    LOG(info) << "[ts.setFrameTransform" ; 

    ts->setFrameTransform(identity);

    ts->addStep();     

    return ts ; 
}


unsigned Opticks::getNumPhotonsPerG4Event() const { return m_cfg->getNumPhotonsPerG4Event() ; }
unsigned Opticks::getManagerMode() const {          return m_cfg->getManagerMode() ; }



const char*        Opticks::getRNGDir() const  { return m_rsc->getRNGDir(); } 
unsigned           Opticks::getRngMax()   const {  return m_cfg->getRngMax() ; }
unsigned long long Opticks::getRngSeed()  const {  return m_cfg->getRngSeed() ; }
unsigned long long Opticks::getRngOffset() const { return m_cfg->getRngOffset() ; }
const char*        Opticks::getCURANDStatePath(bool assert_readable) const 
{
    const char* rngdir = getRNGDir(); 
    unsigned    rngmax = getRngMax();
    unsigned    rngseed = getRngSeed();
    unsigned    rngoffset = getRngOffset();
    const char* path = SRngSpec::CURANDStatePath(rngdir, rngmax, rngseed, rngoffset);

    bool readable = SPath::IsReadable(path); 
    if(!readable)
    {
        LOG(fatal) 
           << " CURANDStatePath IS NOT READABLE " 
           << " INVALID RNG config : change options --rngmax/--rngseed/--rngoffset "  
           << " path " << path 
           << " rngdir " << rngdir
           << " rngmax " << rngmax
           << " rngseed " << rngseed
           << " rngoffset " << rngoffset
           ;
    }
    if(assert_readable)
    {
        assert(readable);
    }
    return path ; 
}



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

bool Opticks::isValid() {   return m_resource->isValid(); }


std::string Opticks::getFlightInputDir() const 
{
    const char* type = "flight" ; 
    const char* udet = nullptr ; 
    const char* subtype = nullptr ; 
    return m_resource->getPreferenceDir(type, udet, subtype);
}

std::string Opticks::getFlightInputPath(const char* name) const 
{
    std::stringstream ss ; 
    ss << getFlightInputDir() ; 
    ss << "/" ; 
    ss << name ; 
    ss << ".npy" ; 
    std::string s = ss.str(); 
    return s ; 
}


std::string Opticks::getPreferenceDir(const char* type, const char* subtype) const 
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





std::string Opticks::formCacheRelativePath(const char* path) { return m_rsc->formCacheRelativePath(path); }

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



const char*     Opticks::getKeyDir() const { return m_rsc ? m_rsc->getIdPath() : NULL ; }
const char*     Opticks::getIdPath() const { return m_rsc ? m_rsc->getIdPath() : NULL ; }


const char*     Opticks::getIdFold() const { return m_rsc ? m_rsc->getIdFold() : NULL ; }


const char*     Opticks::getInstallPrefix() { return m_rsc ? m_rsc->getInstallPrefix() : NULL ; }

bool             Opticks::SetKey(const char* spec) { return BOpticksKey::SetKey(spec) ; }   // static
BOpticksKey*     Opticks::GetKey() {                 return BOpticksKey::GetKey() ; }       // static

BOpticksKey*     Opticks::getKey() const {           return m_rsc->getKey() ; }
const char*      Opticks::getKeySpec() const {       BOpticksKey* key = getKey(); return key ? key->getSpec() : "no-key-spec" ; }

const char*     Opticks::getSrcGDMLPath() const {  return m_rsc ? m_rsc->getSrcGDMLPath() : NULL ; }
const char*     Opticks::getGDMLPath()    const {  return m_rsc ? m_rsc->getGDMLPath() : NULL ; }

const char*     Opticks::getOriginGDMLPath() const {        return m_origin_gdmlpath ; }
const char*     Opticks::getOriginGDMLPathKludged() const { return m_origin_gdmlpath_kludged ; }


/**
Opticks::getCurrentGDMLPath
----------------------------

Returns the absolute path to the origin gdml file within the geocache directory.  
If a kludged gdml file eg origin_CGDMLKludge.gdml exists then the path to the 
kludged file is returned in preference to any origin.gdml

Too create the kludged file it is necessary to use G4Opticks::translateGeometry
with the "--gdmlkludge" option enabled.

**/

const char* Opticks::getCurrentGDMLPath() const 
{
    bool is_direct   = isDirect() ;   
    assert( is_direct ); 
    //return is_direct ? getOriginGDMLPath() : getSrcGDMLPath() ;

    const char* origin = getOriginGDMLPath(); 
    const char* kludge = getOriginGDMLPathKludged() ; 

    bool origin_exists = origin ? BFile::ExistsFile(origin) : false ;
    bool kludge_exists = kludge ? BFile::ExistsFile(kludge) : false ;

    const char* path = nullptr ; 
    if( kludge_exists )
    {
        path = kludge ;
        LOG(LEVEL) << "returning kludge path " << path ;  
    }
    else if( origin_exists )
    {
        path = origin ; 
        LOG(LEVEL) << "returning origin path " << path ;  
    }
    else
    {
        LOG(fatal) << "RETURNING NULL PATH" ; 
    }
    return path ; 
}


void Opticks::setIdPathOverride(const char* idpath_tmp) // used for saves into non-standard locations whilst testing
{
    m_rsc->setIdPathOverride(idpath_tmp);
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


/**
Opticks::setGeo
------------------

*precache*
    Invoked from GGeo::prepare after base objects have been 
    collected but before instances are formed.

*postcache*
     


**/
void Opticks::setGeo(const SGeo* geo)
{
    LOG(LEVEL) ; 
    m_geo = geo ; 
    m_dbg->postgeometry(); 

}
const SGeo* Opticks::getGeo() const 
{
    return m_geo ; 
} 


template <typename T>
void Opticks::set(const char* name, T value)
{
    m_parameters->set<T>(name, value); 
}


template OKCORE_API void Opticks::set(const char* name, bool value);
template OKCORE_API void Opticks::set(const char* name, int value);
template OKCORE_API void Opticks::set(const char* name, unsigned int value);
template OKCORE_API void Opticks::set(const char* name, std::string value);
template OKCORE_API void Opticks::set(const char* name, float value);
template OKCORE_API void Opticks::set(const char* name, double  value);
template OKCORE_API void Opticks::set(const char* name, char value);




