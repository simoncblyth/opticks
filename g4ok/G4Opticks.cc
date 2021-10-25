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

#include <sstream>
#include <iostream>
#include <cstring>
#include <set>

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4PVPlacement.hh"

#include "SSys.hh"
#include "SStr.hh"
#include "BOpticksResource.hh"
#include "BOpticksKey.hh"
#include "NLookup.hpp"
#include "NPY.hpp"
#include "TorchStepNPY.hpp"

#include "OpticksSwitches.h"

#include "CTraverser.hh"
#include "CMaterialTable.hh"
#include "CGenstepCollector.hh"
#include "CPrimaryCollector.hh"
#include "CPhotonCollector.hh"
#include "C4PhotonCollector.hh"
#include "CAlignEngine.hh"
#include "CGDML.hh"
#include "CGDMLKludge.hh"
#include "C4FPEDetection.hh"

#include "G4OpticksHit.hh"
#include "G4Opticks.hh"
#include "G4OpticksRecorder.hh"

#include "OpticksPhoton.h"
#include "OpticksGenstep.h"
#include "OpticksGenstep.hh"
#include "OpticksProfile.hh"
#include "SensorLib.hh"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpMgr.hh"

#include "GGeo.hh"
#include "GPho.hh"
#include "GMaterialLib.hh"
#include "GBndLib.hh"

#include "X4PhysicalVolume.hh"
#include "X4MaterialLib.hh"

#include "G4Material.hh"
#include "G4Event.hh"
#include "G4Track.hh"
#include "G4TransportationManager.hh"
#include "G4Version.hh"


#include "PLOG.hh"

const plog::Severity G4Opticks::LEVEL = PLOG::EnvLevel("G4Opticks", "DEBUG")  ;

G4Opticks* G4Opticks::fInstance = NULL ;

const char* G4Opticks::OPTICKS_EMBEDDED_COMMANDLINE = "OPTICKS_EMBEDDED_COMMANDLINE" ; 
const char* G4Opticks::OPTICKS_EMBEDDED_COMMANDLINE_EXTRA = "OPTICKS_EMBEDDED_COMMANDLINE_EXTRA" ; 
const char* G4Opticks::fEmbeddedCommandLine_pro = " --compute --embedded --xanalytic --production --nosave" ;
const char* G4Opticks::fEmbeddedCommandLine_dev = " --compute --embedded --xanalytic --save --natural --printenabled --pindex 0" ;

/**
G4Opticks::EmbeddedCommandLine
--------------------------------

When the OPTICKS_EMBEDDED_COMMANDLINE envvar is not defined the default value of "pro" 
is used. If the envvar OPTICKS_EMBEDDED_COMMANDLINE is defined with 
special values of "dev" or "pro" then the corresponding static 
variable default commandlines are used for the embedded Opticks commandline.
Other values of the envvar are passed asis to the Opticks instanciation.

Calls to G4Opticks::setEmbeddedCommandLineExtra made prior to 
Opticks instanciation in G4Opticks::setGeometry will append to the 
embedded commandline setup via envvar or default. Caution that duplication 
of options or invalid combinations of options will cause asserts.

**/

std::string G4Opticks::EmbeddedCommandLine(const char* extra1, const char* extra2 )  // static
{
    const char* ecl  = SSys::getenvvar(OPTICKS_EMBEDDED_COMMANDLINE, "pro") ; 

    char mode = '?' ; 
    const char* explanation = "" ; 
    if(strcmp(ecl, "pro") == 0)
    {
        ecl = fEmbeddedCommandLine_pro ; 
        mode = 'P' ; 
        explanation = "using \"pro\" (production) commandline without event saving " ; 
    }
    else if(strcmp(ecl, "dev") == 0)
    {
        ecl = fEmbeddedCommandLine_dev ; 
        mode = 'D' ; 
        explanation = "using \"dev\" (development) commandline with full event saving " ; 
    } 
    else
    {
        mode = 'A' ; 
        explanation = "using custom commandline (for experts only) " ; 
    }

    const char* eclx = SSys::getenvvar(OPTICKS_EMBEDDED_COMMANDLINE_EXTRA, "") ; 

    std::stringstream ss ; 
    ss << ecl << " " ;
    LOG(info) << "Using ecl :[" << ecl << "] " << OPTICKS_EMBEDDED_COMMANDLINE ;
    LOG(info) << " mode(Pro/Dev/Asis) " << mode << " " << explanation ;    ; 

    if(extra1)
    {
        ss << " " << extra1 ; 
        LOG(info) << "Using extra1 argument :[" << extra1 << "]" ; 
    }
    if(extra2)
    {
        ss << " " << extra2 ; 
        LOG(info) << "Using extra2 argument :[" << extra2 << "]" ; 
    }
    if(eclx) 
    {
        ss << " " << eclx ; 
        LOG(info) << "Using eclx envvar :[" << eclx << "] "  << OPTICKS_EMBEDDED_COMMANDLINE_EXTRA ; 
    }
    return ss.str();  
}

/**
G4Opticks::setEmbeddedCommandLineExtra
----------------------------------------

Sets up the Opticks commandline used during Opticks instanciation in G4Opticks::InitOpticks
which is invoked from G4Opticks::setGeometry or G4Opticks::loadGeometry

**/

void  G4Opticks::setEmbeddedCommandLineExtra(const char* extra)  
{
    m_embedded_commandline_extra = extra ? strdup(extra) : NULL ; 
}
const char* G4Opticks::getEmbeddedCommandLineExtra() const 
{
    return m_embedded_commandline_extra ; 
}






/**
G4Opticks::InitOpticks
-------------------------

Invoked from G4Opticks::loadGeometry or G4Opticks::translateGeometry

Steps:

1. set the key 
2. instanciate Opticks in embedded manner, must be after setting the key 


As unfettered access to the commandline is not really practical in production running 
where the commandline is used by the host application the use of parse_argv=true 
by embedded Opticks should be regarded as a temporary kludge during development 
that will not be available in production.

Note that because the keyspec that is obtained from the geometry is needed 
prior to Opticks instanciation it is necessary to defer this until G4Opticks::setGeometry 
is called. That is problematic as it prevents being able to directly configure Opticks, 
eg to change rngmax other than via the commandline and envvars.

As a workaround for this the commandline_extra arg is used to provide a way to 
change the embedded command line from G4Opticks level.

**/
Opticks* G4Opticks::InitOpticks(const char* keyspec, const char* commandline_extra, bool parse_argv ) // static
{
    LOG(LEVEL) << "[" ;
    LOG(LEVEL) << "[BOpticksResource::Get " << keyspec   ;   
    BOpticksResource* rsc = BOpticksResource::Get(keyspec) ; 
    if(keyspec)
    {
        const char* keyspec2 = rsc->getKeySpec(); 
        assert( strcmp( keyspec, keyspec2) == 0 ); // prior creation of BOpticksResource/BOpticksKey with different spec would trip this
    }
    LOG(LEVEL) << "]BOpticksResource::Get" ;
    LOG(info) << std::endl << rsc->export_(); 
    
    const char* geospecific_options = rsc->getGDMLAuxUserinfoGeospecificOptions() ; 
    LOG(LEVEL) << "GDMLAuxUserinfoGeospecificOptions [" << geospecific_options << "]" ;  

    std::string ecl = EmbeddedCommandLine(commandline_extra, geospecific_options) ; 
    LOG(LEVEL) << "EmbeddedCommandLine : [" << ecl << "]" ; 

    LOG(LEVEL) << "[ok" ;
    Opticks* ok = NULL ; 
    if( parse_argv )
    {
        assert( PLOG::instance && "OPTICKS_LOG is needed to instanciate PLOG" );
        const SAr& args = PLOG::instance->args ; 
        LOG(info) << "instanciate Opticks using commandline captured by OPTICKS_LOG + embedded commandline" ;  
        args.dump(); 
        ok = new Opticks(args._argc, args._argv, ecl.c_str() );  // Opticks instanciation must be after BOpticksKey::SetKey
    }
    else
    {
        LOG(info) << "instanciate Opticks using embedded commandline only " ;
        std::cout << ecl << std::endl ;   
        ok = new Opticks(0,0, ecl.c_str() );  // Opticks instanciation must be after BOpticksKey::SetKey
    }
    LOG(LEVEL) << "]ok" ;

    LOG(LEVEL) << "[ok.configure" ;
    ok->configure();       // parses args and does resource setup
    LOG(LEVEL) << "]ok.configure" ;

    const char* idpath = ok->getIdPath(); 
    assert(idpath);
    LOG(LEVEL) << "] " << idpath ;
    return ok ; 
}


std::string G4Opticks::desc() const 
{
    std::stringstream ss ; 
    ss << "G4Opticks.desc"
       << " ok " << m_ok 
       ; 
   if(m_opmgr) ss << " opmgr " << m_opmgr << " " ;

   ss << std::endl 
       << ( m_ok ? m_ok->desc() : "-" ) 
       << std::endl 
       << ( m_ok ? m_ok->export_() : "-" ) 
       ;
    return ss.str() ; 
}

G4Opticks* G4Opticks::Get()
{
    if (!fInstance) fInstance = new G4Opticks;
    return fInstance ;
}

void G4Opticks::Initialize(const char* gdmlpath, bool standardize_geant4_materials)
{
    const G4VPhysicalVolume* world = CGDML::Parse(gdmlpath); 
    Initialize(world, standardize_geant4_materials); 
}

void G4Opticks::Initialize(const G4VPhysicalVolume* world, bool standardize_geant4_materials)
{
    G4Opticks* g4ok = Get(); 
    g4ok->setGeometry(world, standardize_geant4_materials) ; 
}

void G4Opticks::Finalize()  // static 
{
    G4Opticks* g4ok = Get(); 
    LOG(info) << g4ok->desc();

    g4ok->finalize();

    Opticks::Finalize() ; 
    delete fInstance ; 
    fInstance = NULL ;
}

G4Opticks::~G4Opticks()
{
    CAlignEngine::Finalize() ;
}

/**
G4Opticks::G4Opticks
----------------------

**/

G4Opticks::G4Opticks()
    :
    m_standardize_geant4_materials(false), 
    m_placement_outer_volume(false),
    m_world(NULL),
    m_ggeo(NULL),
    m_blib(NULL),
    m_hits_wrapper(NULL),
    m_embedded_commandline_extra(NULL),
    m_ok(NULL),
    m_way_enabled(false),
    m_way_mask(0),
    m_traverser(NULL),
    m_mtab(NULL),
    m_genstep_collector(NULL),
    m_primary_collector(NULL),
    m_lookup(NULL),
    m_opmgr(NULL),
    m_gensteps(NULL),
    m_genphotons(NULL),
    m_hits(NULL),
    m_hiys(NULL),
    m_num_hits(0),
    m_num_hiys(0),
    m_g4hit_collector(NULL),
    m_g4photon_collector(NULL),
    m_genstep_idx(0),
    m_g4evt(NULL),
    m_g4hit(NULL),
    m_sensorlib(NULL),
    m_skip_gencode(),
    m_skip_gencode_count(SSys::getenvintvec("OPTICKS_SKIP_GENCODE", m_skip_gencode, ',')),
    m_skip_gencode_totals(),
    m_profile(false),
    m_profile_path(SSys::getenvvar("OPTICKS_PROFILE_PATH")),
    m_profile_leak_mb(0.f)
{
    initSkipGencode() ; 

    assert( fInstance == NULL ); 
    fInstance = this ; 
    LOG(info) << "ctor : DISABLE FPE detection : as it breaks OptiX launches" ; 
    C4FPEDetection::InvalidOperationDetection_Disable();  // see notes/issues/OKG4Test_prelaunch_FPE_causing_fail.rst
}

void G4Opticks::initSkipGencode() 
{
    LOG(fatal) << "OPTICKS_SKIP_GENCODE m_skip_gencode_count " << m_skip_gencode_count ; 
    assert( m_skip_gencode_count == m_skip_gencode.size() ); 
    for(unsigned i=0 ; i < m_skip_gencode.size() ; i++)
    {
        unsigned gencode = m_skip_gencode[i] ;
        LOG(fatal) << " m_skip_gencode[" << i <<"] " << gencode << " " << OpticksGenstep::Gentype(gencode) ;    
        m_skip_gencode_totals[gencode] = 0 ; 
    }
}

void G4Opticks::setProfile(bool profile)
{
    m_profile = profile ; 
}
void G4Opticks::setProfilePath(const char* path)
{
    m_profile = true ; 
    m_profile_path = strdup(path) ; 
}
void G4Opticks::setProfileLeakMB(float profile_leak_mb)
{
    LOG(info) << " profile_leak_mb " << profile_leak_mb ; 
    m_profile = true ; 
    m_profile_leak_mb = profile_leak_mb ; 
}



void G4Opticks::dumpSkipGencode() const  
{
    LOG(fatal) << "OPTICKS_SKIP_GENCODE m_skip_gencode_count " << m_skip_gencode_count ; 
    assert( m_skip_gencode_count == m_skip_gencode.size() ); 
    for(unsigned i=0 ; i < m_skip_gencode.size() ; i++)
    {
        unsigned gencode = m_skip_gencode[i] ;
        unsigned total = m_skip_gencode_totals.at(gencode)  ; 
        LOG(fatal) 
            << " m_skip_gencode_totals[" 
            << std::setw(2) << gencode 
            <<"] " 
            << std::setw(6) << total 
            << " " 
            << OpticksGenstep::Gentype(gencode)
            ;    
    }
}

bool G4Opticks::isSkipGencode(unsigned gencode) const 
{
    return std::count(m_skip_gencode.begin(), m_skip_gencode.end(), gencode) > 0 ; 
}

void G4Opticks::finalize() const 
{
    dumpSkipGencode();
    if(m_profile) finalizeProfile(); 
}


void G4Opticks::finalizeProfile() const 
{
    NPY<float>* a = NPY<float>::make_from_vec(m_profile_stamps); 
    unsigned num_items = a->getNumItems() ;
    if(num_items < 4 )
    {
        LOG(fatal) << " not enough profile stamps" << num_items ;
        return ; 
    }

    a->reshape(-1, 4);
  
    if(m_profile_path)
    {
        LOG(info) << "saving time/vm stamps to path " << m_profile_path ; 
        LOG(info) << "make plot with: ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py " << m_profile_path ; 
        a->save(m_profile_path);  
    }
    else
    {
        LOG(info) << "to set path to save the profile set envvar OPTICKS_PROFILE_PATH or use G4Opticks::setProfilePath  " ; 
    }

    OpticksProfile::Report(a, m_profile_leak_mb); 
}


void G4Opticks::setPlacementOuterVolume(bool outer_volume)  // TODO: eliminate the need for this
{
    m_placement_outer_volume = outer_volume ;  
}

/**
G4Opticks::createCollectors
-----------------------------

**/

void G4Opticks::createCollectors()
{
    LOG(LEVEL) << "[" ; 
    m_mtab = new CMaterialTable(); 
    setupMaterialLookup();
    m_genstep_collector = new CGenstepCollector(m_lookup);   // <-- CG4 holds an instance too : and they are singletons, so should not use G4Opticks and CG4 together
    m_primary_collector = new CPrimaryCollector ; 
    m_g4hit_collector = new CPhotonCollector ; 
    m_g4photon_collector = new C4PhotonCollector ; 
    LOG(LEVEL) << "]" ; 
}

/**
G4Opticks::dbgdesc
-------------------

Dump the member variables::

   (gdb) printf "%s\n", g4opticks->dbgdesc()

**/

const char* G4Opticks::dbgdesc() const 
{
    std::string s = dbgdesc_() ; 
    return strdup(s.c_str());     
}

std::string G4Opticks::dbgdesc_() const 
{
    std::stringstream ss ; 
    ss
       << std::setw(32) << " this "                           << std::setw(12) << this << std::endl  
       << std::setw(32) << " m_standardize_geant4_materials " << std::setw(12) << m_standardize_geant4_materials << std::endl  
       << std::setw(32) << " m_world "                        << std::setw(12) << m_world << std::endl  
       << std::setw(32) << " m_ggeo "                         << std::setw(12) << m_ggeo << std::endl  
       << std::setw(32) << " m_blib "                         << std::setw(12) << m_blib << std::endl  
       << std::setw(32) << " m_ok "                           << std::setw(12) << m_ok << std::endl  
       << std::setw(32) << " m_traverser "                    << std::setw(12) << m_traverser << std::endl  
       << std::setw(32) << " m_mtab  "                        << std::setw(12) << m_mtab << std::endl
       << std::setw(32) << " m_genstep_collector "            << std::setw(12) << m_genstep_collector << std::endl
       << std::setw(32) << " m_primary_collector "            << std::setw(12) << m_primary_collector << std::endl
       << std::setw(32) << " m_lookup "                       << std::setw(12) << m_lookup << std::endl
       << std::setw(32) << " m_gensteps "                     << std::setw(12) << m_gensteps << std::endl
       << std::setw(32) << " m_genphotons "                   << std::setw(12) << m_genphotons << std::endl
       << std::setw(32) << " m_hits "                         << std::setw(12) << m_hits << std::endl 
       << std::setw(32) << " m_hits_wrapper "                 << std::setw(12) << m_hits_wrapper << std::endl 
       << std::setw(32) << " m_num_hits "                     << std::setw(12) << m_num_hits << std::endl 
       << std::setw(32) << " m_g4hit_collector "              << std::setw(12) << m_g4hit_collector << std::endl
       << std::setw(32) << " m_g4photon_collector "           << std::setw(12) << m_g4photon_collector << std::endl 
       << std::setw(32) << " m_genstep_idx "                  << std::setw(12) << m_genstep_idx << std::endl
       << std::setw(32) << " m_g4evt "                        << std::setw(12) << m_g4evt << std::endl 
       << std::setw(32) << " m_g4hit "                        << std::setw(12) << m_g4hit << std::endl 
       << std::setw(32) << " m_sensorlib "                    << std::setw(12) << m_sensorlib << std::endl  
       ;
    std::string s = ss.str(); 
    return s ; 
}


/**
G4Opticks::reset
-----------------

This *reset* should be called after both *propagateOpticalPhotons* 
and the hit data have been copied into Geant4 hit collections and 
before the next *propagateOpticalPhotons*. 

Omitting to run *reset* will cause gensteps to continually collect
from event to event with each propagation redoing the simulation again.

**/

void G4Opticks::reset()
{
    resetCollectors(); 

    if(m_hits)
    {
        m_hits->reset();   // the cloned hits (and hiys) are owned by G4Opticks, so they must be reset here  
        m_num_hits = 0 ; 
    }

    if(m_way_enabled && m_hiys)
    {
        LOG(fatal) << " m_way_enabled reset m_hiys " ; 
        m_hiys->reset(); 
        m_num_hiys = 0 ; 
    }

}


/**
G4Opticks::resetCollectors
-----------------------------

Resets the collectors and sets the array pointers borrowed from them to NULL.
Note that the arrays belong to their respective collectors
and are managed by them.
**/

void G4Opticks::resetCollectors()
{
    LOG(LEVEL) << "[" ; 
    m_genstep_collector->reset(); 
    m_gensteps = NULL ; 

    m_primary_collector->reset(); 

    m_g4hit_collector->reset(); 
    m_g4hit = NULL ; 

    m_g4photon_collector->reset(); 
    m_genphotons = NULL ; 
    LOG(LEVEL) << "]" ; 
}


/**
G4Opticks::setGeometry
------------------------

**/


void G4Opticks::setGeometry(const char* gdmlpath)
{
    const G4VPhysicalVolume* world = CGDML::Parse(gdmlpath);
    setGeometry(world);  
}

void G4Opticks::setGeometry(const G4VPhysicalVolume* world, bool standardize_geant4_materials)
{
    setStandardizeGeant4Materials(standardize_geant4_materials ); 
    setGeometry(world);  
}

void G4Opticks::setGeometry(const G4VPhysicalVolume* world)
{
    LOG(LEVEL) << "[" ; 

    LOG(LEVEL) << "( translateGeometry " ; 
    GGeo* ggeo = translateGeometry( world ) ;
    LOG(LEVEL) << ") translateGeometry " ; 

    if( m_standardize_geant4_materials )
    {
        standardizeGeant4MaterialProperties();
    }

    m_world = world ; 

    setGeometry(ggeo); 

    LOG(LEVEL) << "]" ; 
}

/**
G4Opticks::loadGeometry
-------------------------

Load geometry cache identified by the OPTICKS_KEY envvar.

**/


void G4Opticks::loadGeometry()
{
    const char* keyspec = NULL ;   // NULL means get keyspec from OPTICKS_KEY envvar 
    bool parse_argv = false ; 
    Opticks* ok = InitOpticks(keyspec, m_embedded_commandline_extra, parse_argv ); 
    GGeo* ggeo = GGeo::Load(ok); 
    setGeometry(ggeo); 
}


/**
G4Opticks::setGeometry(const GGeo* ggeo)
------------------------------------------

When GGeo is loaded from cache the sensor placement origin nodes 
are not available (as there is no Geant4 geometry tree in memory), 
but their number is available.

**/

void G4Opticks::setGeometry(const GGeo* ggeo)
{
    bool loaded = ggeo->isLoadedFromCache() ; 
    unsigned num_sensor = ggeo->getNumSensorVolumes(); 


    if( loaded == false )
    {
        if(m_placement_outer_volume) LOG(error) << "CAUTION : m_placement_outer_volume TRUE " ; 
        X4PhysicalVolume::GetSensorPlacements(ggeo, m_sensor_placements, m_placement_outer_volume);
        assert( num_sensor == m_sensor_placements.size() ) ; 
    }

    LOG(info) 
        << " GGeo: " 
        << ( loaded ? "LOADED FROM CACHE " : "LIVE TRANSLATED " )  
        << " num_sensor " << num_sensor 
        ;

    m_ggeo = ggeo ;
    m_blib = m_ggeo->getBndLib();  
    m_hits_wrapper = new GPho(m_ggeo) ;   // geometry aware photon hits wrapper

    m_ok = m_ggeo->getOpticks(); 
    m_way_enabled = m_ok->isWayEnabled() ; 
    m_way_mask = m_ok->getWayMask(); 
    m_ok->initSensorData(num_sensor);   // instanciates SensorLib 
    m_sensorlib = m_ok->getSensorLib(); 

    createCollectors(); 

    //CAlignEngine::Initialize(m_ok->getIdPath()) ;

    // OpMgr instanciates OpticksHub which adopts the pre-existing m_ggeo instance just translated (or loaded)
    LOG(LEVEL) << "( OpMgr " ; 
    m_opmgr = new OpMgr(m_ok) ;   
    LOG(LEVEL) << ") OpMgr " ; 

    m_recorder = G4OpticksRecorder::Get() ;  
    if(m_recorder) 
    {
        m_recorder->setGeometry(ggeo);  
    }
    else
    {
        LOG(error) << " no G4OpticksRecorder instance, meaning probably no CManager " ; 
    } 
}

bool G4Opticks::isWayEnabled() const 
{
    return m_way_enabled ; 
}
unsigned G4Opticks::getWayMask() const 
{
    return m_way_mask ; 
}




void G4Opticks::setStandardizeGeant4Materials(bool standardize_geant4_materials)
{
    m_standardize_geant4_materials = standardize_geant4_materials ; 
    assert( m_standardize_geant4_materials == false && "needs debugging as observed to mess up source materials"); 
}



bool G4Opticks::isLoadedFromCache() const
{
    return m_ggeo->isLoadedFromCache(); 
}


/**
G4Opticks::getNumSensorVolumes (pre-cache and post-cache)
------------------------------------------------------------

The number is obtained by GNodeLib::initSensorIdentity from 
counting volumes with associated sensorIndex in the volume identity array. 

**/

unsigned G4Opticks::getNumSensorVolumes() const 
{
    return m_ggeo->getNumSensorVolumes(); 
}

/**
G4Opticks::getSensorIdentityStandin (pre-cache and post-cache)
-----------------------------------------------------------------
**/

unsigned G4Opticks::getSensorIdentityStandin(unsigned sensorIndex) const 
{
    return m_ggeo->getSensorIdentityStandin(sensorIndex); 
}

/**
G4Opticks::getSensorPlacements (pre-cache live running only)
---------------------------------------------------------------

Sensor placements are the outer volumes of instance assemblies that 
contain sensor volumes.  The order of the returned vector of G4PVPlacement
is that of the Opticks sensorIndex. 
This vector allows the connection between the Opticks sensorIndex 
and detector specific handling of sensor quantities to be established.

NB this assumes only one volume with a sensitive surface within each 
repeated geometry instance

For example JUNO uses G4PVPlacement::GetCopyNo() as a non-contiguous PMT 
identifier, which allows lookup of efficiencies and PMT categories.

Sensor data is assigned via calls to setSensorData with 
the 0-based contiguous Opticks sensorIndex as the first argument.   

**/

const std::vector<G4PVPlacement*>& G4Opticks::getSensorPlacements() const 
{
    return m_sensor_placements ;
}

/**
G4Opticks::getNumDistinctPlacementCopyNo 
-----------------------------------------

GDML physvol/@copynumber attribute persists the CopyNo, but this 
defaults to 0 unless set at detector level. When CopyNo is not 
used as a sensor identifier or when running from cache this 
is expected to return 1.

**/

unsigned G4Opticks::getNumDistinctPlacementCopyNo() const 
{
    std::set<int> copynumber ; 
    for(unsigned i=0 ; i < m_sensor_placements.size() ; i++)
    {
        const G4PVPlacement* pv = m_sensor_placements[i];
        G4int copyNo = pv->GetCopyNo(); 
        copynumber.insert(copyNo);  
    }
    return copynumber.size(); 
}



/**
G4Opticks::setSensorData
---------------------------

Calls to this for all sensor_placements G4PVPlacement provided by G4Opticks::getSensorPlacements
provides a way to associate the Opticks contiguous 1-based sensorIndex with a detector 
defined sensor identifier. 

Within JUNO simulation framework this is used from LSExpDetectorConstruction::SetupOpticks.

sensorIndex 
    1-based contiguous index used to access the sensor data, 
    the (index-1) must be less than the number of sensors
efficiency_1 
efficiency_2
    two efficiencies which are multiplied together with the local angle dependent efficiency 
    to yield the detection efficiency used together with a uniform random to set the 
    EFFICIENCY_COLLECT (EC) or EFFICIENCY_CULL (EX) flag for photons that already 
    have SURFACE_DETECT flag 
category
    used to distinguish between sensors with different theta-phi textures   
identifier
    detector specific integer representing a sensor, does not need to be contiguous


Why call G4Opticks::setSensorData ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not everything is in GDML.  Detector simulation frameworks often add things on top 
for example local theta and/or phi dependent sensor efficiencies and additional 
efficiency factors.  Also detectors often use there own numbering schemes for sensors. 
That is what the sensor_identifier is. 

Normally after hits are collected detector simulation frameworks cull them 
randomly based on efficiencies. G4Opticks::setSensorData allows that culling 
to effectively be done on the GPU so the CPU side memory requirements can be reduced 
by a factor of the  efficiency. Often that is something like a quarter of the memory 
reqirements. It also correspondingly reduces the volume of hit data that needs to be copied 
from GPU to CPU.

**/

void G4Opticks::setSensorData(unsigned sensorIndex, float efficiency_1, float efficiency_2, int category, int identifier)
{
    assert( m_sensorlib ); 
    m_sensorlib->setSensorData(sensorIndex, efficiency_1, efficiency_2, category, identifier); 
}

void G4Opticks::getSensorData(unsigned sensorIndex, float& efficiency_1, float& efficiency_2, int& category, int& identifier) const 
{
    assert( m_sensorlib ); 
    m_sensorlib->getSensorData(sensorIndex, efficiency_1, efficiency_2, category, identifier);
}

int G4Opticks::getSensorIdentifier(unsigned sensorIndex) const 
{
    assert( m_sensorlib ); 
    return m_sensorlib->getSensorIdentifier(sensorIndex);
}


/**
G4Opticks::setSensorAngularEfficiency
----------------------------------------

Invoked from detector specific code, eg LSExpDetectorConstruction::SetupOpticks

**/

void G4Opticks::setSensorAngularEfficiency( const std::vector<int>& shape, const std::vector<float>& values, 
        int theta_steps, float theta_min, float theta_max, 
        int phi_steps,   float phi_min, float phi_max )
{
    assert(m_sensorlib);
    m_sensorlib->setSensorAngularEfficiency(shape, values, 
        theta_steps, theta_min, theta_max, 
        phi_steps, phi_min, phi_max
    );
    m_ok->setAngularEnabled(true);   // see notes/issues/runtime_angular_control.rst

}

void G4Opticks::setSensorAngularEfficiency( const NPY<float>* sensor_angular_efficiency )
{
    assert( m_sensorlib ); 
    m_sensorlib->setSensorAngularEfficiency( sensor_angular_efficiency ); 
    m_ok->setAngularEnabled(true);   // see notes/issues/runtime_angular_control.rst
}

void G4Opticks::saveSensorLib(const char* dir, const char* reldir) const 
{
    LOG(info) << " saving to " << dir << "/" << ( reldir ? reldir : "" )  ;  
    m_sensorlib->save(dir, reldir ); 
}


void G4Opticks::render_snap() 
{
    if(!m_opmgr) return ; 
    m_opmgr->render_snap(); 
}




/**
G4Opticks::translateGeometry
------------------------------

Canonically invoked by G4Opticks::setGeometry with top:G4VPhysicalVolume.


1. A keyspec representing the identity of the world G4VPhysicalVolume geometry is formed, 
   and this is set with BOpticksKey::SetKey prior to Opticks instanciation.

   NB THIS MAKES THE GEOMETRY DEPEND ONLY ON THE WORLD ARGUMENT, 
   THERE IS NO SENSITIVITY TO THE OPTICKS_KEY ENVVAR 

2. An embedded Opticks instance is instanciated using the embedded commandline 

3. any preexisting geocache is ignored, due to the live=true option to GGeo instanciation

4. X4PhysicalVolume is used to do the direct translation of the Geant4 geometry into the GGeo instance 

**/

GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
{
    LOG(verbose) << "( key" ;
    const char* keyspec = X4PhysicalVolume::Key(top) ; 

    bool parse_argv = false ; 
    Opticks* ok = InitOpticks(keyspec, m_embedded_commandline_extra, parse_argv ); 
 
    const char* dbggdmlpath = ok->getDbgGDMLPath(); 
    if( dbggdmlpath != NULL )
    { 
        LOG(info) << "( CGDML" ;
        CGDML::Export( dbggdmlpath, top ); 
        LOG(info) << ") CGDML" ;
    }

    const char* origin = Opticks::OriginGDMLPath(); 
    LOG(info) << "( CGDML " << origin  ;
    CGDML::Export( origin, top ); 
    LOG(info) << ") CGDML " ;  

    if(ok->isGDMLKludge())
    {
        LOG(info) << "( CGDMLKludge " << origin << " --gdmlkludge"  ;
        const char* kludge_path = CGDMLKludge::Fix( origin );
        if(kludge_path) LOG(info) << "kludge_path " << kludge_path ;  
        LOG(info) << ") CGDMLKludge " ;  
    }
    else
    {
        LOG(info) << "CGDMLKludge not-applied as no option : --gdmlkludge  " ;
    }


    LOG(info) << "( GGeo instanciate" ;
    bool live = true ;       // <--- for now this ignores preexisting cache in GGeo::init 
    GGeo* gg = new GGeo(ok, live) ;
    LOG(info) << ") GGeo instanciate " ;

    LOG(info) << "( GGeo populate" ;
    X4PhysicalVolume xtop(gg, top) ;   
    LOG(info) << ") GGeo populate" ;

    LOG(info) << "( GGeo::postDirectTranslation " ;
    gg->postDirectTranslation(); 
    LOG(info) << ") GGeo::postDirectTranslation " ;

    return gg ; 
}




/**
G4Opticks::standardizeGeant4MaterialProperties
-----------------------------------------------

Invoked by G4Opticks::setGeometry when argument requests.

Standardize G4 material properties to use the Opticks standard domain, 
this works by replacing existing Geant4 MPT 
with one converted from the Opticks property map, which are 
standardized on material collection.

**/

void G4Opticks::standardizeGeant4MaterialProperties()
{
    LOG(fatal) << "[" ; 
    G4MaterialTable* mtab = G4Material::GetMaterialTable();   
    const GMaterialLib* mlib = GMaterialLib::Get(); 
    X4MaterialLib::Standardize( mtab, mlib ) ;  
    LOG(fatal) << "]" ; 
}




void G4Opticks::setupMaterialLookup()
{
    const std::map<std::string, unsigned>& A = m_mtab->getMaterialMap() ;
    const std::map<std::string, unsigned>& B = m_blib->getMaterialLineMapConst() ;
 
    m_lookup = new NLookup ; 
    m_lookup->setA(A,"","CMaterialTable");
    m_lookup->setB(B,"","GBndLib");    // shortname eg "GdDopedLS" to material line mapping 
    m_lookup->close(); 
}











unsigned G4Opticks::getNumPhotons() const 
{
    return m_genstep_collector->getNumPhotons()  ; 
}
unsigned G4Opticks::getNumPhotonsSum() const 
{
    return m_genstep_collector->getNumPhotonsSum()  ; 
}





unsigned G4Opticks::getNumGensteps() const 
{
    return m_genstep_collector->getNumGensteps()  ; 
}

/**
G4Opticks::getMaxGensteps
--------------------------

Default of zero means no limit.  Setting this to 1 or a small 
number of gensteps can sometimes be convienient whilst debugging.
Used for example from the CerenkovMinimal examples Ctx::setTrack. 

**/

unsigned G4Opticks::getMaxGensteps() const 
{
    return 0  ; 
}






void G4Opticks::setAlignIndex(int align_idx) const 
{
    CAlignEngine::SetSequenceIndex(align_idx); 
}

/**
G4Opticks::propagateOpticalPhotons
-----------------------------------

Invoked from EventAction::EndOfEventAction

Note that m_g4evt OpticksEvent here is mostly empty
as the fully instrumented step-by-step recording 
is not used for simplicity. This means the Geant4
simulation is kept very simple and similar to 
standard Geant4 usage.

To allow superficial hit level debugging Geant4 hits 
collected by m_g4hit_collector CPhotonCollector are 
persisted via the OpticksEvent machinery 
with OpticksEvent::saveHitData

optickscore/OpticksEvent.cc::

    291    public:
    292        // used from G4Opticks for the minimal G4 side instrumentation of "1st executable"
    293        void saveHitData(NPY<float>* ht) const ;
    294        void saveSourceData(NPY<float>* so) const ;

who owns the gensteps : who gets to reset them ?

* recall that gensteps are special, they do not get reset by OpticksEvent::reset
  but they are saved by OpticksEvent::save when not in production mode

what is the relationship between gensteps and G4Event ?

* eventually want to be able to gang together gensteps from multiple
  G4Event into one genstep array : but for now are assuming
  that there is 1-to-1 relationship between G4Event and genstep arrays


TO TRY: 

* relying on G4Opticks::reset being called might allow to not cloning the hit/hiy ?  



**/

int G4Opticks::propagateOpticalPhotons(G4int eventID) 
{
    LOG(LEVEL) << "[[" ; 
    assert( m_genstep_collector ); 
    m_gensteps = m_genstep_collector->getGensteps(); 
    m_gensteps->setArrayContentVersion(G4VERSION_NUMBER); 
    m_gensteps->setArrayContentIndex(eventID); 

    unsigned num_gensteps = m_gensteps->getNumItems(); 
    LOG(LEVEL) << " num_gensteps "  << num_gensteps ;  
    if( num_gensteps == 0 )
    {
        LOG(fatal) << "SKIP as no gensteps have been collected " ; 
        return 0 ; 
    }


    unsigned tagoffset = eventID ;  // tags are 1-based : so this will normally be the Geant4 eventID + 1

    if(!m_ok->isProduction()) // --production
    {
        const char* gspath = m_ok->getDirectGenstepPath(tagoffset);   
        LOG(LEVEL) << "[ saving gensteps to " << gspath ; 
        m_gensteps->save(gspath);  
        LOG(LEVEL) << "] saving gensteps to " << gspath ; 
    }

    if(m_ok->isDbgGSSave()) // --dbggssave
    {
        const char* dbggspath = m_ok->getDebugGenstepPath(eventID) ; // --dbggsdir default dir is $TMP/dbggs   
        LOG(LEVEL) << " saving debug gensteps to dbggspath " << dbggspath << " eventID " << eventID ;   
        m_gensteps->save(dbggspath);   // eg $TMP/dbggs/0.npy  
    }


    // initial generated photons before propagation 
    // CPU genphotons needed only while validating 
    m_genphotons = m_g4photon_collector->getPhoton(); 
    m_genphotons->setArrayContentVersion(G4VERSION_NUMBER); 

    //const char* phpath = m_ok->getDirectPhotonsPath(); 
    //m_genphotons->save(phpath); 

   
    if(m_opmgr)
    {
        m_opmgr->setGensteps(m_gensteps);      

        LOG(LEVEL) << "[ m_opmgr->propagate " ; 
        m_opmgr->propagate();     // GPU simulation is done in here 
        LOG(LEVEL) << "] m_opmgr->propagate " ; 

        OpticksEvent* event = m_opmgr->getEvent(); 
        m_hits = event->getHitData()->clone() ;      // TOTRY: moving OpMgr::reset later inside G4Opticks::reset would avoid the need to clone ?
        m_num_hits = m_hits->getNumItems() ; 

        if(m_way_enabled)
        {
            m_hiys = event->getHiyData()->clone() ; 
            m_num_hiys = m_hits->getNumItems() ; 
            LOG(LEVEL) << " m_way_enabled num_hiys " << m_num_hiys ;
            m_hits->setAux(m_hiys);   // associate the extra hiy selected from way buffer with hits array 
        }
        else
        {
            LOG(LEVEL) << " NOT-m_way_enabled " ;  
        }

        m_hits_wrapper->setPhotons( m_hits );  // (GPho)


        if(!m_ok->isProduction())
        {
            // minimal g4 side instrumentation in "1st executable" 
            // do after propagate, so the event will have been created already
            m_g4hit = m_g4hit_collector->getPhoton();  
            m_g4evt = m_opmgr->getG4Event(); 
            if(m_g4evt)  // the "cfg4evt" is no longer created by default
            {
                m_g4evt->saveHitData( m_g4hit ) ; // pass thru to the dir, owned by m_g4hit_collector ?
                m_g4evt->saveSourceData( m_genphotons ) ; 
            }
        }

        m_opmgr->reset();   
        // reset : clears OpticksEvent buffers, excluding gensteps
        //         must clone any buffers to be retained before the reset

        
        if(m_profile && m_profile_leak_mb > 0.f)
        {
            size_t leak_bytes = size_t(1000000*m_profile_leak_mb) ;  
            LOG(fatal) << "m_profile_leak_mb > 0.f : " << m_profile_leak_mb << " deliberate leak_bytes " << leak_bytes ; 
            char* leak = new char[leak_bytes] ; 
            assert( leak ); 
        }
    }

    LOG(LEVEL) << "]] num_hits " << m_num_hits ; 

    if( m_profile )
    {
        glm::vec4 stamp = OpticksProfile::Stamp(); 
        m_profile_stamps.push_back( stamp.x );  
        m_profile_stamps.push_back( stamp.y );  
        m_profile_stamps.push_back( stamp.z );  
        m_profile_stamps.push_back( stamp.w );  
    } 

    return m_num_hits ;   
}




NPY<float>* G4Opticks::getGensteps() const 
{
    return m_gensteps ; 
}
void G4Opticks::saveGensteps(const char* path) const 
{
    m_gensteps->save(path); 
}
void G4Opticks::saveGensteps(const char* dir, const char* name) const 
{
    m_gensteps->save(dir, name ); 
}
void G4Opticks::saveGensteps(const char* dir, const char* name_prefix, int name_index, const char* ext) const 
{
    const char* name = SStr::Concat_<int>(name_prefix, name_index, ext ); 
    m_gensteps->save(dir, name ); 
    free((void*)name); 
}


NPY<float>* G4Opticks::getHits() const 
{
    return m_hits ; 
}
void G4Opticks::saveHits(const char* path) const 
{
    m_hits->save(path); 
}
void G4Opticks::saveHits(const char* dir, const char* name) const 
{
    m_hits->save(dir, name ); 
}
void G4Opticks::saveHits(const char* dir, const char* name_prefix, int name_index, const char* ext) const 
{
    const char* name = SStr::Concat_<int>(name_prefix, name_index, ext ); 
    m_hits->save(dir, name ); 
    free((void*)name); 
}


void G4Opticks::dumpHits(const char* msg) const 
{
    unsigned maxDump = 0 ; 
    m_hits_wrapper->dump(msg, maxDump);
}


unsigned G4Opticks::getNumHit() const
{
    return m_num_hits ; 
}


/**
G4Opticks::getHit
-------------------

The local position, direction and wavelength are within the frame 
of the last intersect volume, ie the sensor volume.

Local positions, directions and polarizations are obtained using 
the geometry aware *m_hits_wrapper(GPho)* which looks up the 
appropriate transform avoiding having to almost double the size 
of the photon.

Hit data is provided by m_hits_wrapper (GPho) which wraps m_hits.


**/

void G4Opticks::getHit(unsigned i, G4OpticksHit* hit, G4OpticksHitExtra* hit_extra ) const 
{
    assert( i < m_num_hits && hit ); 

    glm::vec4 post = m_hits_wrapper->getPositionTime(i);      
    glm::vec4 dirw = m_hits_wrapper->getDirectionWeight(i);      
    glm::vec4 polw = m_hits_wrapper->getPolarizationWavelength(i); 

    glm::vec4 local_post = m_hits_wrapper->getLocalPositionTime(i);      
    glm::vec4 local_dirw = m_hits_wrapper->getLocalDirectionWeight(i);      
    glm::vec4 local_polw = m_hits_wrapper->getLocalPolarizationWavelength(i);      

    OpticksPhotonFlags pflag = m_hits_wrapper->getOpticksPhotonFlags(i); 

    hit->global_position.set(double(post.x), double(post.y), double(post.z)); 
    hit->time = double(post.w) ; 
    hit->global_direction.set(double(dirw.x), double(dirw.y), double(dirw.z)); 
    hit->weight = double(dirw.w) ; 
    hit->global_polarization.set(double(polw.x), double(polw.y), double(polw.z)); 
    hit->wavelength = double(polw.w);  

    hit->local_position.set(double(local_post.x), double(local_post.y), double(local_post.z)); 
    hit->local_direction.set(double(local_dirw.x), double(local_dirw.y), double(local_dirw.z)); 
    hit->local_polarization.set(double(local_polw.x), double(local_polw.y), double(local_polw.z)); 

    hit->boundary      = pflag.boundary ; 
    hit->sensorIndex   = pflag.sensorIndex ; 
    hit->nodeIndex     = pflag.nodeIndex ; 
    hit->photonIndex   = pflag.photonIndex ; 
    hit->flag_mask     = pflag.flagMask ; 

    hit->is_cerenkov       = (pflag.flagMask & CERENKOV) != 0 ; 
    hit->is_reemission     = (pflag.flagMask & BULK_REEMIT) != 0 ; 

    // via m_sensorlib 
    hit->sensor_identifier = getSensorIdentifier(pflag.sensorIndex); 

    if(hit_extra != NULL)
    {
        if(m_hits_wrapper->hasWay() )
        {
            glm::vec4 way_post = m_hits_wrapper->getWayPositionTime(i); 
            float     origin_time   = m_hits_wrapper->getWayOriginTime(i); 
            int       origin_trackID = m_hits_wrapper->getWayOriginTrackID(i); 

            hit_extra->boundary_pos.set(double(way_post.x), double(way_post.y), double(way_post.z));
            hit_extra->boundary_time = double(way_post.w);
            hit_extra->origin_time = origin_time ; 
            hit_extra->origin_trackID = origin_trackID ; 
        }      
        else
        {
            LOG(fatal) << "Extra hit info requires the --way option on embedded opticks commandline " ;    
        } 
    }
}




/**
G4Opticks::collectSecondaryPhotons
-----------------------------------

This is invoked from the tail of the PostStepDoIt of
instrumented photon producing processes. See L4Cerenkov
**/

void G4Opticks::collectSecondaryPhotons(const G4VParticleChange* pc)
{
    // equivalent collection in "2nd" fully instrumented executable 
    // is invoked from CGenstepSource::generatePhotonsFromOneGenstep
    m_g4photon_collector->collectSecondaryPhotons( pc, m_genstep_idx );
    m_genstep_idx += 1 ; 
}


/**
G4Opticks::setGenstepReservation
----------------------------------

Setting the genstep reservation is optional. 
Doing so may reduce the resource usage when collecting 
large numbers of gensteps. For maximum effect the *max_gensteps_expected* 
value should be larger than the maximum expected number of gensteps 
collected prior to a reset bringing that down to zero. 
Values less than the actual maxium do not cause a problem.

**/

void G4Opticks::setGenstepReservation(int max_gensteps_expected)
{
    LOG(LEVEL) << " max_gensteps_expected " << max_gensteps_expected ;  
    m_genstep_collector->setReservation(max_gensteps_expected); 
}

int  G4Opticks::getGenstepReservation() const 
{
    return m_genstep_collector->getReservation() ;  
}


CGenstep G4Opticks::collectGenstep_G4Scintillation_1042(  
     const G4Track* aTrack, 
     const G4Step* aStep, 
     G4int    numPhotons, 
     G4int    ScintillationType,  // 1:Fast, 2:Slow
     G4double ScintillationTime, 
     G4double ScintillationRiseTime
)
{
    // CAUTION : UNTESTED CODE
    G4StepPoint* pPreStepPoint  = aStep->GetPreStepPoint();
    G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();

    G4ThreeVector x0 = pPreStepPoint->GetPosition();
    G4double      t0 = pPreStepPoint->GetGlobalTime();
    G4ThreeVector deltaPosition = aStep->GetDeltaPosition() ;

    const G4DynamicParticle* aParticle = aTrack->GetDynamicParticle();
    const G4Material* aMaterial = aTrack->GetMaterial();

    G4double preVelocity  = pPreStepPoint->GetVelocity() ; 
    G4double postVelocity = pPostStepPoint->GetVelocity() ; 
 
    assert( m_genstep_collector ); 
    CGenstep gs = m_genstep_collector->collectScintillationStep(

         OpticksGenstep_G4Scintillation_1042,           // (int)gentype       (0) 
         aTrack->GetTrackID(),                          // (int)ParenttId     
         aMaterial->GetIndex(),                         // (int)MaterialIndex
         numPhotons,                                    // (int)NumPhotons

         x0.x(),                                        // x0.x               (1)
         x0.y(),                                        // x0.y
         x0.z(),                                        // x0.z
         t0,                                            // t0 

         deltaPosition.x(),                             // DeltaPosition.x    (2)
         deltaPosition.y(),                             // DeltaPosition.y    
         deltaPosition.z(),                             // DeltaPosition.z    
         aStep->GetStepLength(),                        // step_length 

         aParticle->GetDefinition()->GetPDGEncoding(),  // (int)code          (3) 
         aParticle->GetDefinition()->GetPDGCharge(),    // charge
         aTrack->GetWeight(),                           // weight 
         preVelocity,                                   //  

         ScintillationType,                             // (int) Fast:1 Slow:2
         0.,                                            // 
         0.,                                            // 
         0.,                                            // 

         ScintillationTime,                             //  ScintillationTime         (5)
         ScintillationRiseTime,                         // 
         postVelocity,                                  //  Other1
         0                                              //  Other2
    ) ;

    LOG(LEVEL) << gs.desc() ; 
    return gs ; 
}


/**
G4Opticks::collectGenstep_DsG4Scintillation_r3971
-----------------------------------------------------

Gensteps are have the size of 6 quads, eg 6*float4.
The meaning of the content of the genstep, particularly 
the last two quads depend on the particular implementation 
of scintillation and uses parameter names from that implementation.

Separate scintillation structs to handle different versions
of scintillation implementation are used to interpret the 
gensteps, see eg optixrap/cu/scintillationstep.h 

slowerRatio
slowTimeConstant
slowerTimeConstant
ScintillationTime
     will be one of slowTimeConstant or slowerTimeConstant  
     depending on whether have both slow and fast and on the scnt loop index 
scnt
     loop index, 1:fast 2:slow

**/

CGenstep G4Opticks::collectGenstep_DsG4Scintillation_r3971(  
     const G4Track* aTrack, 
     const G4Step* aStep, 
     G4int    numPhotons, 
     G4int    scnt,          //  1:fast 2:slow
     G4double slowerRatio,
     G4double slowTimeConstant,
     G4double slowerTimeConstant,
     G4double ScintillationTime
    )
{
    G4StepPoint* pPreStepPoint  = aStep->GetPreStepPoint();
    G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();

    G4ThreeVector x0 = pPreStepPoint->GetPosition();
    G4double      t0 = pPreStepPoint->GetGlobalTime();
    G4ThreeVector deltaPosition = aStep->GetDeltaPosition() ;
    G4double meanVelocity = (pPreStepPoint->GetVelocity()+pPostStepPoint->GetVelocity())/2. ; 

    const G4DynamicParticle* aParticle = aTrack->GetDynamicParticle();
    const G4Material* aMaterial = aTrack->GetMaterial();

    assert( m_genstep_collector ); 
    CGenstep gs = m_genstep_collector->collectScintillationStep(

        OpticksGenstep_DsG4Scintillation_r3971,         // (int)gentype                   (0)
        aTrack->GetTrackID(),                           // (int)parentId                                        
        aMaterial->GetIndex(),                          // (int)currently_not_used                        
        numPhotons,

        x0.x(),                                         // (double) preStep position      (1)
        x0.y(),
        x0.z(),
        t0,                                             // (double) preStep global time

        deltaPosition.x(),                              // DeltaPosition.x                (2)
        deltaPosition.y(),                              // DeltaPosition.y    
        deltaPosition.z(),                              // DeltaPosition.z    
        aStep->GetStepLength(),                         // step_length 

        aParticle->GetDefinition()->GetPDGEncoding(),  // (int)code                       (3) 
        aParticle->GetDefinition()->GetPDGCharge(),    // charge
        aTrack->GetWeight(),                           // weight 
        meanVelocity,                                  // MeanVelocity 

        scnt,                                          // (int) fast/slow scnt index      (4)
        slowerRatio,                                   // slowerRatio
        slowTimeConstant,                              // slowTimeConstant
        slowerTimeConstant,                            // slowerTimeConstant

        ScintillationTime,                             //  ScintillationTime              (5)
        0,                                             //  
        0,                                             // 
        0                                              // 
    );

    LOG(LEVEL)  << gs.desc() ; 
    return gs ; 
}

/**
G4Opticks::collectGenstep_DsG4Scintillation_r4695
--------------------------------------------------

Genstep slots filled here must correspond to their loading 
and usage in oxrap/cu/Genstep_DsG4Scintillation_r4695.h

**/

CGenstep G4Opticks::collectGenstep_DsG4Scintillation_r4695(  
     const G4Track* aTrack, 
     const G4Step* aStep, 
     G4int    numPhotons, 
     G4int    scnt,          //  1:fast 2:slow
     G4double ScintillationTime
    )
{
    G4StepPoint* pPreStepPoint  = aStep->GetPreStepPoint();
    G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();

    G4ThreeVector x0 = pPreStepPoint->GetPosition();
    G4double      t0 = pPreStepPoint->GetGlobalTime();
    G4ThreeVector deltaPosition = aStep->GetDeltaPosition() ;
    G4double meanVelocity = (pPreStepPoint->GetVelocity()+pPostStepPoint->GetVelocity())/2. ; 

    const G4DynamicParticle* aParticle = aTrack->GetDynamicParticle();
    const G4Material* aMaterial = aTrack->GetMaterial();

    assert( m_genstep_collector ); 
    CGenstep gs = m_genstep_collector->collectScintillationStep(

        OpticksGenstep_DsG4Scintillation_r4695,         // (int)gentype                   (0)
        aTrack->GetTrackID(),                           // (int)parentId                                        
        aMaterial->GetIndex(),                          // (int) not used for scintillation, but is for cerenkov                        
        numPhotons,

        x0.x(),                                         // (double) preStep position      (1)
        x0.y(),
        x0.z(),
        t0,                                             // (double) preStep global time

        deltaPosition.x(),                              // DeltaPosition.x                (2)
        deltaPosition.y(),                              // DeltaPosition.y    
        deltaPosition.z(),                              // DeltaPosition.z    
        aStep->GetStepLength(),                         // step_length 

        aParticle->GetDefinition()->GetPDGEncoding(),  // (int)code                       (3) 
        aParticle->GetDefinition()->GetPDGCharge(),    // charge
        aTrack->GetWeight(),                           // weight 
        meanVelocity,                                  // MeanVelocity 

        scnt,                                          // (int) fast/slow scnt index      (4)
        0,                                             // 
        0,                                             // 
        0,                                             // 

        ScintillationTime,                             // (double)ScintillationTime       (5)
        0,                                             //  
        0,                                             // 
        0                                              // 
    );

    LOG(LEVEL)  << gs.desc() ; 
    return gs ; 
}


/**
G4Opticks::collectGenstep_G4Cerenkov_1042
--------------------------------------------

2018/9/8 Geant4.1042 requires both velocities so:
     meanVelocity->preVelocity
     spare1->postVelocity 
     see notes/issues/ckm_cerenkov_generation_align_small_quantized_deviation_g4_g4.rst
**/

CGenstep G4Opticks::collectGenstep_G4Cerenkov_1042(  
     const G4Track*  aTrack, 
     const G4Step*   aStep, 
     G4int       numPhotons,

     G4double    betaInverse,
     G4double    pmin,
     G4double    pmax,
     G4double    maxCos,

     G4double    maxSin2,
     G4double    meanNumberOfPhotons1,
     G4double    meanNumberOfPhotons2
    )
{
    G4StepPoint* pPreStepPoint  = aStep->GetPreStepPoint();
    G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();

    G4ThreeVector x0 = pPreStepPoint->GetPosition();
    G4double      t0 = pPreStepPoint->GetGlobalTime();
    G4ThreeVector deltaPosition = aStep->GetDeltaPosition() ;

    G4double wmin_nm = h_Planck*c_light/pmax/nm ; 
    G4double wmax_nm = h_Planck*c_light/pmin/nm ; 
    bool wl_minmax = true ; 

    LOG(LEVEL)
        << " pmin/eV " << std::setw(10) << std::fixed << std::setprecision(3) << pmin/eV
        << " pmax/eV " << std::setw(10) << std::fixed << std::setprecision(3) << pmax/eV
        << " wmin_nm  " << std::setw(10) << std::fixed << std::setprecision(3) << wmin_nm
        << " wmax_nm  " << std::setw(10) << std::fixed << std::setprecision(3) << wmax_nm
        << " wl_minmax " << wl_minmax
        ;

    const G4DynamicParticle* aParticle = aTrack->GetDynamicParticle();
    const G4Material* aMaterial = aTrack->GetMaterial();

    G4double preVelocity = pPreStepPoint->GetVelocity() ;
    G4double postVelocity = pPostStepPoint->GetVelocity() ; 
 
    assert( m_genstep_collector ); 
    CGenstep gs = m_genstep_collector->collectCerenkovStep(

         OpticksGenstep_G4Cerenkov_1042,                // (int)gentype       (0)
         aTrack->GetTrackID(),                          // (int)ParenttId     
         aMaterial->GetIndex(),                         // (int)MaterialIndex
         numPhotons,                                    // (int)NumPhotons

         x0.x(),                                        // x0.x               (1)
         x0.y(),                                        // x0.y
         x0.z(),                                        // x0.z
         t0,                                            // t0 

         deltaPosition.x(),                             // DeltaPosition.x    (2)
         deltaPosition.y(),                             // DeltaPosition.y    
         deltaPosition.z(),                             // DeltaPosition.z    
         aStep->GetStepLength(),                        // step_length 

         aParticle->GetDefinition()->GetPDGEncoding(),  // (int)code          (3) 
         aParticle->GetDefinition()->GetPDGCharge(),    // charge
         aTrack->GetWeight(),                           // weight 
         preVelocity,                                   // preVelocity 

         betaInverse,                                   //                    (4) 
         wl_minmax ? wmin_nm : pmin,
         wl_minmax ? wmax_nm : pmax,
         maxCos,

         maxSin2,                                       //                    (5)
         meanNumberOfPhotons1,
         meanNumberOfPhotons2,
         postVelocity
    );

    LOG(LEVEL)  << gs.desc() ;  
    return gs ; 
}



 

/**
G4Opticks::collectDefaultTorchStep
-----------------------------------

Used from G4OKTest for debugging only.

Would like to use from CG4Test OKG4Test so need 
to split/move this to lower level.

Split off helper method to create the default TORCH OpticksGenstep
OpticksGen is the natural place for that : but that seems 
to be in decline : not used from G4Opticks world. So adding 
in 

Needs m_ggeo and m_genstep_collector (CGenstepCollector)
but using frame_transform argument would allow to not need
m_ggeo so could move this to within CGenstepCollector

**/

CGenstep G4Opticks::collectDefaultTorchStep(unsigned num_photons, int node_index, unsigned originTrackID )
{
    const OpticksGenstep* ogs = m_ggeo->createDefaultTorchStep(num_photons, node_index, originTrackID); 
    assert( m_genstep_collector ); 
    CGenstep gs = m_genstep_collector->collectTorchGenstep(ogs);  
    return gs ; 
} 


/**
G4Opticks::collectHit
-----------------------

Intended to collect standard Geant4 hits into the m_g4hit_collector 
CPhotonCollector instance. This allows superficial hit level  
comparisons between Opticks and Geant4 simulations.

For example this can be used from SensitiveDetector::ProcessHits 
in examples/Geant4/CerenkovMinimal/SensitiveDetector.cc

**/

void G4Opticks::collectHit
    (
        G4double             pos_x,  
        G4double             pos_y,  
        G4double             pos_z,  
        G4double             time ,

        G4double             dir_x,  
        G4double             dir_y,  
        G4double             dir_z,  
        G4double             weight ,

        G4double             pol_x,  
        G4double             pol_y,  
        G4double             pol_z,  
        G4double             wavelength ,

        G4int                flags_x, 
        G4int                flags_y, 
        G4int                flags_z, 
        G4int                flags_w
    )
{
     assert( m_g4hit_collector );
     m_g4hit_collector->collectPhoton(
         pos_x, 
         pos_y, 
         pos_z,
         time, 

         dir_x, 
         dir_y, 
         dir_z, 
         weight, 

         pol_x, 
         pol_y, 
         pol_z, 
         wavelength,

         flags_x,
         flags_y,
         flags_z,
         flags_w
     ) ;
}
 

void G4Opticks::setInputPhotons(const char* dir, const char* name, int repeat, const char* wavelength, int eventID )
{
    NPY<float>* input_photons = NPY<float>::load(dir, name) ; 
    setInputPhotons(input_photons, repeat, wavelength, eventID); 
}
void G4Opticks::setInputPhotons(const char* path, int repeat, const char* wavelength, int eventID )
{
    NPY<float>* input_photons = NPY<float>::load(path) ; 
    setInputPhotons(input_photons, repeat, wavelength, eventID); 
}


/**
G4Opticks::setInputPhotons
---------------------------

**/

void G4Opticks::setInputPhotons(NPY<float>* input_photons, int repeat, const char* wavelength, int eventID )
{
    LOG(info) 
        << " input_photons " << ( input_photons ? input_photons->getShapeString() : "-" )
        << " repeat " << repeat 
        << " wavelength " << wavelength
        << " eventID " << eventID
        ;

    if( input_photons == nullptr )
    {
        LOG(error) << " null input_photons, ignore " ; 
        return ; 
    }

    unsigned tagoffset = 0 ;   
    const OpticksGenstep* gs = OpticksGenstep::MakeInputPhotonCarrier(input_photons, tagoffset, repeat, wavelength, eventID );
    assert( m_genstep_collector ); 
    m_genstep_collector->collectTorchGenstep(gs);  
}



/**
G4Opticks::setSave
---------------------

Override the embedded commandline default, used with G4OpticksRecorder
where saving is the entire point of the exercise.

**/
void G4Opticks::setSave(bool save)
{
    if(m_ok)
    {
        m_ok->setSave(save); 
    }
    else
    {
       LOG(error) << "cannot setSave until after setGeometry " ; 
    }
}

