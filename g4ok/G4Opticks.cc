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

#include "SSys.hh"
#include "BOpticksKey.hh"
#include "NLookup.hpp"
#include "NPY.hpp"

#include "CTraverser.hh"
#include "CMaterialTable.hh"
#include "CGenstepCollector.hh"
#include "CPrimaryCollector.hh"
#include "CPhotonCollector.hh"
#include "C4PhotonCollector.hh"
#include "CAlignEngine.hh"
#include "CGDML.hh"
#include "C4FPEDetection.hh"


#include "G4Opticks.hh"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpMgr.hh"

#include "GGeo.hh"
#include "GMaterialLib.hh"
#include "GGeoGLTF.hh"
#include "GBndLib.hh"

#include "X4PhysicalVolume.hh"
#include "X4MaterialLib.hh"

#include "G4Material.hh"
#include "G4Event.hh"
#include "G4TransportationManager.hh"
#include "G4Version.hh"

#include "PLOG.hh"

G4Opticks* G4Opticks::fOpticks = NULL ;

//const char* G4Opticks::fEmbeddedCommandLine = " --gltf 3 --compute --save --embedded --natural --dbgtex --printenabled --pindex 0 --bouncemax 0"  ; 
//  --bouncemax 0 historical for checking generation   ??
//

const char* G4Opticks::fEmbeddedCommandLine = " --gltf 3 --compute --save --embedded --natural --printenabled --pindex 0"  ; 

std::string G4Opticks::EmbeddedCommandLine(const char* extra)
{
    std::stringstream ss ; 
    ss << fEmbeddedCommandLine << " " ;
    if(extra) ss << extra ; 
    return ss.str();  
}


std::string G4Opticks::desc() const 
{

    std::stringstream ss ; 
    ss << "G4Opticks.desc"
       << " ok " << m_ok 
       << " opmgr " << m_opmgr
       << std::endl 
       << ( m_ok ? m_ok->desc() : "-" ) 
       << std::endl 
       ;
    return ss.str() ; 
}


G4Opticks* G4Opticks::GetOpticks()
{
    if (!fOpticks) fOpticks = new G4Opticks;
    return fOpticks ;
}


void G4Opticks::Initialize(const G4VPhysicalVolume* world, bool standardize_geant4_materials)
{
    G4Opticks* g4ok = GetOpticks(); 
    g4ok->setGeometry(world, standardize_geant4_materials) ; 
}

void G4Opticks::Finalize()
{
    LOG(info) << G4Opticks::GetOpticks()->desc();
    delete fOpticks ; 
    fOpticks = NULL ;
}

G4Opticks::~G4Opticks()
{
    CAlignEngine::Finalize() ;
}

/**
G4Opticks::G4Opticks
----------------------

NB no OpticksHub, this is trying to be minimal 

**/

G4Opticks::G4Opticks()
    :
    m_world(NULL),
    m_ggeo(NULL),
    m_ok(NULL),
    m_traverser(NULL),
    m_mtab(NULL),
    m_genstep_collector(NULL),
    m_primary_collector(NULL),
    m_lookup(NULL),
    m_opmgr(NULL),
    m_gensteps(NULL),
    m_genphotons(NULL),
    m_hits(NULL),
    m_g4hit_collector(NULL),
    m_g4photon_collector(NULL),
    m_genstep_idx(0),
    m_g4evt(NULL),
    m_g4hit(NULL),
    m_gpu_propagate(true)
{
    assert( fOpticks == NULL ); 
    LOG(info) << "ctor : DISABLE FPE detection : as it breaks OptiX launches" ; 
    C4FPEDetection::InvalidOperationDetection_Disable();  // see notes/issues/OKG4Test_prelaunch_FPE_causing_fail.rst
}


/**
G4Opticks::setGeometry
------------------------



**/


void G4Opticks::setGeometry(const G4VPhysicalVolume* world, bool standardize_geant4_materials)
{
    LOG(fatal) << "[[[" ; 


    LOG(fatal) << "( translateGeometry " ; 
    GGeo* ggeo = translateGeometry( world ) ;
    LOG(fatal) << ") translateGeometry " ; 

    if( standardize_geant4_materials )
    {
        LOG(fatal) << "( standardizeGeant4MaterialProperties " ; 
        standardizeGeant4MaterialProperties();
        LOG(fatal) << ") standardizeGeant4MaterialProperties " ; 
    }

    m_world = world ; 
    m_ggeo = ggeo ;
    m_blib = m_ggeo->getBndLib();  
    m_ok = m_ggeo->getOpticks(); 

    LOG(fatal) << "( createCollectors " ; 
    createCollectors(); 
    LOG(fatal) << ") createCollectors " ; 

    //CAlignEngine::Initialize(m_ok->getIdPath()) ;

    // OpMgr instanciates OpticksHub which adopts the pre-existing m_ggeo instance just translated
    LOG(fatal) << "( OpMgr " ; 
    m_opmgr = new OpMgr(m_ok) ;   
    LOG(fatal) << ") OpMgr " ; 

    LOG(fatal) << "]]]" ; 
}


/**
G4Opticks::translateGeometry
------------------------------

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
    LOG(error) << "SetKey [" << keyspec << "]"  ;   
    BOpticksKey::SetKey(keyspec);
    LOG(verbose) << ") key" ;

    const char* g4opticks_debug = SSys::getenvvar("G4OPTICKS_DEBUG") ; 
    std::string ecl = EmbeddedCommandLine(g4opticks_debug) ; 
    LOG(info) << "EmbeddedCommandLine : [" << ecl << "]" ; 

    LOG(info) << "( Opticks" ;
    Opticks* ok = new Opticks(0,0, ecl.c_str() );  // Opticks instanciation must be after BOpticksKey::SetKey
    ok->configure();       // parses args and does resource setup
 
    const char* idpath = ok->getIdPath(); 
    assert(idpath);
    LOG(info) << ") Opticks " << idpath ;

    /*
    cannot do this with shared geocache due to permissions 

    const char* gdmlpath = ok->getGDMLPath();   // inside geocache, not SrcGDMLPath from opticksdata
    LOG(info) << "( CGDML" ;
    CGDML::Export( gdmlpath, top ); 
    LOG(info) << ") CGDML" ;
    */

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


    /*
    again permissions prevents this
    
    int root = 0 ; 
    const char* gltfpath = ok->getGLTFPath();   // inside geocache
    LOG(info) << "( gltf " ;
    GGeoGLTF::Save(gg, gltfpath, root );
    LOG(info) << ") gltf " ;

    */


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
    G4MaterialTable* mtab = G4Material::GetMaterialTable();   
    const GMaterialLib* mlib = GMaterialLib::GetInstance(); 
    X4MaterialLib::Standardize( mtab, mlib ) ;  
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


void G4Opticks::createCollectors()
{
    const char* prefix = NULL ; 
    m_mtab = new CMaterialTable(prefix); 

    setupMaterialLookup();
    m_genstep_collector = new CGenstepCollector(m_lookup);   // <-- CG4 holds an instance too : and they are singletons, so should not use G4Opticks and CG4 together
    m_primary_collector = new CPrimaryCollector ; 
    m_g4hit_collector = new CPhotonCollector ; 
    m_g4photon_collector = new C4PhotonCollector ; 
}













unsigned G4Opticks::getNumPhotons() const 
{
    return m_genstep_collector->getNumPhotons()  ; 
}
unsigned G4Opticks::getNumGensteps() const 
{
    return m_genstep_collector->getNumGensteps()  ; 
}


void G4Opticks::setAlignIndex(int align_idx) const 
{
    CAlignEngine::SetSequenceIndex(align_idx); 
}

/**
G4Opticks::propagateOpticalPhotons
-----------------------------------

Invoked from EventAction::EndOfEventAction

TODO: relocate direct events inside the geocache ? 
      and place these direct gensteps and genphotons 
      within the OpticksEvent directory 

      done already ?

**/

int G4Opticks::propagateOpticalPhotons() 
{
    m_gensteps = m_genstep_collector->getGensteps(); 
    const char* gspath = m_ok->getDirectGenstepPath(); 

    LOG(info) << " saving gensteps to " << gspath ; 
    m_gensteps->setArrayContentVersion(G4VERSION_NUMBER); 
    m_gensteps->save(gspath); 

    // initial generated photons before propagation 
    // CPU genphotons needed only while validating 
    m_genphotons = m_g4photon_collector->getPhoton(); 
    m_genphotons->setArrayContentVersion(G4VERSION_NUMBER); 

    //const char* phpath = m_ok->getDirectPhotonsPath(); 
    //m_genphotons->save(phpath); 

   
    if(m_gpu_propagate)
    {
        m_opmgr->setGensteps(m_gensteps);      
        m_opmgr->propagate();

        OpticksEvent* event = m_opmgr->getEvent(); 
        m_hits = event->getHitData()->clone() ; 

        // minimal g4 side instrumentation in "1st executable" 
        // do after propagate, so the event will have been created already
        m_g4hit = m_g4hit_collector->getPhoton();  
        m_g4evt = m_opmgr->getG4Event(); 
        m_g4evt->saveHitData( m_g4hit ) ; // pass thru to the dir, owned by m_g4hit_collector ?

        m_g4evt->saveSourceData( m_genphotons ) ; 


        m_opmgr->reset();   
        // clears OpticksEvent buffers,
        // clone any buffers to be retained before the reset
    }

    return m_hits ? m_hits->getNumItems() : -1 ;   
}

NPY<float>* G4Opticks::getHits() const 
{
    return m_hits ; 
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

void G4Opticks::collectScintillationStep
(
        G4int id,
        G4int parentId,
        G4int materialId,
        G4int numPhotons,

        G4double x0_x,
        G4double x0_y,
        G4double x0_z,
        G4double t0,

        G4double deltaPosition_x,
        G4double deltaPosition_y,
        G4double deltaPosition_z,
        G4double stepLength,

        G4int pdgCode,
        G4double pdgCharge,
        G4double weight,
        G4double meanVelocity,

        G4int scntId,
        G4double slowerRatio,
        G4double slowTimeConstant,
        G4double slowerTimeConstant,

        G4double scintillationTime,
        G4double scintillationIntegrationMax,
        G4double spare1 = 0,
        G4double spare2 = 0
        ) {
    LOG(info) << "[";
    m_genstep_collector->collectScintillationStep(
             id,
             parentId,
             materialId,
             numPhotons,

             x0_x,
             x0_y,
             x0_z,
             t0,

             deltaPosition_x,
             deltaPosition_y,
             deltaPosition_z,
             stepLength,

             pdgCode,
             pdgCharge,
             weight,
             meanVelocity,

             scntId,
             slowerRatio,
             slowTimeConstant,
             slowerTimeConstant,

             scintillationTime,
             scintillationIntegrationMax,
             spare1,
             spare2
            );
    LOG(info) << "]";
}



void G4Opticks::collectCerenkovStep
    (
        G4int                id, 
        G4int                parentId,
        G4int                materialId,
        G4int                numPhotons,
    
        G4double             x0_x,  
        G4double             x0_y,  
        G4double             x0_z,  
        G4double             t0, 

        G4double             deltaPosition_x, 
        G4double             deltaPosition_y, 
        G4double             deltaPosition_z, 
        G4double             stepLength, 

        G4int                pdgCode, 
        G4double             pdgCharge, 
        G4double             weight, 
        G4double             preVelocity,     

        G4double             betaInverse,
        G4double             pmin,
        G4double             pmax,
        G4double             maxCos,

        G4double             maxSin2,
        G4double             meanNumberOfPhotons1,
        G4double             meanNumberOfPhotons2,
        G4double             postVelocity
    )
{
     LOG(info) << "[" ; 
     m_genstep_collector->collectCerenkovStep(
                       id, 
                       parentId,
                       materialId,
                       numPhotons,

                       x0_x,
                       x0_y,
                       x0_z,
                       t0,

                       deltaPosition_x,
                       deltaPosition_y,
                       deltaPosition_z,
                       stepLength,
 
                       pdgCode,
                       pdgCharge,
                       weight,
                       preVelocity,

                       betaInverse,
                       pmin,
                       pmax,
                       maxCos,

                       maxSin2,
                       meanNumberOfPhotons1,
                       meanNumberOfPhotons2,
                       postVelocity
                       ) ;
     LOG(info) << "]" ; 
}
  



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
 

