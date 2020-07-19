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
#include "NPho.hpp"
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

#include "OpticksPhoton.h"
#include "OpticksGenstep.h"

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
#include "G4Track.hh"
#include "G4TransportationManager.hh"
#include "G4Version.hh"


#include "PLOG.hh"


const plog::Severity G4Opticks::LEVEL = PLOG::EnvLevel("G4Opticks", "DEBUG")  ;


G4Opticks* G4Opticks::fOpticks = NULL ;

//const char* G4Opticks::fEmbeddedCommandLine = " --gltf 3 --compute --save --embedded --natural --dbgtex --printenabled --pindex 0 --bouncemax 0"  ; 
//  --bouncemax 0 historical for checking generation   ??
//

//const char* G4Opticks::fEmbeddedCommandLine = " --gltf 3 --compute --save --embedded --natural --printenabled --pindex 0"  ; 
const char* G4Opticks::fEmbeddedCommandLine = " --gltf 3 --compute --save --embedded --natural --printenabled --pindex 0 --xanalytic"  ; 

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
    return Get(); 
}

G4Opticks* G4Opticks::Get()
{
    if (!fOpticks) fOpticks = new G4Opticks;
    return fOpticks ;
}

void G4Opticks::Initialize(const char* gdmlpath, bool standardize_geant4_materials)
{
    const G4VPhysicalVolume* world = CGDML::Parse(gdmlpath); 
    Initialize(world, standardize_geant4_materials); 
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
    m_hits_(NULL),
    m_hits(NULL),
    m_num_hits(0),
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
    m_genstep_collector->reset(); 
    m_gensteps = NULL ; 

    m_primary_collector->reset(); 

    m_g4hit_collector->reset(); 
    m_g4hit = NULL ; 

    m_g4photon_collector->reset(); 
    m_genphotons = NULL ; 
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


    const char* dbggdmlpath = ok->getDbgGDMLPath(); 
    if( dbggdmlpath != NULL )
    { 
        LOG(info) << "( CGDML" ;
        CGDML::Export( dbggdmlpath, top ); 
        LOG(info) << ") CGDML" ;
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

**/

int G4Opticks::propagateOpticalPhotons(G4int eventID) 
{
    LOG(LEVEL) << "[[" ; 

    m_gensteps = m_genstep_collector->getGensteps(); 
    m_gensteps->setArrayContentVersion(G4VERSION_NUMBER); 
    m_gensteps->setArrayContentIndex(eventID); 

    unsigned tagoffset = eventID ;  // tags are 1-based : so this will normally be the Geant4 eventID + 1
    const char* gspath = m_ok->getDirectGenstepPath(tagoffset); 
    LOG(info) << " saving gensteps to " << gspath ; 
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
        m_opmgr->propagate();     // GPU simulation is done in here 

        OpticksEvent* event = m_opmgr->getEvent(); 
        m_hits_ = event->getHitData()->clone() ; 
        m_hits = new NPho(m_hits_) ; 
        m_num_hits = m_hits->getNumPhotons() ; 

        // minimal g4 side instrumentation in "1st executable" 
        // do after propagate, so the event will have been created already
        m_g4hit = m_g4hit_collector->getPhoton();  
        m_g4evt = m_opmgr->getG4Event(); 
        m_g4evt->saveHitData( m_g4hit ) ; // pass thru to the dir, owned by m_g4hit_collector ?

        m_g4evt->saveSourceData( m_genphotons ) ; 


        m_opmgr->reset();   
        // clears OpticksEvent buffers,   excluding gensteps
        // clone any buffers to be retained before the reset
    }


    LOG(LEVEL) << "]] num_hits " << m_num_hits ; 
    return m_num_hits ;   
}

NPY<float>* G4Opticks::getHits() const 
{
    return m_hits_ ; 
}

void G4Opticks::getHit(
            unsigned i,
            G4ThreeVector* position,  
            G4double* time, 
            G4ThreeVector* direction, 
            G4double* weight,
            G4ThreeVector* polarization,  
            G4double* wavelength,
            G4int* flags_x,
            G4int* flags_y,
            G4int* flags_z,
            G4int* flags_w,
            G4bool* is_cerenkov, 
            G4bool* is_reemission
      ) const 
{
    assert( i < m_num_hits ); 

    glm::vec4 post = m_hits->getPositionTime(i);      
    position->set(double(post.x), double(post.y), double(post.z)); 
    *time = double(post.w) ; 

    glm::vec4 dirw = m_hits->getDirectionWeight(i);      
    direction->set(double(dirw.x), double(dirw.y), double(dirw.z)); 
    *weight = double(dirw.w) ; 

    glm::vec4 polw = m_hits->getPolarizationWavelength(i); 
    polarization->set(double(polw.x), double(polw.y), double(polw.z)); 
    *wavelength = double(polw.w);  

    glm::uvec4 flags = m_hits->getFlags(i);
    *flags_x = flags.x ; 
    *flags_y = flags.y ; 
    *flags_z = flags.z ; 
    *flags_w = flags.w ; 

    *is_cerenkov = (flags.w & CERENKOV) != 0 ; 
    *is_reemission = (flags.w & BULK_REEMIT) != 0 ; 

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




void G4Opticks::collectGenstep_G4Scintillation_1042(  
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
 
    collectScintillationStep(

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

}
 



/**
G4Opticks::collectScintillationStep
-------------------------------------

This method is tied to a particular implementation of 
scintillation and uses parameter names from that implementation.
The GPU generation in optixrap/cu/scintillationstep.h 
is tightly tied to this.

slowerRatio
slowTimeConstant
slowerTimeConstant
ScintillationTime
     will be one of slowTimeConstant or slowerTimeConstant  
     depending on whether have both slow and fast and on the scnt loop index 
scnt
     loop index, 1:fast 2:slow

**/



void G4Opticks::collectGenstep_DsG4Scintillation_r3971(  
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

    const G4DynamicParticle* aParticle = aTrack->GetDynamicParticle();
    const G4Material* aMaterial = aTrack->GetMaterial();

    G4double meanVelocity = (pPreStepPoint->GetVelocity()+pPostStepPoint->GetVelocity())/2. ; 
 
    collectScintillationStep(

         OpticksGenstep_DsG4Scintillation_r3971,        // (int)gentype       (0) 
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
         meanVelocity,                                  // MeanVelocity 

         scnt,                                          // (int) scnt    fast/slow loop index  (4)
         slowerRatio,                                   // slowerRatio
         slowTimeConstant,                              // slowTimeConstant
         slowerTimeConstant,                            // slowerTimeConstant

         ScintillationTime,                             //  ScintillationTime         (5)
         0,                                             //  formerly ScintillationIntegralMax : BUT THAT IS BAKED INTO REEMISSION TEXTURE
         0,                                             //  Other1
         0                                              //  Other2
    ) ;

}

void G4Opticks::collectScintillationStep
(
        G4int gentype,
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
    LOG(debug) << "[";
    m_genstep_collector->collectScintillationStep(
             gentype,
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
    LOG(debug) << "]";
}








void G4Opticks::collectGenstep_G4Cerenkov_1042(  
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

    const G4DynamicParticle* aParticle = aTrack->GetDynamicParticle();
    const G4Material* aMaterial = aTrack->GetMaterial();

    G4double preVelocity = pPreStepPoint->GetVelocity() ;
    G4double postVelocity = pPostStepPoint->GetVelocity() ; 
 
    collectCerenkovStep(

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

         betaInverse,
         pmin,
         pmax,
         maxCos,

         maxSin2,
         meanNumberOfPhotons1,
         meanNumberOfPhotons2,
         postVelocity
    ) ;
}




void G4Opticks::collectCerenkovStep
    (
        G4int                gentype, 
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
     LOG(debug) << "[" ; 
     m_genstep_collector->collectCerenkovStep(
                       gentype, 
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
     LOG(debug) << "]" ; 
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
 

