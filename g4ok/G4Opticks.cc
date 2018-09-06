#include <sstream>
#include <iostream>
#include <cstring>

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


const char* G4Opticks::fEmbeddedCommandLine = " --gltf 3 --compute --save --embedded --natural --dbgtex --printenabled --pindex 0 --bouncemax 0"  ; 

std::string G4Opticks::desc() const 
{

    BOpticksKey* key = m_ok ? m_ok->getKey() : NULL ; 

    std::stringstream ss ; 
    ss << "G4Opticks"
       << " ok " << m_ok 
       << " opmgr " << m_opmgr
       << std::endl 
       << ( key ? key->desc() : "NULL-key?" )
       << std::endl
       << ( m_ok ? m_ok->getIdPath() : "-" )
       << std::endl
       ;
    return ss.str() ; 
}


G4Opticks* G4Opticks::GetOpticks()
{
    if (!fOpticks) fOpticks = new G4Opticks;
    return fOpticks ;
}

void G4Opticks::Finalize()
{
    delete fOpticks ; 
    fOpticks = NULL ;
}

G4Opticks::~G4Opticks()
{
    CAlignEngine::Finalize() ;
}

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
    std::cout << "G4Opticks::G4Opticks" << std::endl ; 
    assert( fOpticks == NULL ); 
}


void G4Opticks::setGeometry(const G4VPhysicalVolume* world, bool standardize_geant4_materials)
{
    LOG(fatal) << "[[[" ; 

    GGeo* ggeo = translateGeometry( world ) ;

    if( standardize_geant4_materials )
    {
        standardizeGeant4MaterialProperties();
    }

    m_world = world ; 
    m_ggeo = ggeo ;
    m_blib = m_ggeo->getBndLib();  
    m_ok = m_ggeo->getOpticks(); 

    const char* prefix = NULL ; 
    m_mtab = new CMaterialTable(prefix); 

    setupMaterialLookup();
    m_genstep_collector = new CGenstepCollector(m_lookup);   // <-- CG4 holds an instance too : and they are singletons, so should not use G4Opticks and CG4 together
    m_primary_collector = new CPrimaryCollector ; 
    m_g4hit_collector = new CPhotonCollector ; 
    m_g4photon_collector = new C4PhotonCollector ; 

    CAlignEngine::Initialize(m_ok->getIdPath()) ;

    // OpMgr instanciates OpticksHub which adopts the pre-existing m_ggeo instance just translated
    m_opmgr = new OpMgr(m_ok) ;   

    LOG(fatal) << "]]]" ; 
}

GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
{
    const char* keyspec = X4PhysicalVolume::Key(top) ; 
    BOpticksKey::SetKey(keyspec);
    LOG(error) << " SetKey " << keyspec  ;   

    Opticks* ok = new Opticks(0,0, fEmbeddedCommandLine);  // Opticks instanciation must be after BOpticksKey::SetKey

    const char* gdmlpath = ok->getGDMLPath();   // inside geocache, not SrcGDMLPath from opticksdata
    CGDML::Export( gdmlpath, top ); 

    GGeo* gg = new GGeo(ok) ;
    X4PhysicalVolume xtop(gg, top) ;   // <-- populates gg 
    gg->postDirectTranslation(); 

    int root = 0 ; 
    const char* gltfpath = ok->getGLTFPath();   // inside geocache
    GGeoGLTF::Save(gg, gltfpath, root );

    return gg ; 
}


/**
G4Opticks::standardizeGeant4MaterialProperties
-----------------------------------------------

Standardize G4 material properties to use the Opticks standard domain 

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

TODO: relocate direct events inside the geocache ? 
      and place these direct gensteps and genphotons 
      within the OpticksEvent directory 


**/

int G4Opticks::propagateOpticalPhotons() 
{
    m_gensteps = m_genstep_collector->getGensteps(); 
    const char* gspath = m_ok->getDirectGenstepPath(); 

    LOG(info) << " saving gensteps to " << gspath ; 
    m_gensteps->setArrayContentVersion(G4VERSION_NUMBER); 
    m_gensteps->save(gspath); 

    // initial generated photons before propagation 
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
        // do after propagate, so the event will be created
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


void G4Opticks::collectPrimaries(const G4Event* event)
{
    m_primary_collector->collectPrimaries(event); 

    const char* path = m_ok->getPrimariesPath(); 

    LOG(info) << " saving to " << path ; 
    m_primary_collector->save(path); 
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
        G4double             meanVelocity, 

        G4double             betaInverse,
        G4double             pmin,
        G4double             pmax,
        G4double             maxCos,

        G4double             maxSin2,
        G4double             meanNumberOfPhotons1,
        G4double             meanNumberOfPhotons2,
        G4double             spare2
    )
{
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
                       meanVelocity,

                       betaInverse,
                       pmin,
                       pmax,
                       maxCos,

                       maxSin2,
                       meanNumberOfPhotons1,
                       meanNumberOfPhotons2,
                       spare2
                       ) ;
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
 

