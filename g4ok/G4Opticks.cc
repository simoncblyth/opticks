#include <sstream>
#include <iostream>
#include <cstring>

#include "BOpticksKey.hh"
#include "NLookup.hpp"
#include "NPY.hpp"

#include "CTraverser.hh"
#include "CMaterialTable.hh"
#include "CCollector.hh"
#include "CPrimaryCollector.hh"
#include "CGDML.hh"

#include "G4Opticks.hh"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpMgr.hh"

#include "GGeo.hh"
#include "GGeoGLTF.hh"
#include "GBndLib.hh"
#include "X4PhysicalVolume.hh"

#include "G4Material.hh"
#include "G4Event.hh"
#include "G4TransportationManager.hh"

#include "PLOG.hh"

G4Opticks* G4Opticks::fOpticks = NULL ;


const char* G4Opticks::fEmbeddedCommandLine = " --gltf 3 --compute --save --embedded --natural " ; 

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

G4Opticks::~G4Opticks()
{
    if (fOpticks)
    {
        delete fOpticks ; fOpticks = NULL ;
    }
}

G4Opticks::G4Opticks()
    :
    m_world(NULL),
    m_ggeo(NULL),
    m_ok(NULL),
    m_traverser(NULL),
    m_mtab(NULL),
    m_collector(NULL),
    m_primary_collector(NULL),
    m_lookup(NULL),
    m_opmgr(NULL),
    m_gensteps(NULL),
    m_hits(NULL)
{
    std::cout << "G4Opticks::G4Opticks" << std::endl ; 
    assert( fOpticks == NULL ); 
}


void G4Opticks::setGeometry(const G4VPhysicalVolume* world)
{
    m_world = world ; 
    m_ggeo = translateGeometry( world );
    m_blib = m_ggeo->getBndLib();  
    m_ok = m_ggeo->getOpticks(); 

    const char* prefix = NULL ; 
    m_mtab = new CMaterialTable(prefix); 

    setupMaterialLookup();
    m_collector = new CCollector(m_lookup);   // <-- CG4 holds an instance too : and they are singletons, so should not use G4Opticks and CG4 together
    m_primary_collector = new CPrimaryCollector ; 

    // OpMgr instanciates OpticksHub which adopts the pre-existing m_ggeo instance just translated
    m_opmgr = new OpMgr(m_ok) ;   
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

void G4Opticks::setupMaterialLookup()
{
    const std::map<std::string, unsigned>& A = m_mtab->getMaterialMap() ;
    const std::map<std::string, unsigned>& B = m_blib->getMaterialLineMapConst() ;
 
    m_lookup = new NLookup ; 
    m_lookup->setA(A,"","CMaterialTable");
    m_lookup->setB(B,"","GBndLib");    // shortname eg "GdDopedLS" to material line mapping 
    m_lookup->close(); 
}

int G4Opticks::propagateOpticalPhotons() 
{
    m_gensteps = m_collector->getGensteps(); 
    m_opmgr->setGensteps(m_gensteps);      
    m_opmgr->propagate();

    OpticksEvent* event = m_opmgr->getEvent(); 
    m_hits = event->getHitData()->clone() ; 

    m_opmgr->reset();   
    // clears OpticksEvent buffers,
    // clone any buffers to be retained before the reset

    return m_hits ? m_hits->getNumItems() : -1 ;   
}

NPY<float>* G4Opticks::getHits() const 
{
    return m_hits ; 
}


void G4Opticks::collectPrimaries(const G4Event* event)
{
    m_primary_collector->collectPrimaries(event); 

    //const char* path = "$TMP/G4Opticks/collectPrimaries.npy" ;
    const char* path = m_ok->getPrimariesPath(); 

    LOG(info) << " saving to " << path ; 
    m_primary_collector->save(path); 
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
    m_collector->collectCerenkovStep(
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
  








