#include <sstream>
#include <iostream>
#include <cstring>

#include "BOpticksKey.hh"

#include "CTraverser.hh"
#include "CCollector.hh"
#include "G4Opticks.hh"

#include "Opticks.hh"
#include "OpMgr.hh"

#include "GGeo.hh"
#include "X4PhysicalVolume.hh"

#include "G4Material.hh"
#include "G4Event.hh"
#include "G4TransportationManager.hh"

#include "PLOG.hh"

G4Opticks* G4Opticks::fOpticks = NULL ;


const char* G4Opticks::fEmbeddedCommandLine = " --gltf 3 --compute --save --embedded --natural " ; 

std::string G4Opticks::desc()
{
    std::stringstream ss ; 
    ss << "G4Opticks"
       << " ok " << m_ok 
       << " opmgr " << m_opmgr
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
    m_opmgr(NULL),
    m_traverser(NULL),
    m_collector(NULL),
    m_lookup(NULL)
{
    std::cout << "G4Opticks::G4Opticks" << std::endl ; 
    assert( fOpticks == NULL ); 
}


void G4Opticks::setGeometry(const G4VPhysicalVolume* world)
{
    m_world = world ; 
    m_ggeo = translateGeometry( world );
    m_ok = m_ggeo->getOpticks(); 

    NLookup* lookup = NULL ;  // TODO: come up with the lookup from GGeo/GBndLib without resorting to json maps m_lookup
    m_collector = new CCollector(lookup); 
}

GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
{
    const char* key = X4PhysicalVolume::Key(top) ; 
    BOpticksKey::SetKey(key);
    LOG(error) << " SetKey " << key  ;   
    Opticks* ok = new Opticks(0,0, fEmbeddedCommandLine);  // Opticks instanciation must be after BOpticksKey::SetKey
    GGeo* gg = new GGeo(ok) ;
    X4PhysicalVolume xtop(gg, top) ;   // <-- populates gg 
    return gg ; 
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
  






void G4Opticks::BeginOfRunAction(const G4Run* aRun) 
{
    checkGeometry();
    checkMaterials();
    setupPropagator();
}
void G4Opticks::EndOfRunAction(const G4Run* aRun) 
{
    checkGeometry();
}
void G4Opticks::BeginOfEventAction(const G4Event* evt) 
{
    LOG(info) << " BeginOfEventAction " << evt->GetEventID() ; 
}
void G4Opticks::EndOfEventAction(const G4Event* evt) 
{
    LOG(info) << " EndOfEventAction " << evt->GetEventID() ; 
    unsigned eventId = evt->GetEventID() ; 
    propagate(eventId);
}

void G4Opticks::checkGeometry()
{
    G4VPhysicalVolume* world_pv = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ;

    m_traverser = new CTraverser( m_ok, world_pv, NULL, NULL ); 
    LOG(info) 
        << " world_pv " << world_pv 
        << " traverser " << m_traverser 
        ; 

    m_traverser->setVerbosity(5);
    m_traverser->Traverse();     // both VolumeTree and Ancestor traverses
}

void G4Opticks::checkMaterials()
{
    const G4MaterialTable* mt = G4Material::GetMaterialTable();

    std::stringstream ff; 
    ff << "{" << std::endl;

    for (G4MaterialTable::const_iterator it=mt->begin(); it != mt->end(); ++it) 
    {
          G4Material* material = *it ; 
          const G4String& name = material->GetName() ; 
          size_t index = material->GetIndex() ; 

          m_mat_g[name] = index ;
        
          ff << '"' << name << '"' << ": " << index << ',' << std::endl;
    }
    ff << '"' << "ENDMAT" << '"' << ": 999999" << std::endl;
    ff << "}" << std::endl;

    std::string json_string = ff.str();
    const char* json_str = json_string.c_str();

    m_lookup = strdup(json_str);

    LOG(info) << m_lookup ; 
}


void G4Opticks::setupPropagator()
{
    // hmm this is using a pre-cached geometry : need to 
    // form geometry digest and check if it matches the current G4 context geometry 
    // and export if necessary 

    // m_opmgr->snap(); // take raytrace snapshot of geometry 

    //m_opmgr->setLookup(m_lookup);
}
void G4Opticks::propagate(int eventId)
{
    //std::stringstream ss;
    //ss << "/tmp/output-genstep-" << eventId << ".npy";
    //std::string name = ss.str();
    //m_opmgr->saveEmbeddedGensteps(name.c_str());

    //m_opmgr->propagate();
}

void G4Opticks::addGenstep( float* data, unsigned num_float ) 
{
    assert( num_float == 4*6 ); 
    m_opmgr->addGenstep(data, num_float);
}


