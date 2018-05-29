#include <iostream>
#include <cstring>

#include "G4OpticksManager.hh"

#include "OpMgr.hh"

#include "G4Material.hh"
#include "G4Event.hh"
#include "G4TransportationManager.hh"

#include "PLOG.hh"

//G4ThreadLocal 
G4OpticksManager* G4OpticksManager::fOpticksManager = NULL ;


G4OpticksManager::G4OpticksManager()
   :
   m_opmgr(NULL),
   m_lookup(NULL)
{
    std::cout << "G4OpticksManager::G4OpticksManager" << std::endl ; 
    assert( fOpticksManager == NULL ); 
}

G4OpticksManager* G4OpticksManager::GetOpticksManager()
{
   if (!fOpticksManager)
   {
       fOpticksManager = new G4OpticksManager;
   }  
   return fOpticksManager ;
}

G4OpticksManager::~G4OpticksManager()
{
   if (fOpticksManager)
   {
       delete fOpticksManager ; fOpticksManager = NULL ;
   }
}



void G4OpticksManager::BeginOfRunAction(const G4Run* aRun) 
{
    checkGeometry();
    checkMaterials();
    setupPropagator();
}
void G4OpticksManager::EndOfRunAction(const G4Run* aRun) 
{
    checkGeometry();
}
void G4OpticksManager::BeginOfEventAction(const G4Event* evt) 
{
    LOG(info) << " BeginOfEventAction " << evt->GetEventID() ; 

}
void G4OpticksManager::EndOfEventAction(const G4Event* evt) 
{
    LOG(info) << " EndOfEventAction " << evt->GetEventID() ; 
    unsigned eventId = evt->GetEventID() ; 
    propagate(eventId);
}





void G4OpticksManager::checkGeometry()
{
    G4VPhysicalVolume* world_pv = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ;
    LOG(info) << "world_pv " << world_pv ; 
}

void G4OpticksManager::checkMaterials()
{
    // TODO: review the need for material mapping 

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



void G4OpticksManager::setupPropagator()
{
    // construct opmgr
    // static const char* extracmd = " --gltf 3 --tracer --compute --save --embedded ";
    static const char* extracmd = " --gltf 3 --compute --save --embedded --natural ";
    m_opmgr = new OpMgr(0, 0, extracmd);

    // hmm this is using a pre-cached geometry : need to 
    // form geometry digest and check if it matches the current G4 context geometry 
    // and export if necessary 

    // m_opmgr->snap(); // take raytrace snapshot of geometry 

    m_opmgr->setLookup(m_lookup);

}
void G4OpticksManager::propagate(int eventId)
{
    //std::stringstream ss;
    //ss << "/tmp/output-genstep-" << eventId << ".npy";
    //std::string name = ss.str();
    //m_opmgr->saveEmbeddedGensteps(name.c_str());

    m_opmgr->propagate();
}


void G4OpticksManager::addGenstep( float* data, unsigned num_float ) 
{
    m_opmgr->addGenstep(data, num_float);
}



