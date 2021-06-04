#include "PLOG.hh"
#include "G4OpticksRecorder.hh"

/**


Need to operate at the same dependency level as G4Opticks 
(using Opticks, GGeo but not OpticksHub). 
Actually OpticksHub is there, hidden inside OpMgr but 
still the higher level cfg4 classes look too entangled
to be used easily. 

Lower level cfg4 classes are directly useful, eg::

    CMaterialBridge : just need access to GMaterialLib to boot this 

Higher level classes are less relevant::

    CGeometry 


Can also make CFG4 classes more reusable by replaces use of 
high level instances like CG4 with their constituents that are really needed.

**/


#include "GGeo.hh"
#include "CMaterialBridge.hh"
#include "CManager.hh"

G4OpticksRecorder* G4OpticksRecorder::fInstance = NULL ;
G4OpticksRecorder* G4OpticksRecorder::Get(){ return fInstance ;  }  // static
const plog::Severity G4OpticksRecorder::LEVEL = PLOG::EnvLevel("G4OpticksRecorder", "DEBUG")  ;

G4OpticksRecorder::~G4OpticksRecorder(){ LOG(LEVEL);  }
G4OpticksRecorder::G4OpticksRecorder()
    :
    m_ggeo(nullptr),
    m_ok(nullptr),
    m_material_bridge(nullptr),
    m_manager(nullptr)
{
    LOG(LEVEL); 
    assert( fInstance == NULL ); 
    fInstance = this ; 
}


/**
G4OpticksRecorder::setGeometry
---------------------------------

Invoked by G4Opticks::setGeometry
**/

void G4OpticksRecorder::setGeometry(const GGeo* ggeo_)
{
    m_ggeo = ggeo_ ; 
    m_ok = m_ggeo->getOpticks(); 
    m_material_bridge = new CMaterialBridge(m_ggeo->getMaterialLib()) ; 
    m_manager = new CManager(m_ok);
    m_manager->setMaterialBridge(m_material_bridge); 
  
    LOG(LEVEL); 
} 


void G4OpticksRecorder::BeginOfRunAction(const G4Run* run)
{
    LOG(LEVEL); 
    m_manager->BeginOfRunAction(run); 
}
void G4OpticksRecorder::EndOfRunAction(const G4Run* run)
{
    LOG(LEVEL); 
    m_manager->EndOfRunAction(run); 
}


void G4OpticksRecorder::BeginOfEventAction(const G4Event* event)
{
    LOG(LEVEL); 
    m_manager->BeginOfEventAction(event); 
}
void G4OpticksRecorder::EndOfEventAction(const G4Event* event)
{
    LOG(LEVEL); 
    m_manager->EndOfEventAction(event); 
}


void G4OpticksRecorder::PreUserTrackingAction(const G4Track* track)
{
    LOG(LEVEL); 
    m_manager->PreUserTrackingAction(track); 
}
void G4OpticksRecorder::PostUserTrackingAction(const G4Track* track)
{
    LOG(LEVEL); 
    m_manager->PostUserTrackingAction(track); 
}

void G4OpticksRecorder::UserSteppingAction(const G4Step* step)
{
    LOG(LEVEL); 
    m_manager->UserSteppingAction(step); 
}

