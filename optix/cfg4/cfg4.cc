#include <cstdio>

#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4String.hh"
#include "G4UIExecutive.hh"

// cfg4-
#include "PhysicsList.hh"
#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"
#include "Recorder.hh"

// npy-
#include "TorchStepNPY.hpp"
#include "NLog.hpp"

//opticks-
#include "Opticks.hh"
#include "OpticksPhoton.h"
#include "OpticksCfg.hh"


class CfG4 
{
   public:
        CfG4();
        virtual ~CfG4();
   private:
        void init();
   public:
        void configure(int argc, char** argv);
        void propagate();
        void save();
   private:
        Opticks*              m_opticks ;  
        OpticksCfg<Opticks>*  m_cfg ;
        TorchStepNPY*         m_torch ; 
   private:
        DetectorConstruction* m_detector ; 
        Recorder*             m_recorder ; 
        G4RunManager*         m_runManager ;
   private:
        unsigned int          m_g4_nevt ; 
        unsigned int          m_g4_photons_per_event ; 
        unsigned int          m_num_photons ; 

};

inline CfG4::CfG4() 
   :
     m_opticks(NULL),
     m_cfg(NULL),
     m_torch(NULL),
     m_detector(NULL),
     m_recorder(NULL),
     m_runManager(NULL),
     m_g4_nevt(0),
     m_g4_photons_per_event(0),
     m_num_photons(0)
{
    init();
}

inline void CfG4::init()
{
    m_opticks = new Opticks();
    m_cfg = new OpticksCfg<Opticks>("opticks", m_opticks,false);
}

inline void CfG4::configure(int argc, char** argv)
{
    m_cfg->commandline(argc, argv);  

    unsigned int code = m_opticks->getSourceCode();
    assert(code == TORCH && "cfg4 only supports source type TORCH" );

    std::string typ = Opticks::SourceTypeLowercase(code);

    m_torch = m_opticks->makeSimpleTorchStep();
    m_torch->dump();

    m_num_photons = m_torch->getNumPhotons();
 
    std::string tag = m_cfg->getEventTag();
    std::string cat = m_cfg->getEventCat();

    unsigned int maxrec = m_cfg->getRecordMax();
    assert(maxrec == 10);


    m_g4_photons_per_event = m_cfg->getG4PhotonsPerEvent();
    assert( m_num_photons % m_g4_photons_per_event == 0 && "expecting num_photons to be exactly divisible by g4_photons_per_event" );

    m_g4_nevt = m_num_photons / m_g4_photons_per_event ; 

    LOG(info) << "CfG4::configure" 
              << " typ " << typ
              << " tag " << tag 
              << " cat " << cat
              << " m_g4_nevt " << m_g4_nevt 
              << " m_g4_photons_per_event " << m_g4_photons_per_event
              << " m_num_photons " << m_num_photons
              << " maxrec " << maxrec
              ;

    m_recorder = new Recorder(typ.c_str(),tag.c_str(),cat.c_str(),m_num_photons,maxrec, m_g4_photons_per_event); 
    if(strcmp(tag.c_str(), "-5") == 0)  m_recorder->setIncidentSphereSPolarized(true) ;

    m_detector  = new DetectorConstruction() ; 
    m_runManager = new G4RunManager;

    m_runManager->SetUserInitialization(new PhysicsList());
    m_runManager->SetUserInitialization(m_detector);
    m_runManager->SetUserInitialization(new ActionInitialization(m_recorder));
    m_runManager->Initialize();

    m_recorder->setCenterExtent(m_detector->getCenterExtent());
    m_recorder->setBoundaryDomain(m_detector->getBoundaryDomain());
}
inline void CfG4::propagate()
{
    LOG(info) << "CfG4::propagate"
              << " g4_nevt " << m_g4_nevt 
              << " m_g4_photons_per_event " << m_g4_photons_per_event
              << " num_photons " << m_num_photons 
              ; 
    m_runManager->BeamOn(m_g4_nevt);
}
inline void CfG4::save()
{
    m_recorder->save();
}
inline CfG4::~CfG4()
{
    delete m_runManager;
}



int main(int argc, char** argv)
{
    CfG4* app = new CfG4() ;

    app->configure(argc, argv);
    app->propagate();
    app->save();

    delete app ; 
    return 0 ; 
}
