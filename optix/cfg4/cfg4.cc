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
#include "NLog.hpp"

//opticks-
#include "Opticks.hh"
#include "OpticksCfg.hh"

/*
#include <boost/lexical_cast.hpp>
int parse(char* arg)
{
   int iarg = 0 ;
   try{ 
        iarg = boost::lexical_cast<int>(arg) ;
    }   
    catch (const boost::bad_lexical_cast& e ) { 
        LOG(warning)  << "Caught bad lexical cast with error " << e.what() ;
    }   
    catch( ... ){
        LOG(warning) << "Unknown exception caught!" ;
    }
    return iarg ;   
}
*/


class CfG4 {
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
   private:
        DetectorConstruction* m_detector ; 
        Recorder*             m_recorder ; 
        G4RunManager*         m_runManager ;
   private:
        unsigned int          m_nevt ; 
        unsigned int          m_photons_per_event ; 
        unsigned int          m_nphotons ; 

};

inline CfG4::CfG4() 
   :
     m_opticks(NULL),
     m_cfg(NULL),
     m_detector(NULL),
     m_recorder(NULL),
     m_runManager(NULL),
     m_nevt(0),
     m_photons_per_event(0),
     m_nphotons(0)
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
 
    std::string tag = m_cfg->getEventTag();
    std::string cat = m_cfg->getEventCat();

    m_photons_per_event = m_cfg->getG4PhotonsPerEvent();

   // TODO: get rid of these, by pulling photon count from 
    // torchconfig and moving source check into OpticksCfg
    const char* typ = "torch" ;
    m_nevt = argc > 1 ? parse(argv[argc-1]) : 1 ;

    m_nphotons = m_nevt*m_photons_per_event ; 

    LOG(info) << "CfG4::configure" 
              << " tag " << tag 
              << " cat " << cat
              << " m_nevt " << m_nevt 
              << " m_photons_per_event " << m_photons_per_event
              << " m_nphotons " << m_nphotons
              ;

    m_recorder = new Recorder(typ,tag.c_str(),cat.c_str(),m_nphotons,10, m_photons_per_event); 
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
    m_runManager->BeamOn(m_nevt);
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
