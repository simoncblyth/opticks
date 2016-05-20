// cfg4--;op --cgdmldetector --dbg

// optickscore-
#include "Opticks.hh"
#include "OpticksCfg.hh"

// ggeo-
#include "GCache.hh"

// cfg4-
#include "CTestDetector.hh"
#include "CGDMLDetector.hh"

// g4-
#include "G4VPhysicalVolume.hh"


#include "NLog.hpp"

int main(int argc, char** argv)
{
    Opticks* m_opticks = new Opticks(argc, argv, "CGDMLDetectorTest.log");

    GCache* m_cache = new GCache(m_opticks);

    OpticksCfg<Opticks>* m_cfg = m_opticks->getCfg();

    m_cfg->commandline(argc, argv);  


    CGDMLDetector* m_detector  = new CGDMLDetector(m_cache) ; 

    m_detector->setVerbosity(2) ;

    G4VPhysicalVolume* world_pv = m_detector->Construct();


    return 0 ; 
}
