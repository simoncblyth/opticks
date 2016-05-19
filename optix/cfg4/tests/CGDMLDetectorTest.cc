// cfg4--;op --cgdmldetector --dbg


// optickscore-
#include "Opticks.hh"
#include "OpticksCfg.hh"

// ggeo-
#include "GCache.hh"

// cfg4-
#include "CDetector.hh"
#include "CTraverser.hh"
#include "CGDMLDetector.hh"

// g4-
#include "G4VPhysicalVolume.hh"


#include "NLog.hpp"

int main(int argc, char** argv)
{
    LOG(info) << "klop " ;

    Opticks* m_opticks = new Opticks(argc, argv, "CGDMLDetectorTest.log");

    GCache* m_cache = new GCache(m_opticks);

    OpticksCfg<Opticks>* m_cfg = m_opticks->getCfg();

    m_cfg->commandline(argc, argv);  


    CGDMLDetector* m_detector  = new CGDMLDetector(m_cache) ; 

    m_detector->setVerbosity(2) ;


    G4VPhysicalVolume* world_pv = m_detector->Construct();


    CTraverser* m_traverser = new CTraverser(world_pv) ;

    m_traverser->Traverse(); 

    m_traverser->createGroupVel(); 

    m_traverser->setVerbosity(1); 

    m_traverser->dumpMaterials(); 



    return 0 ; 
}
