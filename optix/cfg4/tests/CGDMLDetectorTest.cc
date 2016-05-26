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

// npy-
#include "NLog.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

int main(int argc, char** argv)
{
    Opticks* m_opticks = new Opticks(argc, argv, "CGDMLDetectorTest.log");

    GCache* m_cache = new GCache(m_opticks);

    OpticksCfg<Opticks>* m_cfg = m_opticks->getCfg();

    m_cfg->commandline(argc, argv);  


    CGDMLDetector* m_detector  = new CGDMLDetector(m_cache) ; 

    m_detector->setVerbosity(2) ;

    NPY<float>* gtransforms = m_detector->getGlobalTransforms();
    gtransforms->save("/tmp/gdml.npy");


    unsigned int index = 3160 ; 

    glm::mat4 mg = m_detector->getGlobalTransform(index);
    glm::mat4 ml = m_detector->getLocalTransform(index);

    LOG(info) << " index " << index 
              << " pvname " << m_detector->getPVName(index) 
              << " global " << gformat(mg)
              << " local "  << gformat(ml)
              ;

    print(mg, "global");
    print(ml, "local");


    G4VPhysicalVolume* world_pv = m_detector->Construct();


    return 0 ; 
}
