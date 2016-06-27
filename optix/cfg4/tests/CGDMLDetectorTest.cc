// cfg4--;op --cgdmldetector --dbg

#include "CFG4_BODY.hh"

// okc-
#include "Opticks.hh"
#include "OpticksQuery.hh"
#include "OpticksCfg.hh"

// cfg4-
#include "CTestDetector.hh"
#include "CGDMLDetector.hh"

// g4-
#include "G4VPhysicalVolume.hh"

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#include "CFG4_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) << argv[0] ;

    CFG4_LOG__ ; 

    Opticks* m_opticks = new Opticks(argc, argv);

    OpticksCfg<Opticks>* m_cfg = m_opticks->getCfg();

    m_cfg->commandline(argc, argv);  

    OpticksQuery* query = m_opticks->getQuery();   // non-done inside Detector classes for transparent control/flexibility 

    CGDMLDetector* m_detector  = new CGDMLDetector(m_opticks, query) ; 


    bool valid = m_detector->isValid();
    if(!valid)
    {
        LOG(error) << "CGDMLDetector not valid " ;
        return 0 ;  
    }


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

    G4VPhysicalVolume* world_pv = m_detector->Construct();

    return 0 ; 
}
