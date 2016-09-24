#include <cassert>
// cfg4--;op --cgdmldetector --dbg

#include "CFG4_BODY.hh"

// okc-
#include "Opticks.hh"
#include "OpticksQuery.hh"
#include "OpticksCfg.hh"

// cfg4-
#include "CTestDetector.hh"
#include "CGDMLDetector.hh"
#include "CMaterialTable.hh"
#include "CBorderSurfaceTable.hh"

// g4-
#include "G4VPhysicalVolume.hh"

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#include "GGEO_LOG.hh"
#include "CFG4_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) << argv[0] ;

    CFG4_LOG__ ; 
    GGEO_LOG__ ; 

    Opticks ok(argc, argv);

    ok.configure();

    //OpticksCfg<Opticks>* m_cfg = m_opticks->getCfg();

    //m_cfg->commandline(argc, argv);  


    OpticksQuery* query = ok.getQuery();   // non-done inside Detector classes for transparent control/flexibility 

    CGDMLDetector* m_detector  = new CGDMLDetector(&ok, query) ; 

    ok.setIdPathOverride("$TMP");
    m_detector->saveBuffers();
    ok.setIdPathOverride(NULL);

    bool valid = m_detector->isValid();
    if(!valid)
    {
        LOG(error) << "CGDMLDetector not valid " ;
        return 0 ;  
    }


    m_detector->setVerbosity(2) ;

    NPY<float>* gtransforms = m_detector->getGlobalTransforms();
    gtransforms->save("$TMP/gdml.npy");

    unsigned int index = 3160 ; 

    glm::mat4 mg = m_detector->getGlobalTransform(index);
    glm::mat4 ml = m_detector->getLocalTransform(index);

    LOG(info) << " index " << index 
              << " pvname " << m_detector->getPVName(index) 
              << " global " << gformat(mg)
              << " local "  << gformat(ml)
              ;

    G4VPhysicalVolume* world_pv = m_detector->Construct();
    assert(world_pv);

    CMaterialTable mt ; 
    mt.dump("CGDMLDetectorTest CMaterialTable");

    CBorderSurfaceTable bst ; 
    bst.dump("CGDMLDetectorTest CBorderSurfaceTable");

    return 0 ; 
}
