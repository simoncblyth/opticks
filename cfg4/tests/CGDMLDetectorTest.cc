#include <cassert>
// cfg4--;op --cgdmldetector --dbg

#include "CFG4_BODY.hh"

#include "Opticks.hh"     // okc-
#include "OpticksQuery.hh"
#include "OpticksCfg.hh"

#include "OpticksHub.hh"   // okg-

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

    OpticksHub hub(&ok);

    //OpticksCfg<Opticks>* m_cfg = m_opticks->getCfg();
    //m_cfg->commandline(argc, argv);  


    OpticksQuery* query = ok.getQuery();   // non-done inside Detector classes for transparent control/flexibility 

    CGDMLDetector* detector  = new CGDMLDetector(&hub, query) ; 

    ok.setIdPathOverride("$TMP");
    detector->saveBuffers();
    ok.setIdPathOverride(NULL);

    bool valid = detector->isValid();
    if(!valid)
    {
        LOG(error) << "CGDMLDetector not valid " ;
        return 0 ;  
    }


    detector->setVerbosity(2) ;

    NPY<float>* gtransforms = detector->getGlobalTransforms();
    gtransforms->save("$TMP/gdml.npy");

    unsigned int index = 3160 ; 

    glm::mat4 mg = detector->getGlobalTransform(index);
    glm::mat4 ml = detector->getLocalTransform(index);

    LOG(info) << " index " << index 
              << " pvname " << detector->getPVName(index) 
              << " global " << gformat(mg)
              << " local "  << gformat(ml)
              ;

    G4VPhysicalVolume* world_pv = detector->Construct();
    assert(world_pv);

    CMaterialTable mt ; 
    mt.dump("CGDMLDetectorTest CMaterialTable");

    CBorderSurfaceTable bst ; 
    bst.dump("CGDMLDetectorTest CBorderSurfaceTable");

    return 0 ; 
}
