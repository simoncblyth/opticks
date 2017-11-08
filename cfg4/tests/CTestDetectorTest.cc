// ggv-;ggv-pmt-test --cdetector
// ggv-;ggv-pmt-test --cdetector --export --exportconfig /tmp/test.dae

#include <cassert>
#include "CFG4_BODY.hh"

#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksMode.hh"
#include "OpticksCfg.hh"

#include "GGeoTestConfig.hh"

#include "CG4.hh"
#include "CMaterialLib.hh"
#include "CTestDetector.hh"
#include "CTraverser.hh"

#include "G4VPhysicalVolume.hh"

#ifdef WITH_G4DAE
#include "G4DAEParser.hh"
#endif

#include "NBoundingBox.hpp"

#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "GGEO_LOG.hh"
#include "OKCORE_LOG.hh"
#include "OKGEO_LOG.hh"
#include "CFG4_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    //PLOG_(argc, argv)
    PLOG_COLOR(argc, argv)

    BRAP_LOG__ ; 
    NPY_LOG__ ; 
    GGEO_LOG__ ; 
    OKCORE_LOG__ ; 
    OKGEO_LOG__ ; 
    CFG4_LOG__ ; 

    LOG(info) << argv[0] ; 

    //const char* forced = "--test --apmtload " ;   // huh : why the --test ? that signifyies modify geometry 
    //  guess that the fail with the forced is because the default moified test bib geometry is not reversed
    const char* forced = NULL ; 

    Opticks ok(argc, argv, forced);
    ok.setModeOverride( OpticksMode::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4

    OpticksHub hub(&ok);

    CG4 g4(&hub);

    LOG(info) << "CG4 DONE" ; 
    CDetector* detector  = g4.getDetector();

    bool valid = detector->isValid();

    if(!valid)
    {
        LOG(error) << "Detector not valid " ;
        return 0 ; 
    } 



    detector->setVerbosity(2) ;

    CMaterialLib* mlib = detector->getMaterialLib() ;
    assert(mlib); 

    G4VPhysicalVolume* world_pv = detector->getTop();
    assert(world_pv);





    return 0 ; 
}
