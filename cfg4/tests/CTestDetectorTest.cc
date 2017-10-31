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
#include "CFG4_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv)

    BRAP_LOG__ ; 
    NPY_LOG__ ; 
    GGEO_LOG__ ; 
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

    CMaterialLib* clib = detector->getPropLib() ;
    assert(clib); 

    G4VPhysicalVolume* world_pv = detector->getTop();
    assert(world_pv);




/*
    // TODO: move below into CGeometry?

    bool expo = m_cfg->hasOpt("export");
    std::string expoconfig = m_cfg->getExportConfig();

    if(expo && expoconfig.size() > 0)
    { 
        const G4String path = expoconfig ; 

        LOG(info) << "export to " << expoconfig ; 

#ifdef WITH_G4DAE 
        G4DAEParser* g4dae = new G4DAEParser ;

        G4bool refs = true ;
        G4bool recreatePoly = false ; 
        G4int nodeIndex = -1 ;   // so World is volume 0 

        g4dae->Write(path, world_pv, refs, recreatePoly, nodeIndex );
#else
        LOG(warning) << " export requires WITH_G4DAE " ; 
#endif

    }
*/



    return 0 ; 
}
