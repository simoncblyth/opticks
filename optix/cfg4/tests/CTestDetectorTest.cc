// ggv-;ggv-pmt-test --cdetector
// ggv-;ggv-pmt-test --cdetector --export --exportconfig /tmp/test.dae

#include "CFG4_BODY.hh"

#include "Opticks.hh"
#include "OpticksCfg.hh"

#include "GGeoTestConfig.hh"

#include "CPropLib.hh"
#include "CTestDetector.hh"
#include "CTraverser.hh"

#include "G4VPhysicalVolume.hh"

#ifdef WITH_G4DAE
#include "G4DAEParser.hh"
#endif

#include "NBoundingBox.hpp"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv)

    Opticks* m_opticks = new Opticks(argc, argv);
    m_opticks->setMode( Opticks::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4

    OpticksCfg<Opticks>* m_cfg = m_opticks->getCfg();
    m_cfg->commandline(argc, argv);  


    std::string testconfig = m_cfg->getTestConfig();

    GGeoTestConfig* m_testconfig = new GGeoTestConfig( testconfig.empty() ? NULL : testconfig.c_str() );

    CTestDetector* m_detector  = new CTestDetector(m_opticks, m_testconfig) ; 

    m_detector->setVerbosity(2) ;

    CPropLib* clib = m_detector->getPropLib() ;

    G4VPhysicalVolume* world_pv = m_detector->getTop();

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

    return 0 ; 
}
