// ggv-;ggv-pmt-test --cdetector
// ggv-;ggv-pmt-test --cdetector --export --exportconfig /tmp/test.dae

#include "Opticks.hh"
#include "OpticksCfg.hh"

#include "GCache.hh"
#include "GGeoTestConfig.hh"

#include "CPropLib.hh"
#include "CDetector.hh"
#include "CTraverser.hh"

#include "G4VPhysicalVolume.hh"

#include "G4DAEParser.hh"


#include "NLog.hpp"

int main(int argc, char** argv)
{
    LOG(info) << "klop " ;

    Opticks* m_opticks = new Opticks(argc, argv, "CPropLibTest.log");
    m_opticks->setMode( Opticks::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4
    GCache* m_cache = new GCache(m_opticks);
    OpticksCfg<Opticks>* m_cfg = m_opticks->getCfg();
    m_cfg->commandline(argc, argv);  

    assert( m_cfg->hasOpt("test") && m_opticks->getSourceCode() == TORCH && "cfg4 only supports source type TORCH with test geometries" );

    std::string testconfig = m_cfg->getTestConfig();

    GGeoTestConfig* m_testconfig = new GGeoTestConfig( testconfig.empty() ? NULL : testconfig.c_str() );

    CDetector* m_detector  = new CDetector(m_cache, m_testconfig) ; 

    m_detector->setVerbosity(2) ;


    CPropLib* clib = m_detector->getPropLib() ;

    G4VPhysicalVolume* world_pv = m_detector->Construct();

    clib->dumpMaterials();

    CTraverser* m_traverser = new CTraverser(world_pv) ;

    m_traverser->Traverse(); 

    m_traverser->createGroupVel(); 

    m_traverser->setVerbosity(10); 

    m_traverser->dumpMaterials(); 



    bool expo = m_cfg->hasOpt("export");
    std::string expoconfig = m_cfg->getExportConfig();

    if(expo && expoconfig.size() > 0)
    {  
        const G4String path = expoconfig ; 

        LOG(info) << "export to " << expoconfig ; 

        G4DAEParser* g4dae = new G4DAEParser ;

        G4bool refs = true ;
        G4bool recreatePoly = false ; 
        G4int nodeIndex = -1 ;   // so World is volume 0 

        g4dae->Write(path, world_pv, refs, recreatePoly, nodeIndex );

    }


    return 0 ; 
}
