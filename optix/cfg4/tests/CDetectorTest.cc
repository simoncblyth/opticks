// ggv-;ggv-pmt-test --cdetector

#include "Opticks.hh"
#include "OpticksCfg.hh"

#include "GCache.hh"
#include "GGeoTestConfig.hh"

#include "CPropLib.hh"
#include "CDetector.hh"
#include "CTraverser.hh"

#include "G4VPhysicalVolume.hh"


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

    CPropLib* clib = m_detector->getPropLib() ;

    G4VPhysicalVolume* pv = m_detector->Construct();

    clib->dumpMaterials();

    CTraverser* m_traverser = new CTraverser(pv) ;

    m_traverser->Traverse(); 

    m_traverser->dumpMaterials(); 



    return 0 ; 
}
