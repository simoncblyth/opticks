#include <string>
class OpticksQuery ; 

#include "OpticksHub.hh"
#include "OpticksCfg.hh"
#include "Opticks.hh"

#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GGeoTestConfig.hh"
#include "GSurLib.hh"
#include "GSur.hh"

#include "CMaterialTable.hh"
#include "CMaterialBridge.hh"
#include "CSurfaceBridge.hh"
#include "CTestDetector.hh"
#include "CGDMLDetector.hh"

class CPropLib ; 
#include "CG4.hh"
#include "CDetector.hh"
#include "CGeometry.hh"
#include "CSurLib.hh"

#include "PLOG.hh"

CGeometry::CGeometry(OpticksHub* hub) 
   :
   m_hub(hub),
   m_ok(m_hub->getOpticks()),
   m_cfg(m_ok->getCfg()),
   m_detector(NULL),
   m_lib(NULL),
   m_material_table(NULL),
   m_material_bridge(NULL),
   m_surface_bridge(NULL)
{
   init();
}

CMaterialLib* CGeometry::getPropLib()
{
   return m_lib ; 
}
CDetector* CGeometry::getDetector()
{
   return m_detector; 
}




void CGeometry::init()
{
    CDetector* detector = NULL ; 
    if(m_ok->hasOpt("test"))
    {
        LOG(fatal) << "CGeometry::init G4 simple test geometry " ; 
        std::string testconfig = m_cfg->getTestConfig();
        GGeoTestConfig* ggtc = new GGeoTestConfig( testconfig.empty() ? NULL : testconfig.c_str() );
        OpticksQuery* query = NULL ;  // normally no OPTICKS_QUERY geometry subselection with test geometries
        detector  = static_cast<CDetector*>(new CTestDetector(m_hub, ggtc, query)) ; 
    }
    else
    {
        // no options here: will load the .gdml sidecar of the geocache .dae 
        LOG(fatal) << "CGeometry::init G4 GDML geometry " ; 
        OpticksQuery* query = m_ok->getQuery();
        detector  = static_cast<CDetector*>(new CGDMLDetector(m_hub, query)) ; 
    }

    detector->attachSurfaces();
    //m_csurlib->convert(detector);

    m_detector = detector ; 
    m_lib = detector->getPropLib();
}


bool CGeometry::hookup(CG4* g4)
{
    g4->setUserInitialization(m_detector);

    glm::vec4 ce = m_detector->getCenterExtent();
    LOG(info) << "CGeometry::hookup"
              << " center_extent " << gformat(ce) 
              ;    

    m_ok->setSpaceDomain(ce); // triggers Opticks::configureDomains

   return true ; 
}


void CGeometry::postinitialize()
{
    // both these are deferred til here as needs G4 materials table to have been constructed 

    m_material_table = new CMaterialTable(m_ok->getMaterialPrefix()); // TODO: move into CGeometry
    //m_material_table->dump("CGeometry::postinitialize");

    GMaterialLib* mlib = m_hub->getMaterialLib(); 
    m_material_bridge = new CMaterialBridge( mlib ); 

    GSurfaceLib* slib = m_hub->getSurfaceLib(); 
    m_surface_bridge = new CSurfaceBridge( slib ); 

}

CMaterialBridge* CGeometry::getMaterialBridge()
{
   // used by CRecorder::postinitialize
    assert(m_material_bridge);
    return m_material_bridge ; 
}
CSurfaceBridge* CGeometry::getSurfaceBridge()
{
    assert(m_surface_bridge);
    return m_surface_bridge ; 
}

std::map<std::string, unsigned>& CGeometry::getMaterialMap()
{
    assert(m_material_table);
    return m_material_table->getMaterialMap();
}



