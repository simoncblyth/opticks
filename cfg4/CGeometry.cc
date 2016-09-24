#include <string>
class OpticksQuery ; 

#include "OpticksHub.hh"
#include "OpticksCfg.hh"
#include "Opticks.hh"

#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GGeoTestConfig.hh"
#include "GGeo.hh"
#include "CTestDetector.hh"
#include "CGDMLDetector.hh"

class CPropLib ; 
#include "CG4.hh"
#include "CDetector.hh"
#include "CGeometry.hh"

#include "PLOG.hh"

CGeometry::CGeometry(OpticksHub* hub, CG4* g4) 
   :
   m_hub(hub),
   m_ggeo(hub->getGGeo()),
   m_g4(g4),
   m_ok(m_hub->getOpticks()),
   m_cfg(m_ok->getCfg()),
   m_detector(NULL),
   m_lib(NULL)
{
   init();
}

CPropLib* CGeometry::getPropLib()
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
        LOG(info) << "CGeometry::init G4 simple test geometry " ; 
        std::string testconfig = m_cfg->getTestConfig();
        GGeoTestConfig* ggtc = new GGeoTestConfig( testconfig.empty() ? NULL : testconfig.c_str() );
        OpticksQuery* query = NULL ;  // normally no OPTICKS_QUERY geometry subselection with test geometries
        detector  = static_cast<CDetector*>(new CTestDetector(m_ok, ggtc, query)) ; 
    }
    else
    {
        // no options here: will load the .gdml sidecar of the geocache .dae 
        LOG(info) << "CGeometry::init G4 GDML geometry " ; 
        OpticksQuery* query = m_ok->getQuery();
        detector  = static_cast<CDetector*>(new CGDMLDetector(m_ok, query)) ; 
        kludgeSensorSurfaces(); 
    }

    m_detector = detector ; 
    m_lib = detector->getPropLib();

    m_g4->setUserInitialization(detector);

    glm::vec4 ce = m_detector->getCenterExtent();
    LOG(info) << "CGeometry::init"
              << " center_extent " << gformat(ce) 
              ;    

    m_ok->setSpaceDomain(ce); // triggers Opticks::configureDomains

}


void CGeometry::kludgeSensorSurfaces()
{
    LOG(info) << "CGeometry::kludgeSensorSurfaces" ; 




}


