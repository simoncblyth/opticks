#include <string>
class OpticksQuery ; 

#include "OpticksHub.hh"
#include "OpticksCfg.hh"
#include "Opticks.hh"

#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GGeoTestConfig.hh"
#include "GGeo.hh"
#include "GSurLib.hh"
#include "GSur.hh"

#include "CTestDetector.hh"
#include "CGDMLDetector.hh"

class CPropLib ; 
#include "CG4.hh"
#include "CDetector.hh"
#include "CGeometry.hh"

#include "PLOG.hh"

CGeometry::CGeometry(OpticksHub* hub) 
   :
   m_hub(hub),
   m_ggeo(hub->getGGeo()),
   m_surlib(m_ggeo->getSurLib()),
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
        kludgeSurfaces(); 
    }

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


void CGeometry::kludgeSurfaces()
{
    unsigned numSur = m_surlib->getNumSur();
    LOG(info) << "CGeometry::kludgeSurfaces numSur " << numSur  ; 
    for(unsigned i=0 ; i < numSur ; i++)
    {   
        GSur* sur = m_surlib->getSur(i);
        char type = sur->getType();
        const char* name = sur->getName() ;

        LOG(info) << sur->brief() ; 

        //if(type == 'B') LOG(info) << type << std::setw(4) << i << " " << name ; 
    }
}


