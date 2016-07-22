#include "GGeo.hh"
#include "Opticks.hh"
#include "OpticksGeometry.hh"

#include "GGEO_LOG.hh"
#include "OKCORE_LOG.hh"

#ifdef WITH_OPTIX
#include "OpEngine.hh"
#include "OXRAP_LOG.hh"
#include "OKOP_LOG.hh"
#endif

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    GGEO_LOG__ ; 
    OKCORE_LOG__ ; 
    OXRAP_LOG__ ; 
    OKOP_LOG__ ; 

    Opticks* m_opticks(NULL) ; 
    m_opticks = new Opticks(argc, argv);
    m_opticks->configure();


    OpticksGeometry* m_geometry(NULL); 
    m_geometry = new OpticksGeometry(m_opticks); 
    m_geometry->loadGeometry();


    GGeo* m_ggeo(NULL) ; 
    m_ggeo = m_geometry->getGGeo();

#ifdef WITH_OPTIX
    OpEngine* m_ope(NULL) ; 
    m_ope = new OpEngine(m_opticks, m_ggeo);
    m_ope->prepareOptiX();
#endif


    return 0 ; 
}
