#include "Opticks.hh"
#include "OpticksHub.hh"

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

    Opticks* m_opticks = new Opticks(argc, argv);
    m_opticks->configure();

    OpticksHub* m_hub = new OpticksHub(m_opticks);
    m_hub->loadGeometry();
     
#ifdef WITH_OPTIX
    OpEngine* m_ope = new OpEngine(m_hub);
    m_ope->prepareOptiX();
#endif


    return 0 ; 
}
