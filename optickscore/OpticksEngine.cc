#include "OpticksEngine.hh"
#include "Opticks.hh"
#include "OpticksCfg.hh"


void OpticksEngine::init()
{
}

void OpticksEngine::configureEngine()
{
    m_opticks->configure();
    m_cfg = m_opticks->getCfg();  
}



