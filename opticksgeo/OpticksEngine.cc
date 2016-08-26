#include <cstddef>

#include "OpticksEngine.hh"
#include "OpticksHub.hh"
#include "Opticks.hh"
#include "OpticksCfg.hh"

#include "PLOG.hh"


OpticksEngine::OpticksEngine(OpticksHub* hub) 
    :
      m_hub(hub),
      m_opticks(hub->getOpticks()),
      m_cfg(NULL)
{
     init();
}

Opticks* OpticksEngine::getOpticks()
{
    return m_opticks ; 
}


void OpticksEngine::init()
{
    m_cfg = m_opticks->getCfg();  
  
    LOG(info) << "OpticksEngine::init" ; 

    assert(m_cfg);
}




