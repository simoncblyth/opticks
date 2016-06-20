#include <cstddef>

#include "OpticksEngine.hh"
#include "Opticks.hh"
#include "OpticksCfg.hh"

#include "BLog.hh"


OpticksEngine::OpticksEngine(Opticks* opticks) 
    :
      m_opticks(opticks),
      m_cfg(NULL),
      m_evt(NULL)
{
     init();
}

void OpticksEngine::setEvent(OpticksEvent* evt)
{
    m_evt = evt ; 
}
Opticks* OpticksEngine::getOpticks()
{
    return m_opticks ; 
}
OpticksEvent* OpticksEngine::getEvent()
{
    return m_evt ; 
}




void OpticksEngine::init()
{
    m_cfg = m_opticks->getCfg();  
  
    LOG(info) << "OpticksEngine::init" ; 

    assert(m_cfg);
}




