#pragma once

// *OpticksEngine* 
// =================
//
//  Represents the common aspects of 
//
//  * "Contained Geant4" simulator  cfg4-/CG4 
//  * Opticks simulator  opticksop-/OpEngine
//
//  So they can be used(at a very high level) interchangably 

#include <cstddef>

// optickscore-
class Opticks ;
class OpticksEvent ;
template <typename T> class OpticksCfg ;

class OpticksEngine {
   public:
       OpticksEngine(Opticks* opticks);
   public:
       void setEvent(OpticksEvent* evt);
   public:
        Opticks*      getOpticks();
        OpticksEvent* getEvent();
   private:
        void init();
   protected:
       Opticks*              m_opticks ; 
       OpticksCfg<Opticks>*  m_cfg ;
       OpticksEvent*         m_evt ; 

};

inline OpticksEngine::OpticksEngine(Opticks* opticks) 
    :
      m_opticks(opticks),
      m_cfg(NULL),
      m_evt(NULL)
{
     init();
}

inline void OpticksEngine::setEvent(OpticksEvent* evt)
{
    m_evt = evt ; 
}
inline Opticks* OpticksEngine::getOpticks()
{
    return m_opticks ; 
}
inline OpticksEvent* OpticksEngine::getEvent()
{
    return m_evt ; 
}

