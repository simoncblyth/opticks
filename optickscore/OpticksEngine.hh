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
//

#include <cstddef>

// optickscore-
class Opticks ;
class OpticksEvent ;
template <typename T> class OpticksCfg ;

// ggeo-
class GCache ; 

class OpticksEngine {
   public:
       OpticksEngine(Opticks* opticks);
       void setEvent(OpticksEvent* evt);
       void setCache(GCache* cache);
   public:
        Opticks*      getOpticks();
        OpticksEvent* getEvent();
   private:
        void init();
   protected:
       Opticks*              m_opticks ; 
       OpticksCfg<Opticks>*  m_cfg ;
       GCache*               m_cache ; 
       OpticksEvent*         m_evt ; 

};

inline OpticksEngine::OpticksEngine(Opticks* opticks) 
    :
      m_opticks(opticks),
      m_cfg(NULL),
      m_cache(NULL),
      m_evt(NULL)
{
     init();
}

inline void OpticksEngine::setEvent(OpticksEvent* evt)
{
    m_evt = evt ; 
}
inline void OpticksEngine::setCache(GCache* cache)
{
    m_cache = cache ; 
}

inline Opticks* OpticksEngine::getOpticks()
{
    return m_opticks ; 
}
inline OpticksEvent* OpticksEngine::getEvent()
{
    return m_evt ; 
}

