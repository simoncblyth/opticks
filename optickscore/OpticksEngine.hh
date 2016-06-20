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


// okc-
class Opticks ;
class OpticksEvent ;
template <typename T> class OpticksCfg ;


#include "OKCORE_API_EXPORT.hh"

class OKCORE_API OpticksEngine {
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


