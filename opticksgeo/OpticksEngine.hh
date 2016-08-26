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
template <typename T> class OpticksCfg ;

// opticksgeo-
class OpticksHub ;


#include "OKGEO_API_EXPORT.hh"

class OKGEO_API OpticksEngine {
   public:
        OpticksEngine(OpticksHub* hub);
   public:
        Opticks*      getOpticks();
   private:
        void init();
   protected:
       OpticksHub*           m_hub ; 
       Opticks*              m_opticks ; 
       OpticksCfg<Opticks>*  m_cfg ;

};


