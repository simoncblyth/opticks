#pragma once

template <typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"
#include "plog/Severity.h"

class NCSG ; 
class NRngDiffuse ; 

class FabStepNPY ; 

struct nnode ; 
struct NEmitConfig ; 

/**
NEmitPhotonsNPY
=================

Prepares input photon buffer for an NCSG instance.

Canonical m_emitter instance is ctor resident of OpticksGen
which in turn is ctor resident of OpticksHub.  m_emitter is only 
instanciated when there is an emitter NCSG configured 
in the geometry.

Masked running is handled by generating all photons as with 
normal runing and then making a masked copy of them.


**/

class NPY_API NEmitPhotonsNPY 
{
      static const plog::Severity LEVEL ; 
   public:
      NEmitPhotonsNPY(NCSG* csg, unsigned gencode, unsigned seed, bool dbgemit, NPY<unsigned>* mask, int num_photons=-1 );

      NPY<float>* getPhotons() const ;
      NPY<float>* getPhotonsRaw() const  ;

      FabStepNPY* getFabStep() const ;
      FabStepNPY* getFabStepRaw() const ;

      NPY<float>* getFabStepData() const ;
      NPY<float>* getFabStepRawData() const ;

   public:
      std::string desc() const  ;
   private:
      void init(); 
   private:
      NCSG*          m_csg ; 
      unsigned       m_gencode ; 
      unsigned       m_seed ; 
      bool           m_dbgemit ;   // --dbgemit
      NPY<unsigned>* m_mask ; 
   private:
      int            m_emit ; 
      const char*    m_emitcfg_ ;
      NEmitConfig*   m_emitcfg  ;       
      int            m_num_photons ;  
      nnode*         m_root ; 
   private:
      // products 
      NPY<float>*   m_photons ; 
      NPY<float>*   m_photons_masked ; 

      FabStepNPY*   m_fabstep ; 
      FabStepNPY*   m_fabstep_masked ;
 
      NRngDiffuse*  m_diffuse ; 

};



