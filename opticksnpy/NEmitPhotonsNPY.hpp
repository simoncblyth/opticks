#pragma once

template <typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"

class NCSG ; 

class FabStepNPY ; 

struct nnode ; 
struct NEmitConfig ; 

/**
NEmitPhotonsNPY
=================

Prepares input photon buffer for an NCSG instance.

Canonical m_emitter instance is ctor resident of OpticksGen
which in turn is ctor resident of OpticksHub.

TODO:

* dependency sink gencode (OpticksPhoton.h) 
  to avoid the gencode EMITSOURCE having to be passed 
  down from higher levels (perhaps a lowest level OpticksBase package ?)

**/

class NPY_API NEmitPhotonsNPY 
{
   public:
      NEmitPhotonsNPY(NCSG* csg, unsigned gencode, unsigned seed, bool emitdbg);

      NPY<float>* getPhotons() const ;
      FabStepNPY* getFabStep() const ;
      NPY<float>* getFabStepData() const ;
      std::string desc() const  ;

   private:
      void init(); 
   private:
      NCSG*         m_csg ; 
      bool          m_emitdbg ; 
      unsigned      m_seed ; 
      int           m_emit ; 
      const char*   m_emitcfg_ ;
      NEmitConfig*  m_emitcfg  ;       
      nnode*        m_root ; 

   private:
      // products 
      NPY<float>*   m_photons ; 
      FabStepNPY*   m_fabstep ; 
      NPY<float>*   m_fabstep_npy ; 

};



