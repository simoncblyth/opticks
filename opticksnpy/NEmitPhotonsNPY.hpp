#pragma once

template <typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"

class NCSG ; 

class FabStepNPY ; 

struct nnode ; 
struct NEmitConfig ; 


class NPY_API NEmitPhotonsNPY 
{
   public:
      NEmitPhotonsNPY(NCSG* csg, unsigned gencode, bool emitdbg=false);

      NPY<float>* getPhotons() const ;
      FabStepNPY* getFabStep() const ;
      NPY<float>* getFabStepData() const ;
      std::string desc() const  ;

   private:
      void init(); 
   private:
      NCSG*         m_csg ; 
      bool          m_emitdbg ; 
      int           m_emit ; 
      const char*   m_cfg_ ;
      NEmitConfig*  m_cfg  ;       
      nnode*        m_root ; 

   private:
      // products 
      NPY<float>*   m_photons ; 
      FabStepNPY*   m_fabstep ; 
      NPY<float>*   m_fabstep_npy ; 

};



