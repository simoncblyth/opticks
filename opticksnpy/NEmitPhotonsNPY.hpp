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
      NEmitPhotonsNPY(NCSG* csg, unsigned gencode);

      NPY<float>* getPhotons() const ;
      NPY<float>* getFabStepData() const ;
      std::string desc() const  ;

   private:
      void init(); 
   private:
      NCSG*         m_csg ; 
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



