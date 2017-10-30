#pragma once

template <typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"

class NCSG ; 
struct nnode ; 
struct NEmitConfig ; 

class NPY_API NEmitPhotonsNPY 
{
   public:
      static NPY<float>* make(NCSG* csg);
   public:
      NEmitPhotonsNPY(NCSG* csg);

      NPY<float>* getNPY() const ;
      std::string desc() const  ;

   private:
      void init(); 

   private:
   
      NCSG*         m_csg ; 
      int           m_emit ; 
      const char*   m_cfg_ ;
      NEmitConfig*  m_cfg  ; 
      
      nnode*        m_root ; 

      NPY<float>*   m_data ; 
};



