#pragma once

template <typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"

class NPY_API TrivialCheckNPY {
   public:  
       TrivialCheckNPY(NPY<float>* photons, NPY<float>* gensteps);
   public:  
       void dump(const char* msg="TrivialCheckNPY::dump");
  private:
        NPY<float>*  m_photons ; 
        NPY<float>*  m_gensteps ; 
};



