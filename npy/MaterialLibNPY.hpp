#pragma once

template <typename T> class NPY ;

#include "NPY_API_EXPORT.hh"

class NPY_API MaterialLibNPY {
   public:  
       MaterialLibNPY(NPY<float>* mlib); 
   public:  
       void dump(const char* msg="MaterialLibNPY::dump");
       void dumpMaterial(unsigned int i);
   private:
       NPY<float>*   m_lib ; 
 
};



 
