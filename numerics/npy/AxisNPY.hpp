#pragma once

template<typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"

class NPY_API AxisNPY {
   public:  
       AxisNPY(NPY<float>* axis); 
       NPY<float>*           getAxis();
       void dump(const char* msg="AxisNPY::dump");
   private:
       NPY<float>*                  m_axis ; 

};

