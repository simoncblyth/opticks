#pragma once

template<typename T>
class NPY ; 


#include "NPY_API_EXPORT.hh"
#include "NPY_FLAGS.hh"


class AxisNPY {
   public:  
       AxisNPY(NPY<float>* axis); 
   public:  
       NPY<float>*           getAxis();
   public:  
       void dump(const char* msg="AxisNPY::dump");
   private:
       NPY<float>*                  m_axis ; 

};

inline AxisNPY::AxisNPY(NPY<float>* axis) 
       :  
       m_axis(axis)
{
}
inline NPY<float>* AxisNPY::getAxis()
{
    return m_axis ; 
}


