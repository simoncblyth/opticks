#pragma once

template <typename T> class NPY ; 
class NSensorList ; 

#include "NPY_API_EXPORT.hh"
class NPY_API HitsNPY {
   public:  
       HitsNPY(NPY<float>* photons, NSensorList* sensorlist); 
   public:
       void debugdump(const char* msg="HitsNPY::debugdump");
   private:
       NPY<float>*                  m_photons ; 
       NSensorList*                 m_sensorlist ; 

};


