#pragma once
#include "NPY.hpp"

class NSensorList ; 

class HitsNPY {
   public:  
       HitsNPY(NPY<float>* photons, NSensorList* sensorlist); 
   public:
       void debugdump(const char* msg="HitsNPY::debugdump");
   private:
       NPY<float>*                  m_photons ; 
       NSensorList*                 m_sensorlist ; 

};

inline HitsNPY::HitsNPY(NPY<float>* photons, NSensorList* sensorlist) 
       :  
       m_photons(photons),
       m_sensorlist(sensorlist)
{
}


