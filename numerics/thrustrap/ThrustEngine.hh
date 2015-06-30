#pragma once

#include "stdlib.h"
#include "ThrustHistogram.hh"

class ThrustEngine {
   public:
       static void version();
   public:
       ThrustEngine();
       void setHistory(unsigned long long* devptr, unsigned int size);
   public:
       void createIndices();
   private:
       ThrustHistogram<unsigned long long>* m_history ; 

};

inline ThrustEngine::ThrustEngine() 
    :
    m_history(NULL)
{
    version();
}





