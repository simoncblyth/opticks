#pragma once

#include "stdlib.h"
#include "ThrustHistogram.hh"



class ThrustEngine {
   public:
       static void version();
   public:
       ThrustEngine();
       void setHistoryTarget(unsigned long long* history_devptr, unsigned int* target_devptr,  unsigned int size);

       ThrustHistogram<unsigned long long, unsigned int>* getHistory();
   public:
       void createIndices();
   private:
       ThrustHistogram<unsigned long long, unsigned int>* m_history ; 

};

inline ThrustEngine::ThrustEngine() 
    :
    m_history(NULL)
{
    version();
}

inline ThrustHistogram<unsigned long long, unsigned int>* ThrustEngine::getHistory() 
{
    return m_history ;
}




