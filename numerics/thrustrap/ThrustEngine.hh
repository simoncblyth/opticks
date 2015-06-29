#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void vers();

#ifdef __cplusplus
}
#endif


#include "stdlib.h"

class ThrustEngine {
   public:
       static void version();
   public:
       ThrustEngine();
       void setHistoryDevicePtr(unsigned long long* devptr);

   private:
       unsigned long long* m_history_devptr ; 


};

inline ThrustEngine::ThrustEngine() 
    :
    m_history_devptr(NULL)
{
}

inline void ThrustEngine::setHistoryDevicePtr(unsigned long long* devptr)
{
    m_history_devptr = devptr ; 
}





