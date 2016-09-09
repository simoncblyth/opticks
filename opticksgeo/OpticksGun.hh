#pragma once

#include <string>
class OpticksHub ; 
class Opticks ; 

#include "OKGEO_API_EXPORT.hh"
class OKGEO_API OpticksGun
{
    public:
         OpticksGun(OpticksHub* hub);
         std::string getConfig();
    private:
         void init();
    private:
         OpticksHub* m_hub ; 
         Opticks*    m_ok ; 

};
