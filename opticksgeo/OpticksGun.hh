#pragma once

#include <string>
class OpticksHub ; 
class Opticks ; 

/**
OpticksGun
===========

When the gun is enabled with option --g4gun the 
config comes from --g4gunconfig, however if that 
is blank a tag default config is used.

This is useful for commandline minimization during testing, where
different guns can be used simply by using a different tag.

**/

#include "OKGEO_API_EXPORT.hh"
class OKGEO_API OpticksGun
{
    public:
         OpticksGun(OpticksHub* hub);
         void assignTagDefault(std::string& config, int itag);
         std::string getConfig();
    private:
         void init();
    private:
         OpticksHub* m_hub ; 
         Opticks*    m_ok ; 

};
