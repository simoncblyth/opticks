#pragma once

#include <string>

struct BConfig ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API NEmitConfig 
{
    static const char* DEFAULT ; 

    NEmitConfig(const char* cfg);

    struct BConfig* bconfig ;  
    std::string desc() const  ;
    void dump(const char* msg="NEmitConfig::dump") const ; 

    int verbosity ; 
    int photons ; 
    int wavelength ; 

    float time ; 
    float weight ; 


};

