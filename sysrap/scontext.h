#pragma once

#include "sdevice.h"
#include "SCVD.h"

struct scontext
{
    static constexpr const bool VERBOSE = false ; 

    scontext(); 
    std::vector<sdevice> visible_devices ;    
    std::vector<sdevice> all_devices ;    

    std::string desc() const ; 
};

/**

cf the former optixrap/OContext.cc OContext::initDevices

**/

inline scontext::scontext()
{
    //SCVD::ConfigureVisibleDevices();  
    // Seems the CVD->CUDA_VISIBLE_DEVICES promotion in code no longer impacts CUDA.
    // Perhaps the CUDA runtime/driver is reading the CUDA_VISIBLE_DEVICES envvar earlier ?
    // Before this sets it. 


    if(VERBOSE) std::cout << "[scontext::scontext" << std::endl ; 

    const char* dirpath = spath::ResolvePath("$HOME/.opticks/scontext") ; 
    int rc = sdirectory::MakeDirs(dirpath, 0); 
    assert(rc == 0); 

    // the below only saves when CVD envvar is not defined, so all dev visible
    bool nosave = false ; 
    sdevice::Visible(visible_devices, dirpath, nosave );  

    sdevice::Load(   all_devices, dirpath); 

    if(VERBOSE) std::cout << "]scontext::scontext" << std::endl ; 
}

inline std::string scontext::desc() const 
{
    std::stringstream ss ; 
    ss << "scontext::desc" << std::endl ; 
    ss << "all_devices" << std::endl ; 
    ss << sdevice::Desc(all_devices) ; 
    ss << "visible_devices" << std::endl ; 
    ss << sdevice::Desc(visible_devices) ; 
    std::string str = ss.str(); 
    return str ; 
}


