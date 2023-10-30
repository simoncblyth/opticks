#pragma once
/**
scontext.h
============

Looks like everything now done in sdevice.h 
plus SCVD.h no longer working 

TODO: remove SCVD.h 
TODO: update this scontext.h for sdevice.h doing most of the task... 
      scontext.h can just hold onto the vectors of sdevice structs 

::

    ./CSGOptiX/tests/CSGOptiXRenderTest.cc:#include "scontext.h"
    ./sysrap/CMakeLists.txt:    scontext.h
    ./sysrap/tests/scontext_test.cc:#include "scontext.h"
    ./sysrap/scontext.h:TODO: remove SCVD.h and scontext.h 
    ./sysrap/SSim.cc:#include "scontext.h"

**/

#include "sdevice.h"
#include "SCVD.h"

struct scontext
{
    static scontext* INSTANCE ; 
    static constexpr const bool VERBOSE = false ; 

    scontext(); 
    std::vector<sdevice> visible_devices ;    
    std::vector<sdevice> all_devices ;    

    std::string desc() const ; 
    std::string brief() const ; 

    static scontext* Get() ; 
    static std::string Desc() ; 
    static std::string Brief() ; 
};

/**

cf the former optixrap/OContext.cc OContext::initDevices

**/

scontext* scontext::INSTANCE = nullptr ; 

inline scontext::scontext()
{
    INSTANCE = this ; 

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
    ss << "scontext::desc [" << brief() << "]" << std::endl ; 
    ss << "all_devices" << std::endl ; 
    ss << sdevice::Desc(all_devices) ; 
    ss << "visible_devices" << std::endl ; 
    ss << sdevice::Desc(visible_devices) ; 
    std::string str = ss.str(); 
    return str ; 
}

inline std::string scontext::brief() const 
{
    return sdevice::Brief(visible_devices) ; 
}


inline scontext* scontext::Get()
{
    return INSTANCE ; 
}
inline std::string scontext::Desc() 
{
    return INSTANCE ? INSTANCE->desc() : "-" ; 
}
inline std::string scontext::Brief() 
{
    return INSTANCE ? INSTANCE->brief() : "-" ; 
}






