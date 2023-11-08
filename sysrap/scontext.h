#pragma once
/**
scontext.h : holds sdevice.h structs for all and visible GPUs
==============================================================

Canonical instance is SSim::sctx instanciated by SSim::SSim


TODO: use this a source of metadata 
      for inclusion into SEvt run_meta ? 

::

    epsilon:opticks blyth$ opticks-fl scontext.h 
    ./sysrap/CMakeLists.txt
    ./sysrap/sdevice.h
    ./sysrap/tests/scontext_test.cc
    ./sysrap/scontext.h
    ./sysrap/SSim.cc

Equivalent of this in the old workflow was 
optixrap/OContext.cc OContext::initDevices

Formerly used SCVD.h to promote "CVD" into "CUDA_VISIBLE_DEVICES" 
in code in order to have more control over visible GPUs. 
However find that approach no longer impacts CUDA. 
Perhaps following a CUDA version/runtime/driver update the 
CUDA_VISIBLE_DEVICES envvar is read earlier than previously. 

**/

#include "sdevice.h"

struct scontext
{
    static constexpr const bool VERBOSE = false ; 

    scontext(); 
    std::vector<sdevice> visible_devices ;    
    std::vector<sdevice> all_devices ;    

    std::string desc() const ; 
    std::string brief() const ; 
};



inline scontext::scontext()
{
    if(VERBOSE) std::cout << "[scontext::scontext" << std::endl ; 

    const char* dirpath = spath::Resolve("$HOME/.opticks/scontext") ; 

    if(VERBOSE) std::cout << " scontext::scontext dirpath " << ( dirpath ? dirpath : "-" )  << std::endl ; 

    int rc = sdirectory::MakeDirs(dirpath, 0); 
    assert(rc == 0); 

    // the below only saves when CUDA_VISIBLE_DEVICES envvar is not defined, so all dev visible
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

