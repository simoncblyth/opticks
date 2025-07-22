#pragma once
/**
scontext.h : holds sdevice.h structs for all and visible GPUs
==============================================================

Canonical instance is SEventConfig::CONTEXT instanciated by
SEventConfig::Initialize with SEventConfig::Initialize_Meta.
This Initialize happens on instanciation of the first SEvt.

::

    A[blyth@localhost opticks]$ opticks-fl scontext.h
    ./sysrap/CMakeLists.txt
    ./sysrap/SEventConfig.cc
    ./sysrap/sdevice.h
    ./sysrap/tests/scontext_test.cc
    ./sysrap/scontext.h

::

   ~/o/sysrap/tests/scontext_test.sh


**/

#include <cstdlib>
#include <csignal>
#include "sdevice.h"
#include "ssys.h"
#include "SEventConfig.hh"

struct scontext
{
    static constexpr const char* _level = "scontext__level" ;
    static int level ;

    scontext();
    void init();
    void initPersist();
    void initConfig();
    void initConfig_SetDevice(int idev);

    std::vector<sdevice> visible_devices ;
    std::vector<sdevice> all_devices ;

    std::string desc() const ;
    std::string brief() const ;
    std::string vram() const ;

    // query visible_devices[idx]
    std::string device_name(int idx) const ;
    size_t totalGlobalMem_bytes(int idx) const ;
    size_t totalGlobalMem_GB(int idx) const ;

    std::string main(int arg, char** argv) const ;
};


inline int scontext::level = ssys::getenvint(_level, 0 );




inline scontext::scontext()
{
    init();
}
inline void scontext::init()
{
    initPersist();
    initConfig();
}


/**
scontext::initPersist
-----------------------

HMM: in workstation context it makes sense to persist
info on all GPUs into $HOME/.opticks/scontext as that
does not change much.

BUT in batch submission context on a GPU cluster
the number and identity of GPUs can depend on the
job submission so using a fixed place makes no
sense.  In that situation a more appropriate
location is the invoking directory.

Original motivation for persisting GPU info for all GPUs
(ie all those detected by CUDA API when CUDA_VISIBLE_DEVICES is not defined)
was for making sense of which GPU is in use in a changing environment
of CUDA_VISIBLE_DEVICES values and hence indices.

Using the record for all GPUs enabled associating an absolute ordinal
(identity based on uuid and name of the GPU) to GPUs even when
CUDA_VISIBLE_DEVICES means that not all GPUs are visible.

**/


inline void scontext::initPersist()
{
    if(level > 0) std::cout << "[scontext::initPersist" << std::endl ;

    sdevice::Visible(visible_devices);
    sdevice::Load(   all_devices );   // seems all_devices not used much from here

    if(level > 0) std::cout << "]scontext::initPersist" << std::endl ;
}

inline void scontext::initConfig()
{
    int numdev = visible_devices.size();
    int idev = -1 ;

    if(numdev == 0)
    {
        std::cerr << "scontext::initConfig : ZERO VISIBLE DEVICES - CHECK CUDA_VISIBLE_DEVICES envvar \n" ;
    }
    else if(numdev == 1)
    {
        idev = 0 ;
    }
    else if(numdev > 1)
    {
        idev = 0 ;
        std::cerr
            << "scontext::initConfig : WARNING - MORE THAN ONE VISIBLE DEVICES - DEFAULTING TO USE idev:[" << idev << "]\n"
            << "scontext::initConfig : QUELL THIS WARNING BY SETTING/CHANGING CUDA_VISIBLE_DEVICES envvar TO SELECT ONE DEVICE\n"
            ;
    }

    initConfig_SetDevice(idev);
}

inline void scontext::initConfig_SetDevice(int idev)
{
    int numdev = visible_devices.size();
    bool idev_valid = idev >=0 && idev < numdev ;

    if( !idev_valid || level > 0 ) std::cerr
        << "scontext::initConfig_SetDevice "
        << " numdev " << numdev
        << " idev " << idev
        << " level " << level
        << " idev_valid " << ( idev_valid ? "YES" : "NO " )
        << "\n"
        ;

    if(!idev_valid) return ;

    std::string name = device_name(idev);
    size_t vram = totalGlobalMem_bytes(idev);
    SEventConfig::SetDevice(vram, name);
}


inline std::string scontext::desc() const
{
    char* cvd = getenv("CUDA_VISIBLE_DEVICES") ;
    std::stringstream ss ;
    ss << "scontext::desc [" << brief() << "]" << std::endl ;
    ss << "CUDA_VISIBLE_DEVICES : [" << ( cvd ? cvd : "-" ) << "]" << std::endl;
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

inline std::string scontext::vram() const
{
    return sdevice::VRAM(visible_devices) ;
}

inline std::string scontext::main(int argc, char** argv) const
{
    std::stringstream ss ;
    if(argc == 1) ss << brief() ;

    for(int i=1 ; i < argc ; i++)
    {
        char* arg = argv[i] ;
        if(strcmp(arg, "--brief")==0) ss << brief() << "\n" ;
        if(strcmp(arg, "--desc")==0)  ss << desc() << "\n" ;
        if(strcmp(arg, "--vram")==0)  ss << vram() << "\n" ;
        if(strcmp(arg, "--name0")==0)  ss << device_name(0) << "\n" ;
        if(strcmp(arg, "--name1")==0)  ss << device_name(1) << "\n" ;
        if(strcmp(arg, "--vram0")==0)  ss << totalGlobalMem_bytes(0) << "\n" ;
        if(strcmp(arg, "--vram1")==0)  ss << totalGlobalMem_bytes(1) << "\n" ;
        if(strcmp(arg, "--vram0g")==0)  ss << totalGlobalMem_GB(0) << "\n" ;
        if(strcmp(arg, "--vram1g")==0)  ss << totalGlobalMem_GB(1) << "\n" ;

    }
    std::string str = ss.str();
    return str ;
}

inline std::string scontext::device_name(int idx) const
{
    return idx < int(visible_devices.size()) ? visible_devices[idx].name : "" ;
}
inline size_t scontext::totalGlobalMem_bytes(int idx) const
{
    return idx < int(visible_devices.size()) ? visible_devices[idx].totalGlobalMem_bytes() : 0 ;
}
inline size_t scontext::totalGlobalMem_GB(int idx) const
{
    return idx < int(visible_devices.size()) ? visible_devices[idx].totalGlobalMem_GB() : 0 ;
}



