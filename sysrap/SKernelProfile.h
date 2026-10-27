#pragma once
/**
SKernelProfile.h
=================

**/

#include "skernel_profile.h"
#include "SU.hh"
#include "NP.hh"

struct SKernelProfile
{
    SKernelProfile();

    skernel_profile* profile ;
    skernel_profile* d_profile ;

    skernel_profile* getDevicePtr() const ;
    void upload();
    void download();

    std::string desc() const ;
    void dummy_populate() ;
    void zero_host() ;
    void zero_device() ;
    NP* serialize() const ;

};

inline SKernelProfile::SKernelProfile()
    :
    profile(new skernel_profile {}),
    d_profile(nullptr)
{
}

inline skernel_profile* SKernelProfile::getDevicePtr() const
{
    return d_profile ;
}

inline void SKernelProfile::upload()
{
    d_profile = SU::upload(profile, 1);
}

inline void SKernelProfile::download()
{
    if(d_profile == nullptr)
    {
        std::cerr << "SKernelProfile::download - ERROR - CALLED BEFORE upload\n" ;
    }
    else
    {
        SU::download(profile, d_profile, 1);
    }
}

inline std::string SKernelProfile::desc() const
{
    return profile->desc();
}

inline void SKernelProfile::dummy_populate()
{
    profile->dummy_populate();
}

inline void SKernelProfile::zero_host()
{
    profile->zero();
}

inline void SKernelProfile::zero_device()
{
    assert(d_profile);
    SU::zero(d_profile, 1);
}

inline NP* SKernelProfile::serialize() const
{
    return profile->serialize();
}



