#pragma once
/**
skernel_profile.h
===================

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "NP.hh"
#endif


struct skernel_profile
{
    enum { MAX_BOUNCE = 64 } ;
    unsigned long long active_counts[MAX_BOUNCE];
    unsigned long long trace_cycles[MAX_BOUNCE];
    unsigned long long propagate_cycles[MAX_BOUNCE];

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void dummy_populate();
    void zero();
    std::string desc() const ;
    NP* serialize() const ;
    void save(const char* path);
#endif

};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
inline void skernel_profile::dummy_populate()
{
    for(size_t bounce=0 ; bounce < MAX_BOUNCE ; bounce++ )
    {
        trace_cycles[bounce] = bounce*10 ;
        propagate_cycles[bounce] = bounce*5 ;
        active_counts[bounce] = bounce ;
    }
}

inline void skernel_profile::zero()
{
    for(size_t bounce=0 ; bounce < MAX_BOUNCE ; bounce++ )
    {
        trace_cycles[bounce] = 0ull ;
        propagate_cycles[bounce] = 0ull ;
        active_counts[bounce] = 0ull ;
    }
}



inline std::string skernel_profile::desc() const
{
    int wid = 20 ;
    std::stringstream ss ;
    ss
         << std::setw(wid) << "bounce"
         << " | "
         << std::setw(wid) << "active"
         << " | "
         << std::setw(wid) << "trace"
         << " | "
         << std::setw(wid) << "propag"
         << "\n"
         ;

    for(size_t bounce=0 ; bounce < MAX_BOUNCE ; bounce++ ) ss
         << std::setw(wid) << bounce
         << " | "
         << std::setw(wid) << active_counts[bounce]
         << " | "
         << std::setw(wid) << trace_cycles[bounce]
         << " | "
         << std::setw(wid) << propagate_cycles[bounce]
         << "\n"
         ;

    std::string str = ss.str() ;
    return str ;
}

inline NP* skernel_profile::serialize() const
{
    size_t ni = MAX_BOUNCE ;
    size_t nj = 3 ;
    NP* a = NP::Make<unsigned long long>( ni, nj );
    unsigned long long* aa = a->values<unsigned long long>();
    for(size_t i=0 ; i < ni ; i++)
    {
        aa[nj*i + 0] = active_counts[i];
        aa[nj*i + 1] = trace_cycles[i];
        aa[nj*i + 2] = propagate_cycles[i];
    }
    return a ;
}
inline void skernel_profile::save(const char* path)
{
    NP* a = serialize();
    a->save(path);
}
#endif


