#pragma once
/**
SDeviceGeometry.h
==================

These members were formerly inlined with CSGOptiX/Params.h
they are now grouped separately for enhanced flexibility
and a nested geom struct is included into CSGOptiX/Params.h

**/

struct CSGNode ;
struct float4 ;
struct qat4 ;

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <sstream>
#include <iomanip>
#endif



struct __align__(16) SDeviceGeometry
{
    // 64-bit (8 byte) pointers and handles : need pairs of them to stay 16-byte aligned

    CSGNode*   node ;
    float4*    plan ;

    qat4*      tran ;
    qat4*      itra ;


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void init();
    std::string detail() const ;
#endif

};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
inline void SDeviceGeometry::init()
{
    node = nullptr ;
    plan = nullptr ;
    tran = nullptr ;
    itra = nullptr ;
}

inline std::string SDeviceGeometry::detail() const
{
    std::stringstream ss ;
    ss
        << "[SDeviceGeometry::detail\n"
        << std::setw(20) << " node " << std::setw(10) << node  << "\n"
        << std::setw(20) << " plan " << std::setw(10) << plan  << "\n"
        << std::setw(20) << " tran " << std::setw(10) << tran  << "\n"
        << std::setw(20) << " itra " << std::setw(10) << itra  << "\n"
        << "]SDeviceGeometry::detail\n"
        ;
    std::string str = ss.str();
    return str ;
}
#endif



