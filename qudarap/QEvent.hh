#pragma once

struct qevent ; 
struct quad4 ;
struct qat4 ; 
struct quad6 ;
struct NP ; 
template <typename T> struct QBuf ; 

#include <vector>
#include <string>
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"

/**
QEvent
=======

TODO: follow OEvent technique of initial allocation and resizing at each event 

**/

struct QUDARAP_API QEvent
{
    static NP* MakeGensteps(const std::vector<quad6>& gs ); 
    static NP* MakeCenterExtentGensteps(const float4& ce, const uint4& cegs, float gridscale ); 
    static NP* MakeCenterExtentGensteps(const float4& ce, const uint4& cegs, float gridscale, const qat4& q ) ; 
    static NP* MakeCountGensteps(); 
    static NP* MakeCountGensteps(const std::vector<int>& photon_counts_per_genstep); 

    static const plog::Severity LEVEL ; 
    static const QEvent* INSTANCE ; 
    static const QEvent* Get(); 

    QEvent(); 

    qevent*      evt ; 
    qevent*      d_evt ; 

    const NP* gs ;  
    QBuf<float>* genstep ; 
    QBuf<int>*   seed  ;

    void setGensteps(const NP* gs);
    void setGensteps(QBuf<float>* dgs ); 

    void downloadPhoton( std::vector<quad4>& photon ); 
    void savePhoton( const char* dir, const char* name ); 
    void saveGenstep( const char* dir_, const char* name); 
 
    void checkEvt() ;  // GPU side 

    qevent* getDevicePtr() const ;
    unsigned getNumPhotons() const ;  
    std::string desc() const ; 
};



