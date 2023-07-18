#pragma once

/**
QOptical
===========

Managing the optical_buffer which holds surface and material 
info for each boundary. 

This is closely related to QBnd 

**/

#include <string>

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
#include "plog/Severity.h"
#endif

#include "QUDARAP_API_EXPORT.hh"

union quad ; 
struct NP ; 
template <typename T> struct QBuf ; 

struct QUDARAP_API QOptical
{

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
    static const plog::Severity LEVEL ;
#endif
    static const QOptical*      INSTANCE ; 
    static const QOptical*      Get(); 

    QOptical(const NP* optical);
    void init(); 

    std::string desc() const ; 
    void check() const ; 

    const NP*       optical ;  
    QBuf<unsigned>* buf ; 
    quad*           d_optical ; 

}; 

