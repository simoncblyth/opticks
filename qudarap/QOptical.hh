#pragma once

/**
QOptical
===========

Managing the optical_buffer which holds surface and material 
info for each boundary. 

This is closely related to QBnd 

**/

#include <string>
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"

union quad ; 
struct NP ; 
template <typename T> struct QBuf ; 

struct QUDARAP_API QOptical
{
    static const plog::Severity LEVEL ;
    static const QOptical*      INSTANCE ; 
    static const QOptical*      Get(); 

    QOptical(const NP* optical);
    std::string desc() const ; 
    void check() const ; 

    const NP*       optical ;  
    QBuf<unsigned>* buf ; 
    quad*           d_optical ; 

}; 

