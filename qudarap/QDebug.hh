#pragma once

struct qdebug ; 
struct quad6 ;
struct NP ; 

template <typename T> struct QBuf ; 

#include <vector>
#include <string>
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"

/**
QDebug
=======

HMM : maybe this is not needed for qdebug 
unlike QProp QRng QOptical which have inputs to manage 

**/

struct QUDARAP_API QDebug
{
    static const plog::Severity LEVEL ; 
    static const QDebug* INSTANCE ; 
    static const QDebug* Get(); 

    QDebug(); 

    qdebug*      dbg ; 
    qdebug*      d_dbg ;

    qdebug*   getDevicePtr() const ;

}; 


