#pragma once

struct qevent ; 
struct quad6 ;
template <typename T> struct QBuf ; 

#include <vector>
#include <string>
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"

/**
QEvent
=======

**/

struct QUDARAP_API QEvent
{
    static const plog::Severity LEVEL ; 
    static const QEvent* INSTANCE ; 
    static const QEvent* Get(); 

    QEvent(); 

    qevent*      evt ; 
    qevent*      d_evt ; 
    QBuf<quad6>* gensteps ; 
    QBuf<int>*   seeds  ;
    unsigned     num_photons ; 

    void setGenstepsFake(const std::vector<int>& photons_per_genstep ); 
    void setGensteps(QBuf<quad6>* gs_ ); 
    void uploadEvt(); 
    void checkEvt() ; 

    qevent* getDevicePtr() const ;
    unsigned getNumPhotons() const ;  
    std::string desc() const ; 
};



