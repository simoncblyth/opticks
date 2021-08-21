#pragma once

struct qevent ; 
struct quad4 ;
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
    QBuf<quad6>* genstep ; 
    QBuf<int>*   seed  ;

    void setGenstepsFake(const std::vector<int>& photons_per_genstep ); 
    void setGensteps(QBuf<quad6>* gs_ ); 

    void downloadPhoton( std::vector<quad4>& photon ); 
 
    void checkEvt() ;  // GPU side 

    qevent* getDevicePtr() const ;
    unsigned getNumPhotons() const ;  
    std::string desc() const ; 
};



