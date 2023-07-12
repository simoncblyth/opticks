#pragma once
/**
QDebug.hh : manages qdebug.h instance
============================================

The hostside qdebug.h instance *dbg* is populated by QDebug::MakeInstance
then uploaded to the device *d_dbg* by QDebug::QDebug 

qdebug avoids having to play pass the parameter thru multiple levels of calls  
to get values onto the device 

Notice how NOT using pointers in qdebug provides a simple plain old struct way 
to get structured info onto the device. 

**/

struct qdebug ; 


#include <string>
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"


struct QUDARAP_API QDebug
{
    static const plog::Severity LEVEL ; 
    static const QDebug* INSTANCE ; 
    static const QDebug* Get(); 
    static qdebug* MakeInstance(); 

    QDebug(); 

    qdebug*      dbg ; 
    qdebug*      d_dbg ;
    qdebug*   getDevicePtr() const ;

    std::string desc() const ; 

}; 


