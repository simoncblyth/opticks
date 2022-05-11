#pragma once

#include <string>
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"

struct qbase ; 

struct QUDARAP_API QBase
{
    static const plog::Severity LEVEL ;
    static const QBase*         INSTANCE ; 
    static const QBase*         Get(); 

    static qbase* MakeInstance();

    QBase();  
    void init(); 
    std::string desc() const ; 

    qbase*  base ; 
    qbase*  d_base ; 

}; 
