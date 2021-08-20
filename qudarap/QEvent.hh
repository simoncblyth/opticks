#pragma once

//struct quad6 ;
//template <typename T> struct QBuf ; 

#include <cuda_runtime.h>
#include "scuda.h"
#include "squad.h"
#include "QBuf.hh"

#include "QUDARAP_API_EXPORT.hh"

/**
QEvent
=======

**/

struct QUDARAP_API QEvent
{
    QBuf<quad6> gs ; 
    QBuf<int>   se ; 

    void setGensteps( QBuf<quad6> gs_ ); 

    static QEvent* MakeFake(); 
};



