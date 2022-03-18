#include <cuda_runtime.h>
#include "QDebug.hh"
#include "qdebug.h"

#include "QU.hh"
#include "PLOG.hh"

const plog::Severity QDebug::LEVEL = PLOG::EnvLevel("QDebug", "DEBUG") ; 
const QDebug* QDebug::INSTANCE = nullptr ; 
const QDebug* QDebug::Get(){ return INSTANCE ; }

/**
QDebug::QDebug
----------------

**/

QDebug::QDebug()
    :
    dbg(new qdebug),
    d_dbg(QU::device_alloc<qdebug>(1))
{
    INSTANCE = this ; 
}


qdebug* QDebug::getDevicePtr() const
{
    return d_dbg ; 
}




