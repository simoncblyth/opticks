#include <sstream>
#include <limits>

#include "SSys.hh"
#include "SLOG.hh"
#include "QU.hh"

#include "QBase.hh"
#include "qbase.h"

const plog::Severity QBase::LEVEL = SLOG::EnvLevel("QBase", "DEBUG"); 
const QBase* QBase::INSTANCE = nullptr ; 
const QBase* QBase::Get(){ return INSTANCE ; }




qbase* QBase::MakeInstance() // static 
{
    qbase* base = new qbase ; 
    base->pidx = SSys::getenvunsigned_fallback_max("PIDX") ; 
    return base ; 
}

QBase::QBase()
    :
    base(MakeInstance()),
    d_base(nullptr)
{
    init(); 
}

/**
QBase::init
------------

This appears to take ~0.25s because it is usually the first access to the GPU. 
This is with the nvidia-persistenced running, without it the time is ~1.5s

Tried changing the visible devices but seems to make no difference::

    CUDA_VISIBLE_DEVICES=0
    CUDA_VISIBLE_DEVICES=1
    CUDA_VISIBLE_DEVICES=0,1

**/

void QBase::init()
{
    INSTANCE = this ; 
    LOG(LEVEL) << "[ QU::UploadArray " ; 
    d_base = QU::UploadArray<qbase>(base,1) ; 
    LOG(LEVEL) << "] QU::UploadArray : takes ~0.25-0.3s : appearing in analog timings as it is first GPU contact " ; 
}

std::string QBase::desc() const 
{
    std::stringstream ss ; 
    ss << "QBase::desc"
       << " base " << base 
       << " d_base " << d_base 
       << " base.desc " << base->desc()
       ; 
    std::string s = ss.str(); 
    return s ; 
}



