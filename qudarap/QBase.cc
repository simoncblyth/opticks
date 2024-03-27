#include <sstream>
#include <limits>

#include "ssys.h"

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
#else
#include "SLOG.hh"
#include "QU.hh"
#endif

#include "QBase.hh"
#include "qbase.h"

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
const plog::Severity QBase::LEVEL = plog::info ; 
#else
const plog::Severity QBase::LEVEL = SLOG::EnvLevel("QBase", "DEBUG"); 
#endif

const QBase* QBase::INSTANCE = nullptr ; 
const QBase* QBase::Get(){ return INSTANCE ; }

qbase* QBase::MakeInstance() // static 
{
    qbase* base = new qbase ; 
    base->pidx = ssys::getenvunsigned_fallback_max("PIDX") ; 
    base->custom_lut = ssys::getenvunsigned("QBase__CUSTOM_LUT", 0u);
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

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    d_base = base ; 
#else
    LOG(LEVEL) << "[ QU::UploadArray " ; 
    d_base = QU::UploadArray<qbase>(base,1, "QBase::init/d_base") ; 
    LOG(LEVEL) << "] QU::UploadArray : takes ~0.25-0.3s : appearing in analog timings as it is first GPU contact " ; 
#endif


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


