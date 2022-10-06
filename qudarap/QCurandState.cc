#include "SLOG.hh"

#include "QU.hh"
#include "QCurandState.hh"

#include "qcurandstate.h"
#include "curand_kernel.h" 

const plog::Severity QCurandState::LEVEL = SLOG::EnvLevel("QCurandState", "DEBUG" ); 

extern "C" void QCurandState_curand_init(qcurandstate* cs, qcurandstate* d_cs) ; 

QCurandState::QCurandState(unsigned long long num_, unsigned long long seed_, unsigned long long offset_)
    :
    cs(nullptr),
    d_cs(nullptr)
{
    cs = new qcurandstate ; 
    cs->num = num_ ; 
    cs->seed = seed_ ; 
    cs->offset = offset_ ; 

    LOG(info) << "init" ; 

    cs->states = QU::device_alloc_zero<curandState>(cs->num,"QCurandState::QCurandState/curandState") ; 

    LOG(info) << "after alloc" ; 

    d_cs = QU::UploadArray<qcurandstate>(cs, 1 );    

    LOG(info) << "after upload" ; 

    QCurandState_curand_init(cs, d_cs); 

    LOG(info) << "after QCurandState_curand_init " ; 
}

std::string QCurandState::desc() const 
{
    std::stringstream ss ; 
    ss << "QCurandState::desc"
       << " cs.num " << cs->num 
       << " cs.seed " << cs->seed 
       << " cs.offset " << cs->offset 
       << " cs.states " << cs->states 
       << " d_cs " << d_cs 
       ;
    std::string s = ss.str(); 
    return s ; 
}

