#include "SLOG.hh"
#include "QU.hh"
#include "QRng.hh"
#include "QCurandState.hh"
#include "SCurandState.hh"
#include "SLaunchSequence.h"
#include "SSys.hh"
#include "SPath.hh"

#include "qcurandstate.h"
#include "curand_kernel.h" 

const plog::Severity QCurandState::LEVEL = SLOG::EnvLevel("QCurandState", "DEBUG" ); 

extern "C" void QCurandState_curand_init(SLaunchSequence* lseq, qcurandstate* cs, qcurandstate* d_cs) ; 

QCurandState* QCurandState::Create(){ return Create(SSys::getenvvar(EKEY,"1:0:0")); }
QCurandState* QCurandState::Create(const char* spec)
{
    SCurandState scs(spec); 
    return new QCurandState(scs); 
}

QCurandState::QCurandState(const SCurandState& scs_)
    :
    scs(scs_),
    h_cs(new qcurandstate  { scs.num, scs.seed, scs.offset , nullptr} ),
    cs(new qcurandstate    { scs.num, scs.seed, scs.offset, nullptr} ),
    d_cs(nullptr),
    lseq(new SLaunchSequence(cs->num))
{
    init(); 
}

void QCurandState::init()
{
    LOG_IF(info, scs.exists) << "scs.path " << scs.path << " exists already : NOTHING TO DO " ;
    if(scs.exists) return ; 



    alloc(); 
    create(); 
    download(); 
    save(); 
}

void QCurandState::alloc()
{ 
    cs->states = QU::device_alloc_zero<curandState>(cs->num,"QCurandState::QCurandState/curandState") ; 
    LOG(info) << "after alloc" ; 
    d_cs = QU::UploadArray<qcurandstate>(cs, 1 );    
    LOG(info) << "after upload" ; 
}
void QCurandState::create() 
{
    QCurandState_curand_init(lseq, cs, d_cs); 
    LOG(info) << "after QCurandState_curand_init lseq.desc " << std::endl << lseq->desc() ; 
}
void QCurandState::download() 
{
    h_cs->states = (curandState*)malloc(sizeof(curandState)*cs->num);
    QU::copy_device_to_host( h_cs->states, cs->states, cs->num ); 
    LOG(info) << "after copy_device_to_host  "  ; 
}
void QCurandState::save() const 
{
    QRng::Save( h_cs->states, h_cs->num , scs.path.c_str() ); 
    LOG(info) << " saved to scs.path " << scs.path ; 
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
       << " scs.path " << scs.path 
       ;
    std::string s = ss.str(); 
    return s ; 
}

