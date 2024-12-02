#include "SLOG.hh"
#include "QU.hh"
#include "QRng.hh"
#include "QCurandStateMonolithic.hh"
#include "SCurandStateMonolithic.hh"
#include "SLaunchSequence.h"
#include "ssys.h"

#include "qcurandstate.h"
#include "curand_kernel.h" 

const plog::Severity QCurandStateMonolithic::LEVEL = SLOG::EnvLevel("QCurandStateMonolithic", "DEBUG" ); 

extern "C" void QCurandStateMonolithic_curand_init(SLaunchSequence* lseq, qcurandstate* cs, qcurandstate* d_cs) ; 

QCurandStateMonolithic* QCurandStateMonolithic::Create(){ return Create(ssys::getenvvar(EKEY,"1:0:0")); }
QCurandStateMonolithic* QCurandStateMonolithic::Create(const char* spec)
{
    SCurandStateMonolithic scs(spec); 
    return new QCurandStateMonolithic(scs); 
}

QCurandStateMonolithic::QCurandStateMonolithic(const SCurandStateMonolithic& scs_)
    :
    scs(scs_),
    h_cs(new qcurandstate  { scs.num, scs.seed, scs.offset , nullptr} ),
    cs(new qcurandstate    { scs.num, scs.seed, scs.offset , nullptr} ),
    d_cs(nullptr),
    lseq(new SLaunchSequence(cs->num))
{
    init(); 
}

void QCurandStateMonolithic::init()
{
    LOG_IF(info, scs.exists) << "scs.path " << scs.path << " exists already : NOTHING TO DO " ;
    if(scs.exists) return ; 

    alloc();
    create(); 
    download(); 
    save(); 
}

void QCurandStateMonolithic::alloc()
{ 
    cs->states = QU::device_alloc_zero<curandState>(cs->num,"QCurandStateMonolithic::QCurandStateMonolithic/curandState") ; 
    LOG(info) << "after alloc" ; 
    d_cs = QU::UploadArray<qcurandstate>(cs, 1, "QCurandStateMonolithic::alloc/qcurandstate" );    
    LOG(info) << "after upload" ; 
}
void QCurandStateMonolithic::create() 
{
    QCurandStateMonolithic_curand_init(lseq, cs, d_cs); 
    LOG(info) << "after QCurandStateMonolithic_curand_init lseq.desc " << std::endl << lseq->desc() ; 
}
void QCurandStateMonolithic::download() 
{
    h_cs->states = (curandState*)malloc(sizeof(curandState)*cs->num);
    QU::copy_device_to_host( h_cs->states, cs->states, cs->num ); 
    LOG(info) << "after copy_device_to_host  "  ; 
}
void QCurandStateMonolithic::save() const 
{
    QRng::Save( h_cs->states, h_cs->num , scs.path.c_str() ); 
    LOG(info) << " saved to scs.path " << scs.path ; 
}

std::string QCurandStateMonolithic::desc() const 
{
    std::stringstream ss ; 
    ss << "QCurandStateMonolithic::desc"
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

