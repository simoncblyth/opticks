#include "SLOG.hh"
#include "QU.hh"
#include "QRng.hh"
#include "QCurandState.hh"
#include "SLaunchSequence.h"
#include "SPath.hh"

#include "qcurandstate.h"
#include "curand_kernel.h" 

const plog::Severity QCurandState::LEVEL = SLOG::EnvLevel("QCurandState", "DEBUG" ); 
const char* QCurandState::RNGDIR = SPath::GetHomePath(".opticks/rngcache/RNG") ; 
const char* QCurandState::NAME_PREFIX = "QCurandState" ; 

std::string QCurandState::Stem(unsigned long long num, unsigned long long seed, unsigned long long offset)
{
    std::stringstream ss ; 
    ss << NAME_PREFIX << "_" << num << "_" << seed << "_" << offset  ; 
    std::string s = ss.str(); 
    return s ;  
} 
std::string QCurandState::Path(unsigned long long num, unsigned long long seed, unsigned long long offset)
{
    std::stringstream ss ; 
    ss << RNGDIR << "/" << Stem(num, seed, offset) << ".bin" ; 
    std::string s = ss.str(); 
    return s ;  
}

extern "C" void QCurandState_curand_init(SLaunchSequence* lseq, qcurandstate* cs, qcurandstate* d_cs) ; 

QCurandState::QCurandState(unsigned long long num_, unsigned long long seed_, unsigned long long offset_)
    :
    path(Path(num_, seed_, offset_ )), 
    h_cs(new qcurandstate { num_, seed_, offset_, nullptr} ),
    cs(new qcurandstate { num_, seed_, offset_, nullptr} ),
    d_cs(nullptr),
    lseq(new SLaunchSequence(cs->num))
{
    init(); 
}

void QCurandState::init()
{
    if(SPath::Exists(path.c_str()))
    {
        LOG(info) << " path " << path << " exists already : NOTHING TO DO " ; 
        return ; 
    }

    cs->states = QU::device_alloc_zero<curandState>(cs->num,"QCurandState::QCurandState/curandState") ; 

    LOG(info) << "after alloc" ; 

    d_cs = QU::UploadArray<qcurandstate>(cs, 1 );    

    LOG(info) << "after upload" ; 

    QCurandState_curand_init(lseq, cs, d_cs); 

    LOG(info) << "after QCurandState_curand_init lseq.desc " << std::endl << lseq->desc() ; 

    h_cs->states = (curandState*)malloc(sizeof(curandState)*cs->num);

    QU::copy_device_to_host( h_cs->states, cs->states, cs->num ); 

    LOG(info) << "after copy_device_to_host  "  ; 

    save(); 
}

void QCurandState::save() const 
{
    QRng::Save( h_cs->states, h_cs->num , path.c_str() ); 
    LOG(info) << " saved to " << path ; 
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
       << " path " << path 
       ;
    std::string s = ss.str(); 
    return s ; 
}

