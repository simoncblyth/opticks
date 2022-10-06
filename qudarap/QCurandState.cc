#include "SLOG.hh"
#include "QU.hh"
#include "QRng.hh"
#include "QCurandState.hh"
#include "SLaunchSequence.h"
#include "SSys.hh"
#include "SStr.hh"
#include "SPath.hh"

#include "qcurandstate.h"
#include "curand_kernel.h" 

const plog::Severity QCurandState::LEVEL = SLOG::EnvLevel("QCurandState", "DEBUG" ); 
const char* QCurandState::RNGDIR = SPath::Resolve("$RNGDir", DIRPATH ) ; 
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

QCurandState* QCurandState::Create(){ return Create(SSys::getenvvar(EKEY,"1:0:0")); }
QCurandState* QCurandState::Create(const char* spec)
{
    std::vector<int> ivec ; 
    SStr::ISplit(spec, ivec, ':' ); 
    unsigned num_vals = ivec.size(); 
    assert( num_vals > 0 && num_vals <= 3 ); 

    unsigned long long num    =  num_vals > 0 ? ivec[0] : 1 ; 
    unsigned long long seed   =  num_vals > 1 ? ivec[1] : 0 ; 
    unsigned long long offset =  num_vals > 2 ? ivec[2] : 0 ; 

    if(num <= 100) num *= 1000000 ; // num <= 100 assumed to be in millions  

    LOG(info) << " spec " << spec << " num " << num << " seed " << seed << " offset " << offset ; 
    return new QCurandState( num, seed, offset ); 
}

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
    bool exists = SPath::Exists(path.c_str()) ; 
    LOG_IF(info, exists) << "path " << path << " exists already : NOTHING TO DO " ;
    if(exists) return ; 

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

