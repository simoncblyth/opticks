#pragma once
/**
QCurandState.h
===============

Aims to replace QCurandState.{hh,cc} with more flexible 
chunk based state handling using SCurandState.h 

This can be dynamic depending on runtime choice of maxphoton, 
also can partially load chunks to get the desired maxphoton
appropriate for VRAM. 

Related::

    ~/o/qudarap/tests/QCurandState_Test.sh
    ~/o/sysrap/tests/SCurandState_test.sh

**/

#include "SCurandState.h"
#include "SLaunchSequence.h"
#include "QU.hh"


extern "C" void QCurandState_curand_init_chunk(SLaunchSequence* lseq, scurandref* cr, scurandref* d_cr) ; 


struct _QCurandState
{
    static _QCurandState* Create(const char* _dir=nullptr); 

    _SCurandState cs = {} ;
 
    _QCurandState(const _SCurandState& _cs); 

    void init(); 
    void initChunk(SCurandChunk& c);

    std::string desc() const ; 

}; 



inline _QCurandState* _QCurandState::Create(const char* _dir)
{
    _SCurandState cs(_dir); 
    return new _QCurandState(cs); 
} 

inline _QCurandState::_QCurandState(const _SCurandState& _cs)
    :
    cs(_cs)
{
    init(); 
}

inline void _QCurandState::init()
{
    if(cs.is_complete()) return ; 
    int num_chunk = cs.chunk.size(); 
    std::cerr 
        << "_QCurandState::init"
        << " num_chunk " << num_chunk  
        << " cs.all.num " << cs.all.num
        << "\n"
        ;

    // HMM: THE OBJECTIVE HERE IS THE CHUNKED STATES IN FILES
    // NOT THE BIG CONTIGUOUS ARRAY OF STATES ON DEVICE
    // SO BETTER TO DO CHUNK BY CHUNK  ? 

    // cs.all.states = QU::device_alloc_zero<curandState>(cs.all.num,"QCurandState::init") ; 
    // cs.d_all = QU::UploadArray<SCurandChunk>(cs.all, 1, "QCurandState::init" );    

    for(int i=0 ; i < num_chunk ; i++)
    {
        SCurandChunk& c = cs.chunk[i]; 
        if(SCurandChunk::IsValid(c)) continue ;
        initChunk(c);  
    }
}


inline void _QCurandState::initChunk(SCurandChunk& c)
{
    scurandref* cr = &(c.data) ; 

    cr->states = QU::device_alloc_zero<curandState>(cr->num,"QCurandState::initChunk") ; 
    scurandref* d_cr = QU::UploadArray<scurandref>(cr, 1, "QCurandState::init" );    

    SLaunchSequence lseq(cr->num);

    QCurandState_curand_init_chunk(&lseq, cr, d_cr); 


}



inline std::string _QCurandState::desc() const 
{
    std::stringstream ss ; 
    ss << "_QCurandState::desc\n"
       << cs.desc() 
       ; 

    std::string str = ss.str() ; 
    return str ;
}





