#pragma once
/**
QCurandState.h
===============

Aims to replace QCurandState.{hh,cc} with more flexible 
chunk based state handling using SCurandState.h 

Need to enable runtime choice of maxphoton, 
also can partially load chunks to get the desired maxphoton
appropriate for VRAM. 

The old impl split:

1. creation of the curandState file with QCurandState.{hh,cc}
2. loading+uploading of curandState file for simulation with QRng.{hh,cc}

That definitely has advantages, as considerations for the 
install time executables to prepare the curandState and for 
runtime usage of the files are very different. 

The chunk-centric impl follows the same split:

1. creation of chunked curandState files with QCurandState.h 
2. loading+uploading of chunked curandState files with QRng.{hh,cc} 

Related tests::

    ~/o/qudarap/tests/QCurandState_Test.sh
    ~/o/sysrap/tests/SCurandState_test.sh
    ~/o/qudarap/tests/QRngTest.sh

**/

#include "SCurandState.h"
#include "SLaunchSequence.h"
#include "QU.hh"


extern "C" void QCurandState_curand_init_chunk(SLaunchSequence* lseq, scurandref* cr, scurandref* d_cr) ; 


struct _QCurandState
{
    static _QCurandState* Create(const char* _dir=nullptr); 

    _SCurandState cs = {} ;
    _QCurandState(const char* _dir); 

    void init(); 
    void initChunk(SCurandChunk& c);

    std::string desc() const ; 

}; 



inline _QCurandState* _QCurandState::Create(const char* _dir)
{
    return new _QCurandState(_dir); 
} 

inline _QCurandState::_QCurandState(const char* _dir)
    :
    cs(_dir)
{
    init(); 
}


/**
_QCurandState::init
--------------------

Completeness means all the chunk files exist. 

Outcome of instanciation is a complete set of
chunk files. 

**/

inline void _QCurandState::init()
{
    if(cs.is_complete()) return ; 
    int num_chunk = cs.chunk.size(); 
    std::cerr 
        << "_QCurandState::init"
        << " num_chunk " << num_chunk  
        << "\n"
        ;

    for(int i=0 ; i < num_chunk ; i++)
    {
        SCurandChunk& c = cs.chunk[i]; 
        if(SCurandChunk::IsValid(c)) continue ;
        initChunk(c);  
    }
}


/**
_QCurandState::initChunk : generates curandState for chunk and saves to file
-------------------------------------------------------------------------------

1. prep sequence of launches needed for c.ref.num slots
2. allocate + zero space on device for c.ref.num curandState 
3. upload scurandref metadata struct 
4. launch curand_init kernel on device populating curandState buffer
5. allocate h_states on host 
6. copy device to host 
7. free on device
8. change c.ref.states to host r_states
9. save c.ref.states to file named according to scurandref chunk metadata values
10. free h_states 

**/


inline void _QCurandState::initChunk(SCurandChunk& c)
{
    scurandref* cr = &(c.ref) ; 

    SLaunchSequence lseq(cr->num);

    cr->states = QU::device_alloc_zero<curandState>(cr->num,"QCurandState::initChunk") ;
 
    scurandref* d_cr = QU::UploadArray<scurandref>(cr, 1, "QCurandState::initChunk" );    

    QCurandState_curand_init_chunk(&lseq, cr, d_cr); 

    curandState* h_states = (curandState*)malloc(sizeof(curandState)*cr->num);

    QU::copy_device_to_host( h_states, cr->states, cr->num ); 

    QU::device_free<curandState>(cr->states); 

    cr->states = h_states ; 

    c.save(cs.dir);

    free(h_states); 
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




