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

1. creation of the RNG file with QCurandState.{hh,cc}
2. loading+uploading of RNG file for simulation with QRng.{hh,cc}

That definitely has advantages, as considerations for the 
install time executables to prepare the RNG and for 
runtime usage of the files are very different. 

The chunk-centric impl follows the same split:

1. creation of chunked RNG files with QCurandState.h 
2. loading+uploading of chunked RNG files with QRng.{hh,cc} 

Related tests::

    ~/o/qudarap/tests/QCurandState_Test.sh
    ~/o/sysrap/tests/SCurandState_test.sh
    ~/o/qudarap/tests/QRngTest.sh

**/

#include "SCurandState.h"
#include "SLaunchSequence.h"
#include "QU.hh"

#include "qrng.h"


extern "C" void QCurandState_curand_init_chunk(SLaunchSequence* lseq, scurandref<XORWOW>* cr, scurandref<XORWOW>* d_cr) ; 


struct QCurandState
{
    static QCurandState* Create(const char* _dir=nullptr); 

    SCurandState cs = {} ;
    QCurandState(const char* _dir); 

    void init(); 
    void initChunk(SCurandChunk& c);

    std::string desc() const ; 
}; 


inline QCurandState* QCurandState::Create(const char* _dir)
{
    std::cerr << "[QCurandState::Create\n" ; 
    QCurandState* qcs = new QCurandState(_dir); 
    std::cerr << "]QCurandState::Create\n" ; 
    return qcs ; 
} 

inline QCurandState::QCurandState(const char* _dir)
    :
    cs(_dir)
{
    init(); 
}


/**
QCurandState::init
--------------------

Completeness means all the chunk files exist. 

Outcome of instanciation is a complete set of
chunk files. 

**/

inline void QCurandState::init()
{
    int num_chunk = cs.chunk.size(); 
    bool complete = cs.is_complete() ; 
    std::cerr 
        << "QCurandState::init"
        << " cs.chunk.size " << num_chunk  
        << " is_complete " << ( complete ? "YES" : "NO " ) 
        << "\n"
        ;

    if(complete) return ; 

    for(int i=0 ; i < num_chunk ; i++)
    {
        SCurandChunk& c = cs.chunk[i]; 
        if(SCurandChunk::IsValid(c)) continue ;
        initChunk(c);  
    }
}


/**
QCurandState::initChunk : generates RNG for chunk and saves to file
-------------------------------------------------------------------------------

1. prep sequence of launches needed for c.ref.num slots
2. allocate + zero space on device for c.ref.num RNG 
3. upload scurandref<XORWOW> metadata struct 
4. launch curand_init kernel on device populating RNG buffer
5. allocate h_states on host 
6. copy device to host 
7. free on device
8. change c.ref.states to host r_states
9. save c.ref.states to file named according to scurandref<XORWOW> chunk metadata values
10. free h_states 

**/


inline void QCurandState::initChunk(SCurandChunk& c)
{
    scurandref<XORWOW>* cr = &(c.ref) ; 

    SLaunchSequence lseq(cr->num);

    cr->states = QU::device_alloc_zero<XORWOW>(cr->num,"QCurandState::initChunk") ;
 
    scurandref<XORWOW>* d_cr = QU::UploadArray<scurandref<XORWOW>>(cr, 1, "QCurandState::initChunk" );    

    QCurandState_curand_init_chunk(&lseq, cr, d_cr); 

    XORWOW* h_states = (XORWOW*)malloc(sizeof(XORWOW)*cr->num);

    QU::copy_device_to_host( h_states, cr->states, cr->num ); 

    QU::device_free<XORWOW>(cr->states); 

    cr->states = h_states ; 

    c.save(cs.dir);

    free(h_states); 
}



inline std::string QCurandState::desc() const 
{
    std::stringstream ss ; 
    ss << "QCurandState::desc\n"
       << cs.desc() 
       ; 

    std::string str = ss.str() ; 
    return str ;
}




