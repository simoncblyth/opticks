#pragma once
/**
SRecorder.h
=========

To some extent this acts as a minimal selection of the 
full stree.h info needed to render

::

    ~/o/sysrap/tests/SRecorder_test.sh 

* OpenGL/CUDA interop-ing the triangle data is possible (but not straight off)

* TODO: solid selection eg skipping virtuals so can see PMT shapes 

* WIP: incorporate into standard workflow

  * treat SRecorder.h as sibling to stree.h within SSim.hh 
  * invoke the SRecorder.h creation from stree immediately after stree creation by U4Tree 

**/


#include "SGLFW.h"
#include "NP.hh"


struct SRecorder
{

    static constexpr const char* RPOS_SPEC = "4,GL_FLOAT,GL_FALSE,64,0,false";  
    bool dump ;

    NP* record;

            
    GLint   record_first; 
    GLsizei record_count;   // all step points across all photon

    float4 mn = {} ; 
    float4 mx = {} ; 
      
    float4 ce = {} ; 
    
    static SRecorder* Load(const char* record_path);
    SRecorder();

   
    void init_minmax2D() ; 
    void load(const char* record_path) ; 
 
    const float* get_mn() const ; 
    const float* get_mx() const ; 
    const float* get_ce() const ; 
    const float get_t0() const ; 
    const float get_t1() const ; 
    const float get_ts() const ; 

    
    void desc() const ;


};

inline void SRecorder::desc() const
{
    std::cout

        << "SRecorder.desc() "
        << std::endl   
        << std::setw(20) << " mn " << mn 
        << std::endl   
        << std::setw(20) << " mx " << mx 
        << std::endl   
        << std::setw(20) << " ce " << ce
        << std::endl
        << std::setw(20) << " record.sstr " << record->sstr()
        << std::endl
        << std::setw(20) << " record_first " << record_first
        << std::endl
        << std::setw(20) << " record_count " << record_count
        << std::endl
        ;
}



inline SRecorder* SRecorder::Load(const char* record_path)
{
    SRecorder* s = new SRecorder ;
    s->load(record_path);
    return s ;
}

inline void SRecorder::load(const char* record_path)
{
    
    // expect shape like (10000, 10, 4, 4) of type np.float32
    NP* _a = NP::Load(record_path) ;   
    record = NP::MakeNarrowIfWide(_a) ; 


    if(record==nullptr) std::cout << "FAILED to load RECORD_PATH [" << record_path << "]" << std::endl ;
    if(record==nullptr) std::cout << " CREATE IT WITH [TEST=make_record_array ~/o/sysrap/tests/sphoton_test.sh] " << std::endl ; 
    assert(record); 

    assert(record->shape.size() == 4);   
    bool is_compressed = record->ebyte == 2 ; 
    assert( is_compressed == false ); 


    record_first = 0 ; 
    record_count = record->shape[0]*record->shape[1] ;   // all step points across all photon
}    

inline SRecorder::SRecorder()
    :
    dump(ssys::getenvbool("SRecorder_dump"))
{
}

inline void SRecorder::init_minmax2D(){
     
    static const int N = 4 ;   

    int item_stride = 4 ; 
    int item_offset = 0 ; 

    record->minmax2D_reshaped<N,float>(&mn.x, &mx.x, item_stride, item_offset ); 
    // temporarily 2D array with item: 4-element space-time points

    // HMM: with sparse "post" cloud this might miss the action
    // by trying to see everything ? 

    ce = scuda::center_extent( mn, mx ); 

}



inline const float* SRecorder::get_mn() const 
{
    return &mn.x ;  
} 
inline const float* SRecorder::get_mx() const 
{
    return &mx.x ;  
} 
inline const float* SRecorder::get_ce() const 
{
    return &ce.x ;  
} 

inline const float SRecorder::get_t0() const
{
    float t0 = ssys::getenvfloat("T0", mn.w ); 
    return t0;
}

inline const float SRecorder::get_t1() const
{
    float t1 = ssys::getenvfloat("T1", mx.w ); 
    return t1;
}

inline const float SRecorder::get_ts() const
{
    float ts = ssys::getenvfloat("TS", 5000. ); 
    return ts;
}









