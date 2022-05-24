#pragma once

#include "scuda.h"
#include "sqat4.h"

struct sframe
{
    static constexpr const unsigned NUM_VALUES = 3*4*4 ; 
    static constexpr const char* NAME = "sframe.npy" ; 

    static sframe Load(const char* dir, const char* name=NAME); 

    float4 ce = {} ; 
    quad4  q1 = {} ; // padding for simplicity of persisting and usage from numpy 
    quad4  q2 = {} ; 
    quad4  q3 = {} ; 

    qat4   m2w ; 
    qat4   w2m ; 

    float* data() ; 
    const float* cdata() const ; 

    void write( float* dst, unsigned num_values ) const ;
    void save(const char* dir, const char* name=NAME) const ; 

    void read( const float* src, unsigned num_values ) ; 
    void load(const char* dir, const char* name=NAME) ; 

}; 

#include <cassert>
#include "NP.hh"

inline sframe sframe::Load(const char* dir, const char* name)
{
    sframe fr ; 
    fr.load(dir, name); 
    return fr ; 
}

inline const float* sframe::cdata() const 
{
    return (const float*)&ce.x ;  
}
inline float* sframe::data()  
{
    return (float*)&ce.x ;  
}
inline void sframe::write( float* dst, unsigned num_values ) const 
{
    assert( num_values == NUM_VALUES ); 
    char* dst_bytes = (char*)dst ; 
    char* src_bytes = (char*)cdata(); 
    unsigned num_bytes = sizeof(float)*num_values ; 
    memcpy( dst_bytes, src_bytes, num_bytes );
}    

inline void sframe::read( const float* src, unsigned num_values ) 
{
    assert( num_values == NUM_VALUES ); 
    char* src_bytes = (char*)src ; 
    char* dst_bytes = (char*)data(); 
    unsigned num_bytes = sizeof(float)*num_values ; 
    memcpy( dst_bytes, src_bytes, num_bytes );
}    

inline void sframe::save(const char* dir, const char* name) const
{
    NP* a = NP::Make<float>(3, 4, 4) ; 
    write( a->values<float>(), 3*4*4 ) ; 
    a->save(dir, name); 
}
inline void sframe::load(const char* dir, const char* name) 
{
    NP* a = NP::Load(dir, name); 
    read( a->values<float>() , NUM_VALUES );   
}

inline std::ostream& operator<<(std::ostream& os, const sframe& fr)
{
    os 
       << " ce  " << fr.ce << std::endl 
       << " m2w " << fr.m2w << std::endl 
       << " w2m " << fr.w2m << std::endl 
       ;
    return os; 
}


