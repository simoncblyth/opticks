#pragma once

/**
sframe.h
===========

Persisted into (3,4,4) array.
Any extension should be in quad4 blocks 
for persisting, alignment and numpy convenience

Note that some variables like *frs* are
persisted in metadata, not in the array. 

Currently *frs* is usually the same as *moi* from MOI envvar
but are using *frs* to indicate intension for generalization 
to frame specification using global instance index rather than MOI
which uses the gas specific instance index. 

**/

#include <cassert>
#include <vector>
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "NP.hh"


struct sframe
{
    static constexpr const char* NAME = "sframe.npy" ; 
    static sframe Load(const char* dir, const char* name=NAME); 
    static constexpr const unsigned NUM_VALUES = 3*4*4 ; 

    float4 ce = {} ; 
    quad   q1 = {} ; 
    quad   q2 = {} ; 
    quad   q3 = {} ; 

    qat4   m2w ; 
    qat4   w2m ; 


    // on the edge, the above are memcpy in/out by load/save
    const char* frs = nullptr ; 

    void set_grid(const std::vector<int>& cegs, float gridscale); 
    int ix0() const ; 
    int ix1() const ; 
    int iy0() const ; 
    int iy1() const ; 
    int iz0() const ; 
    int iz1() const ; 
    int num_photon() const ; 
    float gridscale() const ; 


    void set_midx_mord_iidx(int midx, int mord, int iidx); 
    int midx() const ; 
    int mord() const ; 
    int iidx() const ; 

    float* data() ; 
    const float* cdata() const ; 

    void write( float* dst, unsigned num_values ) const ;
    NP* make_array() const ; 
    void save(const char* dir, const char* name=NAME) const ; 

    void read( const float* src, unsigned num_values ) ; 
    void load(const char* dir, const char* name=NAME) ; 

}; 

inline sframe sframe::Load(const char* dir, const char* name)
{
    sframe fr ; 
    fr.load(dir, name); 
    return fr ; 
}

inline void sframe::set_grid(const std::vector<int>& cegs, float gridscale)
{
    assert( cegs.size() == 7 );   // use QEvent::StandardizeCEGS to convert 4 to 7  

    q1.i.x = cegs[0] ;  // ix0   these are after standardization
    q1.i.y = cegs[1] ;  // ix1
    q1.i.z = cegs[2] ;  // iy0 
    q1.i.w = cegs[3] ;  // iy1

    q2.i.x = cegs[4] ;  // iz0
    q2.i.y = cegs[5] ;  // iz1 
    q2.i.z = cegs[6] ;  // num_photon
    q2.f.w = gridscale ; 
}

inline int sframe::ix0() const { return q1.i.x ; }
inline int sframe::ix1() const { return q1.i.y ; }
inline int sframe::iy0() const { return q1.i.z ; }
inline int sframe::iy1() const { return q1.i.w ; }
inline int sframe::iz0() const { return q2.i.x ; }
inline int sframe::iz1() const { return q2.i.y ; }
inline int sframe::num_photon() const { return q2.i.z ; }
inline float sframe::gridscale() const { return q2.f.w ; }


inline void sframe::set_midx_mord_iidx(int midx, int mord, int iidx)
{
    q3.i.x = midx ; 
    q3.i.y = mord ; 
    q3.i.z = iidx ; 
    q3.i.w = 0 ; 
}

inline int sframe::midx() const { return q3.i.x ; }
inline int sframe::mord() const { return q3.i.y ; }
inline int sframe::iidx() const { return q3.i.z ; }


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

inline NP* sframe::make_array() const 
{
    NP* a = NP::Make<float>(3, 4, 4) ; 
    write( a->values<float>(), 3*4*4 ) ; 
    return a ; 
}
inline void sframe::save(const char* dir, const char* name) const
{
    NP* a = make_array(); 
    a->set_meta<std::string>("creator", "sframe::save"); 
    if(frs) a->set_meta<std::string>("frs", frs); 
    a->save(dir, name); 
}
inline void sframe::load(const char* dir, const char* name) 
{
    NP* a = NP::Load(dir, name); 
    read( a->values<float>() , NUM_VALUES );   

    std::string _frs = a->get_meta<std::string>("frs", ""); 
    if(!_frs.empty()) frs = strdup(_frs.c_str()); 
}

inline std::ostream& operator<<(std::ostream& os, const sframe& fr)
{
    os 
       << " frs " << ( fr.frs ? fr.frs : "-" ) << std::endl 
       << " ce  " << fr.ce 
       << std::endl 
       << " m2w " << fr.m2w 
       << std::endl 
       << " w2m " << fr.w2m 
       << std::endl 
       << " midx " << std::setw(4) << fr.midx()
       << " mord " << std::setw(4) << fr.mord()
       << " iidx " << std::setw(4) << fr.iidx()
       << std::endl 
       << " ix0  " << std::setw(4) << fr.ix0()
       << " ix1  " << std::setw(4) << fr.ix1()
       << " iy0  " << std::setw(4) << fr.iy0()
       << " iy1  " << std::setw(4) << fr.iy1()
       << " iz0  " << std::setw(4) << fr.iz0()
       << " iz1  " << std::setw(4) << fr.iz1()
       << " num_photon " << std::setw(4) << fr.num_photon()
       << std::endl 
 
       ;
    return os; 
}


