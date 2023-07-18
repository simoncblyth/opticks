#pragma once
/**
s_mock_texture : exploring CUDA texture lookup API on CPU
=============================================================

The cudaTextureObject_t just probably typedef to unsigned long 
so its an "int" pointer. 


**/

#include <vector>
#include <iomanip>

#include <vector_types.h>
#include "NP.hh"
#include "scuda.h"

struct MockTexture
{
    NP* a ; 
    int width ; 
    int height ; 
    float4 dom ; 

    MockTexture(NP* a_, int width_, int height_ ); 

    std::string desc() const ; 
    template<typename T> T lookup(float x, float y ) const ; 
    template<typename T> std::string dump() const ; 
}; 

inline MockTexture::MockTexture(NP* a_, int width_, int height_ )
    :
    a(a_),
    width(width_),
    height(height_)
{
    dom.x = a->get_meta<float>("domain_low",  0.f );
    dom.y = a->get_meta<float>("domain_high",  0.f );
    dom.z = a->get_meta<float>("domain_step",  0.f );
    dom.w = a->get_meta<float>("domain_range", 0.f );
}

inline std::string MockTexture::desc() const 
{
    std::stringstream ss ; 
    ss << "MockTexture::desc"  
       << " a " << ( a ? a->sstr() : "-" )
       << " width " << width
       << " height " << height
       << " dom " << dom 
       ; 

    std::string str = ss.str(); 
    return str ; 
}

template<typename T> 
inline T MockTexture::lookup(float x, float y ) const
{
    const T* vv = a->cvalues<T>() ; 
    int nx = width ; 
    int ny = height ; 

    int ix = int(x*float(nx)) ;  // NB no subtraction of 0.5f to get match 
    int iy = int(y*float(ny)) ; 
    int idx = iy*nx + ix ; 

    T v0 = vv[idx] ;     // hmm should be interpolating between v0 and v1 presumably 
    //T v1 = vv[idx+1] ;  // can this go beyond the array ?
    return v0  ; 
}

template<typename T> 
inline std::string MockTexture::dump() const
{
    std::stringstream ss ; 
    ss << "MockTexture::dump<" <<  (sizeof(T) == 16 ? "float4" : "float" ) << ">" << std::endl ;    
    const T* vv = a->cvalues<T>() ; 
    for(int i=0 ; i < std::min(10, width*height) ; i++)
    {
        ss << " *(vv+" << std::setw(3) << i << ") : " << *(vv+i) << std::endl; 
    }
    std::string str = ss.str(); 
    return str ; 
}



struct MockTextureManager 
{
    static MockTextureManager* INSTANCE ; 
    static MockTextureManager* Get(); 
    static MockTexture Get(cudaTextureObject_t tex); 
    static cudaTextureObject_t Add(const NP* a, int width, int height ); 

    std::vector<MockTexture> tt ; 

    MockTextureManager() ;

    cudaTextureObject_t add( const NP* a, int width, int height ); 

    static std::string Desc(); 
    std::string desc() const ; 
    MockTexture get(cudaTextureObject_t tex ) const ; 

    template<typename T> T tex2D( cudaTextureObject_t t, float x, float y ) const  ; 

    float4 boundary_lookup(cudaTextureObject_t tex,  float nm, int line, int k ) const ; 

    template<typename T> std::string dump(cudaTextureObject_t tex) const ; 
};

MockTextureManager* MockTextureManager::INSTANCE = nullptr ; 

inline MockTextureManager* MockTextureManager::Get()  // static 
{
    return INSTANCE ; 
}

inline MockTextureManager::MockTextureManager()
{
    INSTANCE = this ; 
}

inline MockTexture MockTextureManager::Get(cudaTextureObject_t obj) // static
{
    assert(INSTANCE); 
    return INSTANCE->get(obj) ; 
}
inline cudaTextureObject_t MockTextureManager::Add(const NP* a, int width, int height )
{
    if(INSTANCE == nullptr) new MockTextureManager ; 
    assert(INSTANCE); 
    return INSTANCE->add(a, width, height); 
}

inline cudaTextureObject_t MockTextureManager::add(const NP* a_, int width, int height )
{
    NP* a = NP::MakeNarrowIfWide(a_) ;  // NB even if narrow already, still copies
    cudaTextureObject_t idx = tt.size() ; 
    MockTexture tex(a, width, height) ; 
    tt.push_back(tex); 
    return idx ; 
}

inline std::string MockTextureManager::Desc() // static
{
    return INSTANCE ? INSTANCE->desc() : "-" ; 
}
inline std::string MockTextureManager::desc() const 
{
    int num_tex = tt.size(); 
    std::stringstream ss ;
    ss << "MockTextureManager::desc num_tex " << num_tex << std::endl ; 
    for(int i=0 ; i < num_tex ; i++) ss << std::setw(4) << i << " : " << tt[i].desc() << std::endl ;  
    std::string str = ss.str(); 
    return str ; 
}

inline MockTexture MockTextureManager::get(cudaTextureObject_t t ) const
{
    assert( t < tt.size() ); 
    return tt[t] ; 
}

template<typename T> 
inline std::string MockTextureManager::dump( cudaTextureObject_t t ) const
{
    MockTexture tex = get(t) ; 
    return tex.dump<T>(); 
}

template<typename T> 
inline T MockTextureManager::tex2D( cudaTextureObject_t t, float x, float y ) const 
{
    MockTexture tex = get(t) ; 
    return tex.lookup<T>(x,y) ; 
}

template<typename T> T tex2D(cudaTextureObject_t t, float x, float y )
{
    return MockTextureManager::INSTANCE->tex2D<T>( t, x, y ); 
}


