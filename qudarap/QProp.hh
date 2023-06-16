#pragma once
/**
QProp : setup to allow direct (no texture) interpolated property access on device 
=====================================================================================

See NP::Combine for the construction of compound prop array 
from many indiviual prop arrays with various domain lengths 
and differing domain values 

See ~/np/tests/NPInterp.py for prototyping the linear interpolation 

QProp vs textures
---------------------

GPU texture interpolation does something very similar to qprop::interpolate 
but QProp has the advantage that it can handle multiple properties all with 
different numbers of items. This removes the need with textures to  establish 
a common domain and pre-interpolate everything to use that domain. 

The QProp approach is very close to what Geant4 does, the texture approach 
is likely faster but takes more effort to setup and probably requires fine
textures to reproduce the Geant4 results. 

**/

#include <vector>
#include <string>
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"

union quad ; 
struct float4 ; 
struct dim3 ; 
template <typename T> struct qprop ; 
struct NP ; 

template <typename T>
struct QUDARAP_API QProp
{
    static const plog::Severity LEVEL ;
    static const QProp<T>*  INSTANCE ; 
    static const QProp<T>*  Get(); 

    const NP* a  ;  
    const T* pp ; 
    unsigned nv ; 

    unsigned ni ; 
    unsigned nj ; 
    unsigned nk ; 

    qprop<T>* prop ; 
    qprop<T>* d_prop ; 

    // scrunch the high dimensions yielding (num_prop, num_energy, 2)
    //static QProp<T>* Make3D( const NP* a );  
    QProp(const NP* a); 

    virtual ~QProp(); 
    void init(); 
    void upload(); 
    void cleanup(); 

    void dump() const ; 
    std::string desc() const ;
    qprop<T>* getDevicePtr() const ;
    void lookup( T* lookup, const T* domain,  unsigned num_prop, unsigned domain_width ) const ; 
    void lookup_scan(T x0, T x1, unsigned nx, const char* fold, const char* reldir=nullptr ) const ; 

    //void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height ) const ;
};


