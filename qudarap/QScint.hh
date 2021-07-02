#pragma once

#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

class GScintillatorLib ; 
template <typename T> class NPY ; 
template <typename T> struct QTex ; 
struct dim3 ; 

struct QUDARAP_API QScint
{
    static const plog::Severity LEVEL ; 
    static const QScint*        INSTANCE ; 
    static const QScint*        Get(); 

    const GScintillatorLib* slib ; 
    const NPY<double>*      dsrc ; 
    NPY<float>*             src ; 
    QTex<float>*            tex ; 

    QScint(const GScintillatorLib* slib_); 
    QScint(const NPY<double>* icdf); 

    void init(); 
    void makeScintTex(const NPY<float>* src);
    std::string desc() const ; 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    void check();
    NPY<float>* lookup();
    void lookup( float* lookup, unsigned num_lookup, unsigned width, unsigned height ); 
    void dump(   float* lookup, unsigned num_lookup, unsigned edgeitems=10 ); 

};


