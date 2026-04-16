#pragma once

#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

struct dim3 ;
struct NP ;
template <typename T> struct QTex ;
struct qscint ;

struct QUDARAP_API QScint
{
    static const plog::Severity LEVEL ;
    static const QScint*        INSTANCE ;
    static const QScint*        Get();

    static QTex<float>* MakeScintTex(const NP* src, unsigned hd_factor);
    static qscint* MakeInstance(const QTex<float>* tex);


    const NP*      dsrc ;
    const NP*      src ;
    QTex<float>*    tex ;
    qscint*       scint ;
    qscint*       d_scint ;


    QScint(const NP* icdf, unsigned hd_factor);
    void init();


    std::string desc() const ;

    static void ConfigureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    void check() const ;
    NP*  lookup() const ;

    void lookup( float* lookup, unsigned num_lookup, unsigned width, unsigned height ) const ;


    static void Dump(   float* lookup, unsigned num_lookup, unsigned edgeitems=10 );

};


