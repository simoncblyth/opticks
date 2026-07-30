#pragma once

#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"
#include "sblackbody.h"

struct dim3 ;
struct NP ;
template <typename T> struct QTex ;
struct qplanck ;

struct QUDARAP_API QPlanck
{
    static const plog::Severity LEVEL ;
    static const QPlanck*        INSTANCE ;
    static const QPlanck*        Get();

    static QTex<float>* MakeTex(const NP* src);
    static qplanck*     MakeInstance(const QTex<float>* tex);

    sblackbody<double> blackbody ;
    const NP*      bb_icdf ;

    const NP*      dsrc ;
    const NP*      src ;
    QTex<float>*   tex ;
    qplanck*       planck ;
    qplanck*       d_planck ;


    QPlanck();
    void init();
    float icdf_wavelength(float u) const;
    size_t setPhotonWavelength(NP* ph) const;


    std::string desc() const ;

/**
    static void ConfigureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );
    void check() const ;
    NP*  lookup() const ;
    void lookup( float* lookup, unsigned num_lookup, unsigned width, unsigned height ) const ;
    static void Dump(   float* lookup, unsigned num_lookup, unsigned edgeitems=10 );
**/


};


