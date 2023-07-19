#pragma once

/**
QBnd
=====

CUDA-centric equivalent for optixrap/OBndLib 

Lots of former bnd array and metadata related methods from QBnd 
were relocated down to SBnd as they were unrelated to CUDA, and 
in order to facilitate reuse from eg U4Recorder. 

* everything in QUDARap should be related to CUDA uploading, downloading, launching 
* anything data preparation related that is not using CUDA should be down in sysrap


TODO: consider combine QBnd and QOptical into QOpticalBnd or incorporating 
      QOptical within QBnd as bnd and optical are so closely related 
      and require coordinated changes when adding dynamic boundaries
      as done in SSim::addFake_ 

**/

#include <vector>
#include <string>

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
#include "plog/Severity.h"
#endif

#include "QUDARAP_API_EXPORT.hh"

union quad ; 
struct float4 ; 
struct dim3 ; 
struct qbnd ; 

template <typename T> struct QTex ; 
struct NP ; 
struct NPFold ; 
struct SBnd ; 

struct QUDARAP_API QBnd
{
#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
    static const plog::Severity LEVEL ;
#endif
    static const QBnd*          INSTANCE ; 
    static const QBnd*          Get(); 

    static qbnd* MakeInstance(const QTex<float4>* tex, const std::vector<std::string>& names ); 

    const NP*      dsrc ;  
    const NP*      src ;  
    SBnd*          sbn ; 

    QTex<float4>*  tex ; 

    qbnd*          qb ;    // formerly bnd 
    qbnd*          d_qb ;  // formerly d_bnd

    QBnd(const NP* buf); 
    void init(); 

    std::string desc() const ; 

    static QTex<float4>* MakeBoundaryTex(const NP* buf ) ;
    static void ConfigureLaunch( dim3& numBlocks, dim3& threadsPerBlock, int width, int height );
    static std::string DescLaunch( const dim3& numBlocks, const dim3& threadsPerBlock, int width, int height ); 

    NP*  lookup() const ;
    NPFold* serialize() const ; 
    void save(const char* dir) const ; 

    void lookup( quad* lookup, int num_lookup, int width, int height ) const ;
    static std::string Dump(   quad* lookup, int num_lookup, int edgeitems=10 );

};


