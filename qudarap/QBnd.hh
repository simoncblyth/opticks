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
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"

union quad ; 
struct float4 ; 
struct dim3 ; 
struct qbnd ; 

template <typename T> struct QTex ; 
struct NP ; 
struct SBnd ; 

struct QUDARAP_API QBnd
{
    static const plog::Severity LEVEL ;
    static const QBnd*          INSTANCE ; 
    static const QBnd*          Get(); 

    static qbnd* MakeInstance( const QTex<float4>* tex, const std::vector<std::string>& names ); 

    const NP*      dsrc ;  
    const NP*      src ;  
    SBnd*          sbn ; 

    QTex<float4>*  tex ; 

    qbnd*          bnd ; 
    qbnd*          d_bnd ; 

    QBnd(const NP* buf); 

    std::string desc() const ; 
    static QTex<float4>* MakeBoundaryTex(const NP* buf ) ;
    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    NP*  lookup();
    void lookup( quad* lookup, unsigned num_lookup, unsigned width, unsigned height );
    void dump(   quad* lookup, unsigned num_lookup, unsigned edgeitems=10 );

};


