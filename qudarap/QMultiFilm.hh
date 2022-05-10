#pragma once
/**
QMultiFilm.hh
===============

For the 3 PMT Types : boundary * resolution * wavelength * aoi * payload 

**/

#include <string>
#include <vector>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

union quad;
struct dim3 ; 
struct NP ; 
struct float4;
struct qmultifilm ;
template <typename T> struct QTex ; 

struct QUDARAP_API QMultiFilm
{
    static const plog::Severity LEVEL ; 
    static const QMultiFilm*        INSTANCE ; 
    static const QMultiFilm*        Get(); 

    const NP*      dsrc ; 
    const NP*      src ; 
    QTex<float4>* tex_nnvt_normal[4];
    QTex<float4>* tex_nnvt_highqe[4];
    QTex<float4>* tex_hama[4];

    qmultifilm*   multifilm ;
    qmultifilm*   d_multifilm ;
 
    /* 4 = 2 * 2 ( boundary , resolution)     */
   
    void makeMultiFilmAllTex();
    void makeMultiFilmOnePMTTex(int pmtcatIdx , QTex<float4> ** tex_pmt);
    QTex<float4>* makeMultiFilmOneTex(int pmtcatIdx , int bndIdx , int resIdx);


    QMultiFilm(const NP* lut ); 

    void init(); 
    void uploadMultifilmlut();
    qmultifilm* getDevicePtr() const;
    std::string desc() const ; 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    void check();
    void check(QTex<float4>* tex);
    NP*  lookup(int pmtcatIdx , int bndIdx , int resIdx);
    NP*  lookup(QTex<float4>* tex);

    void lookup( QTex<float4>* tex, float4* lookup, unsigned num_lookup, unsigned width, unsigned height); 
    void dump(   float4* lookup, unsigned num_lookup, unsigned edgeitems=10 ); 
  
    float4 multifilm_lookup( unsigned pmtType, unsigned boundary, float nm, float aoi );
    QTex<float4> ** choose_tex(int pmtcatIdx);
};


