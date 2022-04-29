#pragma once

#include <string>
#include <vector>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

union quad;
struct dim3 ; 
struct NP ; 
struct float4;
struct qmultifilmlut;
template <typename T> struct QTex ; 

struct QUDARAP_API QMultiFilmLUT
{
    static const plog::Severity LEVEL ; 
    static const QMultiFilmLUT*        INSTANCE ; 
    static const QMultiFilmLUT*        Get(); 

    const NP*      dsrc ; 
    const NP*      src ; 

    /* 
       3 PMT Type   
       each PMT :  boundary *  resolution * wavelength * aoi * payload 
    */
    QTex<float4>  * tex_nnvt_normal[4];
    QTex<float4>  * tex_nnvt_highqe[4];
    QTex<float4>  * tex_hama[4];
    
    /* 4 = 2 * 2 ( boundary , resolution)     */
   
    void makeMultiFilmAllTex();
    void makeMultiFilmOnePMTTex(int pmtcatIdx , QTex<float4> ** tex_pmt);
    QTex<float4>* makeMultiFilmOneTex(int pmtcatIdx , int bndIdx , int resIdx);


    QMultiFilmLUT(const NP* lut ); 

    void init(); 
    void uploadMultifilmlut();
    qmultifilmlut* getDevicePtr() const;
    qmultifilmlut* multifilmlut;
    qmultifilmlut* d_multifilmlut;
    
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


