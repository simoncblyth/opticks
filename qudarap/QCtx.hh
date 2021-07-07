#pragma once

#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

/**
QCtx
======

TODO: 

1. genstep provisioning 

**/

class GGeo ; 
class GScintillatorLib ; 
class GBndLib ; 

template <typename T> class NPY ; 
template <typename T> struct QTex ; 

struct QRng ; 
struct QScint ;
struct QBnd ; 

struct qctx ; 
struct quad4 ; 
union  quad ; 

struct QUDARAP_API QCtx
{
    static const plog::Severity LEVEL ; 
    static const QCtx* INSTANCE ; 
    static const QCtx* Get(); 
    static QScint* MakeScint(const GScintillatorLib* slib);
    static void Init(const GGeo* ggeo); 

    const QRng*    rng ; 
    const QScint*  scint ; 
    const QBnd*    bnd ; 
    qctx*          ctx ;  
    qctx*          d_ctx ;  

    QCtx();
    void init(); 
    char getScintTexFilterMode() const ;

    std::string desc() const ; 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    void generate_scint(    float* wavelength, unsigned num_wavelength, unsigned& hd_factor ); 
    void generate_cerenkov( float* wavelength, unsigned num_wavelength ); 
    void dump(              float* wavelength, unsigned num_wavelength, unsigned edgeitems=10 ); 

    void generate( quad4* photon,     unsigned num_photon ); 
    void dump(     quad4* photon,     unsigned num_photon, unsigned egdeitems=10 ); 

    template<typename T> T* device_alloc( unsigned num_items ) ; 
    template<typename T> void copy_device_to_host( T* h, T* d,  unsigned num_items);


    unsigned getBoundaryTexWidth() const ;
    unsigned getBoundaryTexHeight() const ;
    const NPY<float>* getBoundaryTexSrc() const ; 

    void boundary_lookup(quad* lookup, unsigned width, unsigned height ) ; 


};
