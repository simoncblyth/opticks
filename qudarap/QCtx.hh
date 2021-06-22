#pragma once

#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

/**
QCtx
======

TODO: 

1. genstep provisioning 

**/

class GScintillatorLib ; 
class GBndLib ; 

template <typename T> class NPY ; 
template <typename T> struct QTex ; 
struct QRng ; 
struct quad4 ; 

struct QUDARAP_API QCtx
{
    static const plog::Severity LEVEL ; 

    const QRng*             rng ; 
    QTex<float>*            scint_tex ; 
    QTex<float>*            boundary_tex ; 

    QCtx();

    void upload(const GGeo* ggeo); 
    void uploadScintTex(const GScintillatorLib* slib); 
    void uploadBoundaryTex(const GBndLib* blib); 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    void generate( float* wavelength, unsigned num_wavelength ); 
    void dump(     float* wavelength, unsigned num_wavelength, unsigned edgeitems=10 ); 

    void generate( quad4* photon,     unsigned num_photon ); 
    void dump(     quad4* photon,     unsigned num_photon, unsigned egdeitems=10 ); 
};

