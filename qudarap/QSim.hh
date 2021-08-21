#pragma once

#include <string>
#include <vector>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

/**
QSim
======

QSim is mostly constant and needs initializing once only 
corresponding to the geometry and the physics process
implementations.  Perhaps call it QPhys or QSim ?

Contrast with the QEvent with a very 
different event-by-event lifecycle  

**/

struct NP ; 
template <typename T> struct QTex ; 
template <typename T> struct QProp ; 
template <typename T> struct qsim ; 

struct QRng ; 
struct QScint ;
struct QBnd ; 

struct quad4 ; 
union  quad ; 

template <typename T>
struct QUDARAP_API QSim
{
    static const plog::Severity LEVEL ; 
    static const char* PREFIX ; 
    static const QSim* INSTANCE ; 
    static const QSim* Get(); 

    static void UploadComponents(const NP* icdf, const NP* bnd );   

    const QRng*    rng ;     // need to template these too ?
    const QScint*  scint ; 
    const QBnd*    bnd ; 
    const QProp<T>*  prop ; 

    qsim<T>*          sim ;  
    qsim<T>*          d_sim ;  

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 

    QSim();
    qsim<T>* init_upload(); 
    qsim<T>* getDevicePtr() const ; 


    char getScintTexFilterMode() const ;

    std::string desc() const ; 

    void configureLaunch( unsigned width, unsigned height );
    void configureLaunch2D( unsigned width, unsigned height );

    void rng_sequence_0( T* rs, unsigned num_items );
    void rng_sequence( dim3 numblocks, dim3 threadsPerBlock, qsim<T>* d_sim, T* d_seq, unsigned ni_tranche, unsigned nv, unsigned ioffset );

    void rng_sequence( T* seq, unsigned ni, unsigned nj, unsigned ioffset ); 
    void rng_sequence( const char* dir, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size );


    void scint_wavelength(    T* wavelength, unsigned num_wavelength, unsigned& hd_factor ); 
    void cerenkov_wavelength( T* wavelength, unsigned num_wavelength ); 
    void dump_wavelength(     T* wavelength, unsigned num_wavelength, unsigned edgeitems=10 ); 

    // hmm need to template quad4 ? or narrowing on output ?
    void scint_photon(           quad4* photon, unsigned num_photon ); 
    void cerenkov_photon(        quad4* photon, unsigned num_photon, int print_id ) ; 
    void cerenkov_photon_enprop( quad4* photon, unsigned num_photon, int print_id ) ;
    void cerenkov_photon_expt(   quad4* photon, unsigned num_photon, int print_id ) ;

    void dump_photon(            quad4* photon, unsigned num_photon, unsigned egdeitems=10 ); 

    unsigned getBoundaryTexWidth() const ;
    unsigned getBoundaryTexHeight() const ;
    const NP* getBoundaryTexSrc() const ; 

    void boundary_lookup_all(  quad* lookup, unsigned width, unsigned height ) ; 
    void boundary_lookup_line( quad* lookup, T* domain, unsigned num_lookup, unsigned line, unsigned k ) ; 

    void prop_lookup(          T* lookup, const T* domain, unsigned domain_width, const std::vector<unsigned>& pids ) ;
    void prop_lookup_onebyone( T* lookup, const T* domain, unsigned domain_width, const std::vector<unsigned>& pids ) ;
};


