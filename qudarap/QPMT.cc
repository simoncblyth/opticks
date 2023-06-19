
#include <cuda_runtime.h>
#include <vector_types.h>

#include "SLOG.hh"
#include "NP.hh"
#include "QPMT.hh"

template<typename T>
const plog::Severity QPMT<T>::LEVEL = SLOG::EnvLevel("QPMT", "DEBUG"); 

template<typename T>
const NP* QPMT<T>::EDOMAIN = NP::Linspace<T>( 1.55, 15.50, 1550-155+1 ); //  np.linspace( 1.55, 15.50, 1550-155+1 )  


// NB this cannot be extern "C" as need C++ name mangling for template types

template <typename T>
extern void QPMT_rindex_interpolate(
    dim3 numBlocks,
    dim3 threadsPerBlock,
    qpmt<T>* pmt,
    T* lookup,
    const T* domain,
    unsigned domain_width
);


/**
QPMT::rindex_interpolate
--------------------------

lookup needs to energy_eV scan all pmt cat (3), layers (4) and props (2) (RINDEX, KINDEX)  
arrange that as three in kernel nested for loops (24 props) 
with the energy domain passed in as input so the parallelism is over the energy 

So the shape of the lookup output is  (3,4,2, domain_width )   

**/

template<typename T>
NP* QPMT<T>::rindex_interpolate( const NP* domain ) const 
{
    assert( domain->shape.size() == 1 && domain->shape[0] > 0 ); 
    unsigned domain_width = domain->shape[0] ; 
    
    T* d_domain = QU::UploadArray<T>( domain->cvalues<T>(), domain_width ) ; 

    const int& ni = qpmt<T>::NUM_CAT ; 
    const int& nj = qpmt<T>::NUM_LAYR ; 
    const int& nk = qpmt<T>::NUM_PROP ;  

    NP* lookup = NP::Make<T>( ni, nj, nk, domain_width ) ; 
    unsigned num_lookup = lookup->num_values() ; 

    LOG(LEVEL) 
        << " domain " << domain->sstr()
        << " domain_width " << domain_width 
        << " lookup " << lookup->sstr()
        << " num_lookup " << num_lookup 
        ;

    T* d_lookup = QU::device_alloc<T>(num_lookup,"QPMT<T>::rindex_interpolate::d_lookup") ;
   
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    QU::ConfigureLaunch1D( numBlocks, threadsPerBlock, domain_width, 512u ); 
    
    QPMT_rindex_interpolate(numBlocks, threadsPerBlock, d_pmt, d_lookup, d_domain, domain_width );

    QU::copy_device_to_host_and_free<T>( lookup->values<T>(), d_lookup, num_lookup );

    cudaDeviceSynchronize();  

    return lookup ; 
}

template<typename T>
NP* QPMT<T>::rindex_interpolate() const { return rindex_interpolate(EDOMAIN) ; }

// found the below can live in header, when headeronly 
//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wattributes"
// quell warning: type attributes ignored after type is already defined [-Wattributes]
template struct QUDARAP_API QPMT<float>;
template struct QUDARAP_API QPMT<double>;
//#pragma GCC diagnostic pop
 
