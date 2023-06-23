/**
QPMT.cc
==========

QPMT::interpolate
    prep (etype, domain) kernel calls 


**/

#include <cuda_runtime.h>
#include <vector_types.h>

#include "SLOG.hh"
#include "NP.hh"
#include "QPMT.hh"

template<typename T>
const plog::Severity QPMT<T>::LEVEL = SLOG::EnvLevel("QPMT", "DEBUG"); 

// NB this cannot be extern "C" as need C++ name mangling for template types

template <typename T>
extern void QPMT_lpmtcat(
    dim3 numBlocks,
    dim3 threadsPerBlock,
    qpmt<T>* pmt,
    int etype, 
    T* lookup,
    const T* domain,
    unsigned domain_width
);

template <typename T>
extern void QPMT_lpmtid(
    dim3 numBlocks,
    dim3 threadsPerBlock,
    qpmt<T>* pmt,
    int etype, 
    T* lookup,
    const T* domain,
    unsigned domain_width,
    const int* lpmtid, 
    unsigned num_lpmtid 
);


template<typename T>
void QPMT<T>::lpmtcat_check( int etype, const NP* domain, const NP* lookup) const 
{

    assert( domain->shape.size() == 1 && domain->shape[0] > 0 ); 
    unsigned num_domain = domain->shape[0] ; 
    unsigned num_domain_1 = 0 ; 

    if( etype == qpmt<T>::RINDEX || etype == qpmt<T>::QESHAPE )
    {
        num_domain_1 = lookup->shape[lookup->shape.size()-1] ; 
    } 
    else if ( etype == qpmt<T>::LPMTCAT_STACKSPEC )
    {
        num_domain_1 = lookup->shape[lookup->shape.size()-3] ;  // (4,4) payload
    }
    assert( num_domain == num_domain_1 ); 
}



/**
QPMT::lpmtcat_
--------------------

lookup needs to energy_eV scan all pmt cat (3), layers (4) and props (2) (RINDEX, KINDEX)  
arrange that as three in kernel nested for loops (24 props) 
with the energy domain passed in as input so the parallelism is over the energy 

So the shape of the lookup output is  (3,4,2, domain_width )   

**/

template<typename T>
NP* QPMT<T>::lpmtcat_(int etype, const NP* domain ) const 
{
    unsigned num_domain = domain->shape[0] ; 
    NP* lookup = MakeLookup_lpmtcat(etype, num_domain ); 
    lpmtcat_check(etype, domain, lookup) ; 
    unsigned num_lookup = lookup->num_values() ; 

    const T* d_domain = QU::UploadArray<T>( domain->cvalues<T>(), num_domain ) ; 


    LOG(LEVEL) 
        << " etype " << etype 
        << " domain " << domain->sstr()
        << " num_domain " << num_domain 
        << " lookup " << lookup->sstr()
        << " num_lookup " << num_lookup 
        ;

    T* h_lookup = lookup->values<T>() ; 
    T* d_lookup = QU::device_alloc<T>(num_lookup,"QPMT<T>::lpmtcat::d_lookup") ;
   
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    QU::ConfigureLaunch1D( numBlocks, threadsPerBlock, num_domain, 512u ); 
    
    QPMT_lpmtcat(numBlocks, threadsPerBlock, d_pmt, etype, d_lookup, d_domain, num_domain );

    QU::copy_device_to_host_and_free<T>( h_lookup, d_lookup, num_lookup );
    cudaDeviceSynchronize();  

    return lookup ; 
}


template<typename T>
NP* QPMT<T>::lpmtid_(int etype, const NP* domain, const NP* lpmtid ) const 
{
    unsigned num_domain = domain->shape[0] ; 
    unsigned num_lpmtid = lpmtid->shape[0] ; 

    NP* lookup = MakeLookup_lpmtid(etype, num_domain, num_lpmtid ); 

    unsigned num_lookup = lookup->num_values() ; 
    T* h_lookup = lookup->values<T>() ; 
    T* d_lookup = QU::device_alloc<T>(num_lookup,"QPMT<T>::lpmtid::d_lookup") ;
 
    assert( lpmtid->uifc == 'i' && lpmtid->ebyte == 4 ); 

    const T*   d_domain = QU::UploadArray<T>(   domain->cvalues<T>(), num_domain ) ; 
    const int* d_lpmtid = QU::UploadArray<int>( lpmtid->cvalues<int>(), num_lpmtid ) ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    QU::ConfigureLaunch1D( numBlocks, threadsPerBlock, num_domain, 512u ); 
    
    QPMT_lpmtid(numBlocks, threadsPerBlock, d_pmt, etype, d_lookup, d_domain, num_domain, d_lpmtid, num_lpmtid );

    QU::copy_device_to_host_and_free<T>( h_lookup, d_lookup, num_lookup );
    cudaDeviceSynchronize();  

    return lookup ; 
}



// found the below can live in header, when headeronly 
//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wattributes"
// quell warning: type attributes ignored after type is already defined [-Wattributes]
template struct QUDARAP_API QPMT<float>;
//template struct QUDARAP_API QPMT<double>;
//#pragma GCC diagnostic pop
 
