/**
QPMT.cu
==========
**/


#include "QUDARAP_API_EXPORT.hh"
#include <stdio.h>
#include "qpmt.h"
#include "qprop.h"


/**
_QPMT_rindex_interpolate
---------------------------

max_iprop::

   . (ni-1)*nj*nk + (nj-1)*nk + (nk-1) 
   =  ni*nj*nk - nj*nk + nj*nk - nk + nk - 1 
   =  ni*nj*nk - 1 


HMM: not so easy to generalize from rindex to also do qeshape
because of the different array shapes 

Each thread does all pmtcat,layers and props for a single energy_eV. 

**/

template <typename T>
__global__ void _QPMT_rindex( qpmt<T>* pmt, T* lookup , const T* domain, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    T energy_eV = domain[ix] ; 

    //printf("//_QPMT_rindex domain_width %d ix %d energy_eV %10.4f \n", domain_width, ix, energy_eV ); 

    // wierd unsigned/int diff between qpmt.h and here ? to get it to compile fo device
    // switching to enum rather than constexpr const avoids the wierdness

    const int& ni = qpmt<T>::NUM_CAT ; 
    const int& nj = qpmt<T>::NUM_LAYR ; 
    const int& nk = qpmt<T>::NUM_PROP ; 

    //printf("//_QPMT_rindex ni %d nj %d nk %d \n", ni, nj, nk ); 
    // cf the CPU equivalent NP::combined_interp_5

    for(int i=0 ; i < ni ; i++)
    for(int j=0 ; j < nj ; j++)
    for(int k=0 ; k < nk ; k++) 
    {
        int iprop = i*nj*nk+j*nk+k ;            // linearized higher dimensions 
        int index = iprop * domain_width + ix ; // output index into lookup 

        T value = pmt->rindex_prop->interpolate(iprop, energy_eV ); 

        //printf("//_QPMT_rindex iprop %d index %d value %10.4f \n", iprop, index, value );  

        lookup[index] = value ; 
    }
}

template <typename T>
__global__ void _QPMT_qeshape( qpmt<T>* pmt, T* lookup , const T* domain, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    T energy_eV = domain[ix] ; 

    //printf("//_QPMT_qeshape domain_width %d ix %d energy_eV %10.4f \n", domain_width, ix, energy_eV ); 

    const int& ni = qpmt<T>::NUM_CAT ; 

    for(int i=0 ; i < ni ; i++)
    {
        int index = i * domain_width + ix ; // output index into lookup 
        T value = pmt->qeshape_prop->interpolate(i, energy_eV ); 
        lookup[index] = value ; 
    }
}



template <typename T>
__global__ void _QPMT_stackspec( qpmt<T>* pmt, T* lookup , const T* domain, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    T energy_eV = domain[ix] ; 

    //printf("//_QPMT_stackspec domain_width %d ix %d energy_eV %10.4f \n", domain_width, ix, energy_eV ); 

    const int& ni = qpmt<T>::NUM_CAT ; 
    const int& nj = domain_width ; 
    const int  nk = 16 ; 
    const int&  j = ix ; 

    quad4 spec ; 
    const float* d = spec.cdata(); 

    for(int i=0 ; i < ni ; i++)  // over pmtcat 
    {
        int index = i*nj*nk + j*nk  ; 
        pmt->get_stackspec(spec, i, energy_eV ); 
        for( int k=0 ; k < nk ; k++) lookup[index+k] = d[k] ;  
    }
}



template <typename T> extern void QPMT_interpolate(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    qpmt<T>* pmt, 
    int etype, 
    T* lookup, 
    const T* domain, 
    unsigned domain_width
)
{
    if( etype == qpmt<T>::RINDEX )
    {
        _QPMT_rindex<T><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width ) ; 
    }
    else if( etype == qpmt<T>::QESHAPE )
    {
        _QPMT_qeshape<T><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width ) ; 
    }
    else if( etype == qpmt<T>::STACKSPEC )
    {
        _QPMT_stackspec<T><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width ) ; 
    }
} 

//template void QPMT_interpolate(dim3, dim3, qpmt<double>*, int etype, double*, const double*, unsigned ) ; 
template void QPMT_interpolate(dim3, dim3, qpmt<float>*,  int etype, float*,  const float* , unsigned ) ; 


