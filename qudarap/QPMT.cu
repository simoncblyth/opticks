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
__global__ void _QPMT_rindex_interpolate( qpmt<T>* pmt, T* lookup , const T* domain, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    T energy_eV = domain[ix] ; 

    //printf("//_QPMT_interpolate domain_width %d ix %d energy_eV %10.4f \n", domain_width, ix, energy_eV ); 

    // wierd unsigned/int diff between qpmt.h and here ? to get it to compile fo device
    const unsigned& ni = qpmt<T>::NUM_CAT ; 
    const unsigned& nj = qpmt<T>::NUM_LAYR ; 
    const unsigned& nk = qpmt<T>::NUM_PROP ; 

    //printf("//_QPMT_interpolate ni %d nj %d nk %d \n", ni, nj, nk ); 
    // cf the CPU equivalent NP::combined_interp_5

    for(int i=0 ; i < ni ; i++)
    for(int j=0 ; j < nj ; j++)
    for(int k=0 ; k < nk ; k++) 
    {
        int iprop = i*nj*nk+j*nk+k ;            // linearized higher dimensions 
        int index = iprop * domain_width + ix ; // output index into lookup 

        T value = pmt->rindex_prop->interpolate(iprop, energy_eV ); 

        //printf("//_QPMT_interpolate iprop %d index %d value %10.4f \n", iprop, index, value );  

        lookup[index] = value ; 
    }
}

template <typename T>
__global__ void _QPMT_qeshape_interpolate( qpmt<T>* pmt, T* lookup , const T* domain, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    T energy_eV = domain[ix] ; 

    //printf("//_QPMT_interpolate domain_width %d ix %d energy_eV %10.4f \n", domain_width, ix, energy_eV ); 

    const unsigned& ni = qpmt<T>::NUM_CAT ; 

    for(int i=0 ; i < ni ; i++)
    {
        int index = i * domain_width + ix ; // output index into lookup 
        T value = pmt->qeshape_prop->interpolate(i, energy_eV ); 
        lookup[index] = value ; 
    }
}



template <typename T> extern void QPMT_rindex_interpolate(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    qpmt<T>* pmt, 
    T* lookup, 
    const T* domain, 
    unsigned domain_width
)
{
    _QPMT_rindex_interpolate<T><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width ) ; 
} 

template void QPMT_rindex_interpolate(dim3, dim3, qpmt<double>*, double*, const double*, unsigned ) ; 
template void QPMT_rindex_interpolate(dim3, dim3, qpmt<float>*,  float*,  const float* , unsigned ) ; 


template <typename T> extern void QPMT_qeshape_interpolate(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    qpmt<T>* pmt, 
    T* lookup, 
    const T* domain, 
    unsigned domain_width
)
{
    _QPMT_qeshape_interpolate<T><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width ) ; 
} 

template void QPMT_qeshape_interpolate(dim3, dim3, qpmt<double>*, double*, const double*, unsigned ) ; 
template void QPMT_qeshape_interpolate(dim3, dim3, qpmt<float>*,  float*,  const float* , unsigned ) ; 

