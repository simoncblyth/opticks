/**
QPMT.cu
==========

_QPMT_lpmtcat_rindex
_QPMT_lpmtcat_qeshape
_QPMT_lpmtcat_stackspec
    kernel funcs taking (qpmt,lookup,domain,domain_width) args 

QPMT_lpmtcat
    CPU entry point to launch above kernels controlled by etype 


_QPMT_lpmtid_stackspec
    kernel funcs taking (qpmt,lookup,domain,domain_width,lpmtid,num_lpmtid) args 

QPMT_lpmtid
    CPU entry point to launch above kernels controlled by etype 



**/

#include "QUDARAP_API_EXPORT.hh"
#include <stdio.h>
#include "qpmt.h"
#include "qprop.h"

/**
_QPMT_lpmtcat_rindex
---------------------------

max_iprop::

   . (ni-1)*nj*nk + (nj-1)*nk + (nk-1) 
   =  ni*nj*nk - nj*nk + nj*nk - nk + nk - 1 
   =  ni*nj*nk - 1 


HMM: not so easy to generalize from rindex to also do qeshape
because of the different array shapes 

Each thread does all pmtcat,layers and props for a single energy_eV. 

**/

template <typename F>
__global__ void _QPMT_lpmtcat_rindex( qpmt<F>* pmt, F* lookup , const F* domain, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    F energy_eV = domain[ix] ; 

    //printf("//_QPMT_rindex domain_width %d ix %d energy_eV %10.4f \n", domain_width, ix, energy_eV ); 

    // wierd unsigned/int diff between qpmt.h and here ? to get it to compile fo device
    // switching to enum rather than constexpr const avoids the wierdness

    const int& ni = qpmt<F>::NUM_CAT ; 
    const int& nj = qpmt<F>::NUM_LAYR ; 
    const int& nk = qpmt<F>::NUM_PROP ; 

    //printf("//_QPMT_lpmtcat_rindex ni %d nj %d nk %d \n", ni, nj, nk ); 
    // cf the CPU equivalent NP::combined_interp_5

    for(int i=0 ; i < ni ; i++)
    for(int j=0 ; j < nj ; j++)
    for(int k=0 ; k < nk ; k++) 
    {
        int iprop = i*nj*nk+j*nk+k ;            // linearized higher dimensions 
        int index = iprop * domain_width + ix ; // output index into lookup 

        F value = pmt->rindex_prop->interpolate(iprop, energy_eV ); 

        //printf("//_QPMT_lpmtcat_rindex iprop %d index %d value %10.4f \n", iprop, index, value );  

        lookup[index] = value ; 
    }
}

template <typename F>
__global__ void _QPMT_lpmtcat_qeshape( qpmt<F>* pmt, F* lookup , const F* domain, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    F energy_eV = domain[ix] ; 

    //printf("//_QPMT_lpmtcat_qeshape domain_width %d ix %d energy_eV %10.4f \n", domain_width, ix, energy_eV ); 

    const int& ni = qpmt<F>::NUM_CAT ; 

    for(int i=0 ; i < ni ; i++)
    {
        int index = i * domain_width + ix ; // output index into lookup 
        F value = pmt->qeshape_prop->interpolate(i, energy_eV ); 
        lookup[index] = value ; 
    }
}



template <typename F>
__global__ void _QPMT_lpmtcat_stackspec( qpmt<F>* pmt, F* lookup , const F* domain, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    F energy_eV = domain[ix] ; 

    //printf("//_QPMT_lpmtcat_stackspec domain_width %d ix %d energy_eV %10.4f \n", domain_width, ix, energy_eV ); 

    const int& ni = qpmt<F>::NUM_CAT ; 
    const int& nj = domain_width ; 
    const int  nk = 16 ; 
    const int&  j = ix ; 

    F ss[nk] ;
 
    for(int i=0 ; i < ni ; i++)  // over pmtcat 
    {
        int index = i*nj*nk + j*nk  ; 
        pmt->get_lpmtcat_stackspec(ss, i, energy_eV ); 
        for( int k=0 ; k < nk ; k++) lookup[index+k] = ss[k] ;  
    }
}


template <typename F> extern void QPMT_lpmtcat(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    qpmt<F>* pmt, 
    int etype, 
    F* lookup, 
    const F* domain, 
    unsigned domain_width
)
{
    if( etype == qpmt<F>::RINDEX )
    {
        _QPMT_lpmtcat_rindex<F><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width ) ; 
    }
    else if( etype == qpmt<F>::QESHAPE )
    {
        _QPMT_lpmtcat_qeshape<F><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width ) ; 
    }
    else if( etype == qpmt<F>::LPMTCAT_STACKSPEC )
    {
        _QPMT_lpmtcat_stackspec<F><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width ) ; 
    }
} 

//template void QPMT_lpmtcat(dim3, dim3, qpmt<double>*, int etype, double*, const double*, unsigned ) ; 
template void QPMT_lpmtcat(
   dim3, 
   dim3, 
   qpmt<float>*,
   int etype, 
   float*, 
   const float* , 
   unsigned 
  ); 


/**
_QPMT_lpmtid_stackspec
-------------------------

**/


template <typename F>
__global__ void _QPMT_lpmtid_stackspec( 
    qpmt<F>* pmt, 
    F* lookup , 
    const F* domain, 
    unsigned domain_width,
    const int* lpmtid,
    unsigned num_lpmtid )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    F energy_eV = domain[ix] ; 

    const int& ni = num_lpmtid ; 
    const int& nj = domain_width ; 
    const int  nk = 16 ; 
    const int&  j = ix ; 

    F ss[nk] ; 

    for(int i=0 ; i < ni ; i++)  // over num_lpmtid
    {
        int pmtid = lpmtid[i] ; 
        int index = i*nj*nk + j*nk  ; 
        pmt->get_lpmtid_stackspec(ss, pmtid, energy_eV ); 
        for( int k=0 ; k < nk ; k++) lookup[index+k] = ss[k] ;  
    }
}



template <typename F>
__global__ void _QPMT_lpmtid_ART( 
    qpmt<F>* pmt, 
    F* lookup , 
    const F* domain, 
    unsigned domain_width,
    const int* lpmtid,
    unsigned num_lpmtid )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;

    // HMM: get all these from domain ? 
    F minus_cos_theta = domain[ix] ; 
    F wavelength_nm = 440.f ; 
    F dot_pol_cross_mom_nrm = 0.f ;    


    const int& ni = num_lpmtid ; 
    const int& nj = domain_width ; 

    const int  nk = 16 ; // 4*4 stack.art payload values  
    const int&  j = ix ; 

    F art[nk] ;  

    for(int i=0 ; i < ni ; i++)  // over num_lpmtid
    {
        int pmtid = lpmtid[i] ; 
        int index = i*nj*nk + j*nk  ; 
        pmt->get_lpmtid_ART(art, pmtid, wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm ); 
        for( int k=0 ; k < nk ; k++) lookup[index+k] = art[k] ;  
    }
}



template <typename F>
__global__ void _QPMT_lpmtid_ARTE( 
    qpmt<F>* pmt, 
    F* lookup , 
    const F* domain, 
    unsigned domain_width,
    const int* lpmtid,
    unsigned num_lpmtid )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;

    F minus_cos_theta = domain[ix] ; 
    F wavelength_nm = 440.f ; 
    F dot_pol_cross_mom_nrm = 0.f ;    


    const int& ni = num_lpmtid ; 
    const int& nj = domain_width ;   // minus_cos_theta values "AOI"

    const int  nk = 4 ; // 4 ARTE payload values  
    const int&  j = ix ; 

    F arte[nk] ;  

    for(int i=0 ; i < ni ; i++)  // over num_lpmtid
    {
        int pmtid = lpmtid[i] ; 
        int index = i*nj*nk + j*nk  ; 
        pmt->get_lpmtid_ARTE(arte, pmtid, wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm ); 
        for( int k=0 ; k < nk ; k++) lookup[index+k] = arte[k] ;  
    }
}


template <typename F> extern void QPMT_lpmtid(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    qpmt<F>* pmt, 
    int etype, 
    F* lookup, 
    const F* domain, 
    unsigned domain_width,
    const int* lpmtid, 
    unsigned num_lpmtid
)
{
    if( etype == qpmt<F>::LPMTID_STACKSPEC )
    {
         _QPMT_lpmtid_stackspec<F><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width, lpmtid, num_lpmtid ) ; 
    }
    else if( etype == qpmt<F>::LPMTID_ART )
    {
         _QPMT_lpmtid_ART<F><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width, lpmtid, num_lpmtid ) ; 
    }
    else if( etype == qpmt<F>::LPMTID_ARTE )
    {
         _QPMT_lpmtid_ARTE<F><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width, lpmtid, num_lpmtid ) ; 
    }
    else
    {
         printf("//QPMT_lpmtid etype unhandled \n"); 
    }
}

template void QPMT_lpmtid(
   dim3, 
   dim3, 
   qpmt<float>*, 
   int etype, 
   float*,  
   const float* , 
   unsigned,
   const int*, 
   unsigned
 ); 

