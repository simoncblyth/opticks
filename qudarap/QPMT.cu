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

_QPMT_mct_lpmtid
    payload size P templated kernel function with domain and lpmtid array inputs

    * within lpmtid loop calls qpmt.h method depending on etype
    * etype : (qpmt_SPEC qpmt_LL qpmt_COMP qpmt_ART qpmt_ARTE)

QPMT_mct_lpmtid
    CPU entry point to launch above kernel passing etype


**/

#include "QUDARAP_API_EXPORT.hh"
#include <stdio.h>
#include "qpmt_enum.h"
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

    const int& ni = qpmt_NUM_CAT ;
    const int& nj = qpmt_NUM_LAYR ;
    const int& nk = qpmt_NUM_PROP ;

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

    const int& ni = qpmt_NUM_CAT ;

    for(int i=0 ; i < ni ; i++)
    {
        F value = pmt->qeshape_prop->interpolate(i, energy_eV );

        int index = i * domain_width + ix ; // output index into lookup
        lookup[index] = value ;
    }
}



template <typename F>
__global__ void _QPMT_lpmtcat_cetheta( qpmt<F>* pmt, F* lookup , const F* domain, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    F theta_radians = domain[ix] ;

    //printf("//_QPMT_lpmtcat_cetheta domain_width %d ix %d theta_radians %10.4f \n", domain_width, ix, theta_radians );

    const int& ni = qpmt_NUM_CAT ;

    for(int i=0 ; i < ni ; i++)
    {
        F value = pmt->cetheta_prop->interpolate(i, theta_radians );

        int index = i * domain_width + ix ; // output index into lookup
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

    const int& ni = qpmt_NUM_CAT ;
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
    switch(etype)
    {
        case qpmt_RINDEX   : _QPMT_lpmtcat_rindex<F><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width )    ; break ;
        case qpmt_QESHAPE  : _QPMT_lpmtcat_qeshape<F><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width )   ; break ;
        case qpmt_CETHETA  : _QPMT_lpmtcat_cetheta<F><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width )   ; break ;
        case qpmt_CATSPEC  : _QPMT_lpmtcat_stackspec<F><<<numBlocks,threadsPerBlock>>>( pmt, lookup, domain, domain_width ) ; break ;
    }
}

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


#ifdef WITH_CUSTOM4
// templated payload size P as it needs to be a compile time constant
template <typename F, int P>
__global__ void _QPMT_mct_lpmtid(
    qpmt<F>* pmt,
    int etype,
    F* lookup ,
    const F* domain,
    unsigned domain_width,
    const int* lpmtid,
    unsigned num_lpmtid )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;

    //printf("//_QPMT_mct_lpmtid ix %d num_lpmtid %d P %d \n", ix, num_lpmtid, P );

    F minus_cos_theta = domain[ix] ;
    F wavelength_nm = 440.f ;
    F dot_pol_cross_mom_nrm = 0.f ; // SPOL zero is pure P polarized

    const int& ni = num_lpmtid ;
    const int& nj = domain_width ;   // minus_cos_theta values "AOI"
    const int&  j = ix ;

    F payload[P] ;

    for(int i=0 ; i < ni ; i++)  // over num_lpmtid
    {
        int index = i*nj*P + j*P  ;
        int pmtid = lpmtid[i] ;

        if( etype == qpmt_SPEC )
        {
            pmt->get_lpmtid_SPEC(payload, pmtid, wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm );
        }
        else if( etype == qpmt_LL )
        {
            pmt->get_lpmtid_LL(payload, pmtid, wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm );
        }
        else if( etype == qpmt_COMP )
        {
            pmt->get_lpmtid_COMP(payload, pmtid, wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm );
        }
        else if( etype == qpmt_ART )
        {
            pmt->get_lpmtid_ART(payload, pmtid, wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm );
        }
        else if( etype == qpmt_ARTE )
        {
            pmt->get_lpmtid_ARTE(payload, pmtid, wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm );
        }

        for( int k=0 ; k < P ; k++) lookup[index+k] = payload[k] ;
    }
}


template <typename F> extern void QPMT_mct_lpmtid(
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
    printf("//QPMT_mct_lpmtid etype %d domain_width %d num_lpmtid %d \n", etype, domain_width, num_lpmtid);

    switch(etype)
    {
        case qpmt_SPEC:
           _QPMT_mct_lpmtid<F,16><<<numBlocks,threadsPerBlock>>>(
              pmt, etype, lookup, domain, domain_width, lpmtid, num_lpmtid ) ;  break ;

        case qpmt_ART:
           _QPMT_mct_lpmtid<F,16><<<numBlocks,threadsPerBlock>>>(
              pmt, etype, lookup, domain, domain_width, lpmtid, num_lpmtid ) ;  break ;

        case qpmt_COMP:
           _QPMT_mct_lpmtid<F,32><<<numBlocks,threadsPerBlock>>>(
              pmt, etype, lookup, domain, domain_width, lpmtid, num_lpmtid ) ;  break ;

        case qpmt_LL:
           _QPMT_mct_lpmtid<F,128><<<numBlocks,threadsPerBlock>>>(
              pmt, etype, lookup, domain, domain_width, lpmtid, num_lpmtid ) ;  break ;

        case qpmt_ARTE:
           _QPMT_mct_lpmtid<F,4><<<numBlocks,threadsPerBlock>>>(
              pmt, etype, lookup, domain, domain_width, lpmtid, num_lpmtid ) ;  break ;
    }
}

template void QPMT_mct_lpmtid<float>(   dim3, dim3, qpmt<float>*, int etype, float*,  const float* , unsigned, const int*, unsigned);
#endif

