/**
QPMT.cu
==========

_QPMT_lpmtcat_rindex
_QPMT_lpmtcat_qeshape
_QPMT_lpmtcat_stackspec
    kernel funcs taking (qpmt,lookup,domain,domain_width) args

QPMT_pmtcat_scan
    CPU entry point to launch above kernels controlled by etype


_QPMT_lpmtid_stackspec
    kernel funcs taking (qpmt,lookup,domain,domain_width,lpmtid,num_lpmtid) args

_QPMT_mct_lpmtid
    payload size P templated kernel function with domain and lpmtid array inputs

    * within lpmtid loop calls qpmt.h method depending on etype
    * etype : (qpmt_SPEC qpmt_LL qpmt_COMP qpmt_ART qpmt_ARTE)

QPMT_mct_lpmtid_scan
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
__global__ void _QPMT_lpmtcat_rindex( int etype, qpmt<F>* pmt, F* lookup , const F* domain, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    F domain_value = domain[ix] ;    // energy_eV

    //printf("//_QPMT_rindex domain_width %d ix %d domain_value %10.4f \n", domain_width, ix, domain_value );
    // wierd unsigned/int diff between qpmt.h and here ? to get it to compile for device
    // switching to enum rather than constexpr const avoids the wierdness

    const int& ni = s_pmt::NUM_CAT ;
    const int& nj = s_pmt::NUM_LAYR ;
    const int& nk = s_pmt::NUM_PROP ;

    //printf("//_QPMT_lpmtcat_rindex ni %d nj %d nk %d \n", ni, nj, nk );
    // cf the CPU equivalent NP::combined_interp_5

    for(int i=0 ; i < ni ; i++)
    for(int j=0 ; j < nj ; j++)
    for(int k=0 ; k < nk ; k++)
    {
        int iprop = i*nj*nk+j*nk+k ;            // linearized higher dimensions
        int index = iprop * domain_width + ix ; // output index into lookup

        F value = pmt->rindex_prop->interpolate(iprop, domain_value );

        //printf("//_QPMT_lpmtcat_rindex iprop %d index %d value %10.4f \n", iprop, index, value );

        lookup[index] = value ;
    }
}



template <typename F>
__global__ void _QPMT_lpmtcat_stackspec( int etype, qpmt<F>* pmt, F* lookup , const F* domain, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    F domain_value = domain[ix] ;

    //printf("//_QPMT_lpmtcat_stackspec domain_width %d ix %d domain_value %10.4f \n", domain_width, ix, domain_value );

    const int& ni = s_pmt::NUM_CAT ;
    const int& nj = domain_width ;
    const int  nk = 16 ;
    const int&  j = ix ;

    F ss[nk] ;

    for(int i=0 ; i < ni ; i++)  // over pmtcat
    {
        int index = i*nj*nk + j*nk  ;
        pmt->get_lpmtcat_stackspec(ss, i, domain_value );
        for( int k=0 ; k < nk ; k++) lookup[index+k] = ss[k] ;
    }
}



template <typename F>
__global__ void _QPMT_pmtcat_launch( int etype, qpmt<F>* pmt, F* lookup , const F* domain, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;
    F domain_value = domain[ix] ;

    //printf("//_QPMT_pmtcat_launch etype %d domain_width %d ix %d  \n", etype, domain_width, ix  );

    const int ni = ( etype == qpmt_S_QESHAPE ) ? 1 : s_pmt::NUM_CAT ;

    for(int i=0 ; i < ni ; i++)
    {
        int pmtcat = i ;
        F value = 0.f ;

        if( etype == qpmt_QESHAPE )
        {
            value = pmt->qeshape_prop->interpolate( pmtcat, domain_value );
        }
        else if( etype == qpmt_CETHETA )
        {
            //value = pmt->cetheta_prop->interpolate(lpmtcat, domain_value );
            value = pmt->get_lpmtcat_ce( pmtcat, domain_value );
        }
        else if ( etype == qpmt_CECOSTH )
        {
            value = pmt->cecosth_prop->interpolate( pmtcat, domain_value );
        }
        else if( etype == qpmt_S_QESHAPE )
        {
            value = pmt->s_qeshape_prop->interpolate( pmtcat, domain_value );
        }


        int index = i * domain_width + ix ; // output index into lookup
        lookup[index] = value ;
    }
}




/**
QPMT_pmtcat_scan
-------------------

Performs CUDA launches, invoked from QPMT.cc QPMT<T>::pmtcat_scan

**/


template <typename F> extern void QPMT_pmtcat_scan(
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
        case qpmt_RINDEX     : _QPMT_lpmtcat_rindex<F><<<numBlocks,threadsPerBlock>>>(    etype, pmt, lookup, domain, domain_width )   ; break ;
        case qpmt_CATSPEC    : _QPMT_lpmtcat_stackspec<F><<<numBlocks,threadsPerBlock>>>( etype, pmt, lookup, domain, domain_width )   ; break ;
        case qpmt_QESHAPE    : _QPMT_pmtcat_launch<F><<<numBlocks,threadsPerBlock>>>(    etype, pmt, lookup, domain, domain_width )   ; break ;
        case qpmt_CETHETA    : _QPMT_pmtcat_launch<F><<<numBlocks,threadsPerBlock>>>(    etype, pmt, lookup, domain, domain_width )   ; break ;
        case qpmt_CECOSTH    : _QPMT_pmtcat_launch<F><<<numBlocks,threadsPerBlock>>>(    etype, pmt, lookup, domain, domain_width )   ; break ;
        case qpmt_S_QESHAPE  : _QPMT_pmtcat_launch<F><<<numBlocks,threadsPerBlock>>>(    etype, pmt, lookup, domain, domain_width )   ; break ;
    }
}

template void QPMT_pmtcat_scan(
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





/**
_QPMT_mct_lpmtid
-----------------

* using templated payload size P as it needs to be a compile time constant
* parallelism over mct domain only
* loops over the provided list of pmtid


**/

#ifdef WITH_CUSTOM4
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

    //printf("//_QPMT_mct_lpmtid etype %d ix %d num_lpmtid %d P %d \n", etype, ix, num_lpmtid, P );

    F minus_cos_theta = domain[ix] ;
    F wavelength_nm = 440.f ;
    F dot_pol_cross_mom_nrm = 0.f ; // SPOL zero is pure P polarized
    F lposcost = 0.5f ;  // np.acos(0.5) 1.047197

    const int& ni = num_lpmtid ;
    const int& nj = domain_width ;   // minus_cos_theta values "AOI"
    const int&  j = ix ;

    F payload[P] ;

    for(int i=0 ; i < ni ; i++)  // over num_lpmtid
    {
        int pmtid = lpmtid[i] ;

        if( etype == qpmt_SPEC )
        {
            pmt->get_lpmtid_SPEC(payload, pmtid, wavelength_nm );
        }
        else if( etype == qpmt_SPEC_ce )
        {
            pmt->get_lpmtid_SPEC_ce(payload, pmtid, wavelength_nm, lposcost );
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
        else if( etype == qpmt_ATQC )
        {
            pmt->get_lpmtid_ATQC(payload, pmtid, wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm, lposcost );
        }


        int index = i*nj*P + j*P  ;  // output index
        for( int k=0 ; k < P ; k++) lookup[index+k] = payload[k] ;
    }
}


template <typename F> extern void QPMT_mct_lpmtid_scan(
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
    printf("//QPMT_mct_lpmtid_scan etype %d domain_width %d num_lpmtid %d \n", etype, domain_width, num_lpmtid);

    switch(etype)
    {
        case qpmt_SPEC:
           _QPMT_mct_lpmtid<F,16><<<numBlocks,threadsPerBlock>>>(
              pmt, etype, lookup, domain, domain_width, lpmtid, num_lpmtid ) ;  break ;

        case qpmt_SPEC_ce:
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

        case qpmt_ATQC:
           _QPMT_mct_lpmtid<F,4><<<numBlocks,threadsPerBlock>>>(
              pmt, etype, lookup, domain, domain_width, lpmtid, num_lpmtid ) ;  break ;

        default:
              printf("//PMT_mct_lpmtid_scan etype %d UNHANDLED \n", etype)   ; break ;

    }
}

template void QPMT_mct_lpmtid_scan<float>(   dim3, dim3, qpmt<float>*, int etype, float*,  const float* , unsigned, const int*, unsigned);
// end WITH_CUSTOM4
#endif








template <typename F>
__global__ void _QPMT_spmtid(
    qpmt<F>* pmt,
    int etype,
    F* lookup ,
    const int* spmtid,
    unsigned num_spmtid )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= num_spmtid ) return;
    int _spmtid = spmtid[ix];
    //printf("//_QPMT_spmtid etype %d ix %d num_spmtid %d _spmtid %d \n", etype, ix, num_spmtid, _spmtid );

    F value = 0.f ;
    if( etype == qpmt_S_QESCALE )
    {
        value = pmt->get_s_qescale_from_spmtid( _spmtid );
    }
    lookup[ix] = value ;
}




template <typename F> extern void QPMT_spmtid_scan(
    dim3 numBlocks,
    dim3 threadsPerBlock,
    qpmt<F>* pmt,
    int etype,
    F* lookup,
    const int* spmtid,
    unsigned num_spmtid
)
{
    printf("//QPMT_spmtid_scan etype %d num_spmtid %d \n", etype, num_spmtid);
    switch(etype)
    {
        case qpmt_S_QESCALE:
           _QPMT_spmtid<F><<<numBlocks,threadsPerBlock>>>(pmt, etype, lookup, spmtid, num_spmtid ) ;  break ;
    }
}

template void QPMT_spmtid_scan<float>( dim3, dim3, qpmt<float>*, int, float*, const int*, unsigned );


