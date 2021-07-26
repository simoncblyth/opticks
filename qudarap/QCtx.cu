/**

REMEMBER TO KEEP CODE HERE TO A MINIMUM : PUT AS MUCH AS POSSIBLE INTO THE 
MORE EASILY TESTED FROM MULTIPLE ENVIRONMENTS HEADERS 

**/

#include "stdio.h"
#include "curand_kernel.h"
#include "scuda.h"
#include "qgs.h"
#include "qprop.h"
#include "qctx.h"
#include "qcurand.h"


template <typename T>
__global__ void _QCtx_rng_sequence_0(qctx<T>* ctx, T* rs, unsigned num_items )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_items) return;
    curandState rng = *(ctx->r + id) ; 
    T u = qcurand<T>::uniform(&rng) ;
    if(id % 100000 == 0) printf("//_QCtx_rng_sequence id %d u %10.4f    \n", id, u  ); 
    rs[id] = u ; 
}

template <typename T>
extern void QCtx_rng_sequence_0(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* ctx, T* rs, unsigned num_items )
{
    printf("//QCtx_rng_sequence_0 num_items %d \n", num_items ); 
    _QCtx_rng_sequence_0<T><<<numBlocks,threadsPerBlock>>>( ctx, rs, num_items );
} 


template void QCtx_rng_sequence_0(dim3, dim3, qctx<float>*, float*, unsigned); 
template void QCtx_rng_sequence_0(dim3, dim3, qctx<double>*, double*, unsigned); 




template <typename T>
__global__ void _QCtx_rng_sequence(qctx<T>* ctx, T* seq, unsigned ni, unsigned nv, unsigned ioffset )
{
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ni) return;
    curandState rng = *(ctx->r + i + ioffset) ; 
    unsigned ibase = i*nv ; 

    for(unsigned v=0 ; v < nv ; v++)
    {
        T u = qcurand<T>::uniform(&rng) ;
        seq[ibase+v] = u ;
    } 
}


template <typename T>
extern void QCtx_rng_sequence(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* ctx, T*  seq, unsigned ni, unsigned nv, unsigned ioffset )
{
    printf("//QCtx_rng_sequence_f ni %d nv %d ioffset %d  \n", ni, nv, ioffset ); 
    _QCtx_rng_sequence<T><<<numBlocks,threadsPerBlock>>>( ctx, seq, ni, nv, ioffset );

}

template void QCtx_rng_sequence(dim3, dim3, qctx<float>*, float*, unsigned, unsigned, unsigned); 
template void QCtx_rng_sequence(dim3, dim3, qctx<double>*, double*, unsigned, unsigned, unsigned); 






/**
HMM hd_factor is more appropriate as a property of the uploaded texture than it is an input argument 
TODO: rearrange hd_factor 
**/

template<typename T>
__global__ void _QCtx_scint_wavelength(qctx<T>* ctx, T* wavelength, unsigned num_wavelength, unsigned hd_factor )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_wavelength) return;

    curandState rng = *(ctx->r + id) ; 

    T wl ; 
    switch(hd_factor)
    {
        case 0:  wl = ctx->scint_wavelength_hd0(rng)  ; break ; 
        case 10: wl = ctx->scint_wavelength_hd10(rng) ; break ; 
        case 20: wl = ctx->scint_wavelength_hd20(rng) ; break ; 
        default: wl = 0.f ; 
    }
    if(id % 100000 == 0) printf("//_QCtx_scint_wavelength id %d hd_factor %d wl %10.4f    \n", id, hd_factor, wl  ); 
    wavelength[id] = wl ; 
}

template <typename T>
extern void QCtx_scint_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* ctx, T* wavelength, unsigned num_wavelength, unsigned hd_factor ) 
{
    printf("//QCtx_scint_wavelength num_wavelength %d \n", num_wavelength ); 
    _QCtx_scint_wavelength<T><<<numBlocks,threadsPerBlock>>>( ctx, wavelength, num_wavelength, hd_factor );
} 


template void QCtx_scint_wavelength(dim3, dim3, qctx<double>*, double*, unsigned, unsigned ); 
template void QCtx_scint_wavelength(dim3, dim3, qctx<float>*, float*, unsigned, unsigned ); 






/**
genstep provisioning ? gensteps need to be uploaded with pointer held in qctx 
but for testing need to be able to manually fabricate a genstep
**/

template <typename T>
__global__ void _QCtx_cerenkov_wavelength(qctx<T>* ctx, T* wavelength, unsigned num_wavelength )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_wavelength) return;

    curandState rng = *(ctx->r + id) ; 

    T wl = ctx->cerenkov_wavelength(id, rng);   

    if(id % 100000 == 0) printf("//_QCtx_cerenkov_wavelength id %d wl %10.4f    \n", id, wl  ); 
    wavelength[id] = wl ; 
}


template <typename T>
extern void QCtx_cerenkov_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* ctx, T* wavelength, unsigned num_wavelength ) 
{
    printf("//QCtx_cerenkov_wavelength num_wavelength %d \n", num_wavelength ); 
    _QCtx_cerenkov_wavelength<T><<<numBlocks,threadsPerBlock>>>( ctx, wavelength, num_wavelength );
} 

template void QCtx_cerenkov_wavelength(dim3, dim3, qctx<double>*, double*, unsigned ); 
template void QCtx_cerenkov_wavelength(dim3, dim3, qctx<float>*, float*, unsigned ); 




template <typename T>
__global__ void _QCtx_cerenkov_photon(qctx<T>* ctx, quad4* photon, unsigned num_photon, int print_id )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;

    curandState rng = *(ctx->r + id) ; 

    quad4 p ;   
    ctx->cerenkov_photon(p, id, rng, print_id);   

    if(id % 100000 == 0) printf("//_QCtx_cerenkov_photon id %d \n", id  ); 
    photon[id] = p ; 
}

template <typename T>
extern void QCtx_cerenkov_photon(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* ctx, quad4* photon, unsigned num_photon, int print_id ) 
{
    printf("//QCtx_cerenkov_photon num_photon %d \n", num_photon ); 
    _QCtx_cerenkov_photon<T><<<numBlocks,threadsPerBlock>>>( ctx, photon, num_photon, print_id );
} 

template void QCtx_cerenkov_photon(dim3, dim3, qctx<double>*, quad4*, unsigned, int ); 
template void QCtx_cerenkov_photon(dim3, dim3, qctx<float>*, quad4*, unsigned, int ); 








template <typename T>
__global__ void _QCtx_cerenkov_photon_enprop(qctx<T>* ctx, quad4* photon, unsigned num_photon, int print_id )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;

    curandState rng = *(ctx->r + id) ; 

    quad4 p ;   
    ctx->cerenkov_photon_enprop(p, id, rng, print_id);   

    if(id % 100000 == 0) printf("//_QCtx_cerenkov_photon_enprop id %d \n", id  ); 
    photon[id] = p ; 
}

template <typename T>
extern void QCtx_cerenkov_photon_enprop(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* ctx, quad4* photon, unsigned num_photon, int print_id ) 
{
    printf("//QCtx_cerenkov_photon_enprop num_photon %d \n", num_photon ); 
    _QCtx_cerenkov_photon_enprop<T><<<numBlocks,threadsPerBlock>>>( ctx, photon, num_photon, print_id );
} 

template void QCtx_cerenkov_photon_enprop(dim3, dim3, qctx<double>*, quad4*, unsigned, int ); 
template void QCtx_cerenkov_photon_enprop(dim3, dim3, qctx<float>*, quad4*, unsigned, int ); 








template <typename T>
__global__ void _QCtx_cerenkov_photon_expt(qctx<T>* ctx, quad4* photon, unsigned num_photon, int print_id )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;

    curandState rng = *(ctx->r + id) ; 

    quad4 p ;   
    ctx->cerenkov_photon_expt(p, id, rng, print_id);   

    if(id % 100000 == 0) printf("//_QCtx_cerenkov_photon_expt id %d \n", id  ); 
    photon[id] = p ; 
}

template <typename T>
extern void QCtx_cerenkov_photon_expt(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* ctx, quad4* photon, unsigned num_photon, int print_id ) 
{
    printf("//QCtx_cerenkov_photon_expt num_photon %d \n", num_photon ); 
    _QCtx_cerenkov_photon_expt<T><<<numBlocks,threadsPerBlock>>>( ctx, photon, num_photon, print_id );
} 

template void QCtx_cerenkov_photon_expt(dim3, dim3, qctx<double>*, quad4*, unsigned, int ); 
template void QCtx_cerenkov_photon_expt(dim3, dim3, qctx<float>*, quad4*, unsigned, int ); 














template <typename T>
__global__ void _QCtx_scint_photon(qctx<T>* ctx, quad4* photon, unsigned num_photon )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;
    
    //ctx->r += id ;   
    //  would be problematic, do not want to change the the rng_states in global mem and get interference between threads

    curandState rng = *(ctx->r + id) ; 

    quad4 p ;   
    ctx->scint_photon(p, rng); 

    photon[id] = p ; 
}

template <typename T>
extern void QCtx_scint_photon(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* ctx, quad4* photon, unsigned num_photon ) 
{
    printf("//QCtx_scint_photon num_photon %d \n", num_photon ); 
    _QCtx_scint_photon<T><<<numBlocks,threadsPerBlock>>>( ctx, photon, num_photon );
} 

template void QCtx_scint_photon(dim3, dim3, qctx<double>*, quad4*, unsigned ); 
template void QCtx_scint_photon(dim3, dim3, qctx<float>*, quad4*, unsigned ); 



template <typename T>
__global__ void _QCtx_boundary_lookup_all(qctx<T>* ctx, quad* lookup, unsigned width, unsigned height )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned index = iy * width + ix ;
    if (ix >= width | iy >= height ) return;

    quad q ; 
    q.f = ctx->boundary_lookup( ix, iy ); 
    lookup[index] = q ; 
}

template <typename T>
extern void QCtx_boundary_lookup_all(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* ctx, quad* lookup, unsigned width, unsigned height )
{
    printf("//QCtx_boundary_lookup width %d  height %d \n", width, height ); 
    _QCtx_boundary_lookup_all<T><<<numBlocks,threadsPerBlock>>>( ctx, lookup, width, height );
}

template void QCtx_boundary_lookup_all(dim3, dim3, qctx<double>*, quad*, unsigned, unsigned ); 
template void QCtx_boundary_lookup_all(dim3, dim3, qctx<float>*, quad*, unsigned, unsigned ); 


template <typename T>
__global__ void _QCtx_boundary_lookup_line(qctx<T>* ctx, quad* lookup, T* domain, unsigned num_lookup, unsigned line, unsigned k )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_lookup) return;
    T wavelength = domain[id] ;  
    quad q ; 
    q.f = ctx->boundary_lookup( wavelength, line, k ); 
    lookup[id] = q ; 
}


template <typename T>
extern void QCtx_boundary_lookup_line(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* ctx, quad* lookup, T* domain, unsigned num_lookup, unsigned line, unsigned k )
{
    printf("//QCtx_boundary_lookup_line num_lookup %d line %d k %d  \n", num_lookup, line, k ); 
    _QCtx_boundary_lookup_line<T><<<numBlocks,threadsPerBlock>>>( ctx, lookup, domain, num_lookup, line, k );
}

template void QCtx_boundary_lookup_line(dim3, dim3, qctx<double>*, quad*, double*, unsigned, unsigned, unsigned ); 
template void QCtx_boundary_lookup_line(dim3, dim3, qctx<float>*, quad*, float*, unsigned, unsigned, unsigned ); 


template <typename T>
__global__ void _QCtx_prop_lookup(qctx<T>* ctx, T* lookup, const T* domain, unsigned domain_width, unsigned* pids, unsigned num_pids )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= domain_width || iy >= num_pids  ) return;

    T x = domain[ix] ;  
    unsigned pid = pids[iy] ; 

    T y = ctx->prop->interpolate( pid, x ); 
    lookup[iy*domain_width + ix] = y ; 
}

template <typename T>
extern void QCtx_prop_lookup( dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* ctx, T* lookup, const T* domain, unsigned domain_width, unsigned* pids, unsigned num_pids )
{
    printf("//QCtx_prop_lookup domain_width %d num_pids %d  \n", domain_width, num_pids ); 
    _QCtx_prop_lookup<T><<<numBlocks,threadsPerBlock>>>( ctx, lookup, domain, domain_width, pids, num_pids );
}


template void QCtx_prop_lookup(dim3, dim3, qctx<double>*, double*, double const*, unsigned, unsigned*, unsigned) ; 
template void QCtx_prop_lookup(dim3, dim3, qctx<float>*,  float*,  float const*, unsigned, unsigned*, unsigned ) ; 







/**
ipid : index of the lookup outputs for that pid, which may differ from index of the pid   
**/

template <typename T>
__global__ void _QCtx_prop_lookup_one(qctx<T>* ctx, T* lookup, const T* domain, unsigned domain_width, unsigned num_pids, unsigned pid, unsigned ipid )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width || pid >= num_pids  ) return;

    T x = domain[ix] ;  
    T y = ctx->prop->interpolate( pid, x ); 

    lookup[ipid*domain_width + ix] = y ; 
}

template <typename T>
extern  void QCtx_prop_lookup_one(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* ctx, T* lookup, const T* domain, unsigned domain_width, unsigned num_pids, unsigned pid, unsigned ipid )
{
    printf("//QCtx_prop_lookup_one domain_width %d num_pids %d pid %d ipid %d \n", domain_width, num_pids, pid, ipid ); 
    _QCtx_prop_lookup_one<T><<<numBlocks,threadsPerBlock>>>( ctx, lookup, domain, domain_width, num_pids, pid, ipid );
}

template void QCtx_prop_lookup_one(dim3, dim3, qctx<double>*, double*, const double*, unsigned, unsigned, unsigned, unsigned ) ; 
template void QCtx_prop_lookup_one(dim3, dim3, qctx<float>*, float*, const float*, unsigned, unsigned, unsigned, unsigned ) ; 


