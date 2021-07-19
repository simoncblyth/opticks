/**

REMEMBER TO KEEP CODE HERE TO A MINIMUM : PUT AS MUCH AS POSSIBLE INTO THE 
MORE EASILY TESTED FROM MULTIPLE ENVIRONMENTS HEADERS 

**/

#include "stdio.h"
#include "curand_kernel.h"
#include "scuda.h"
#include "qgs.h"
#include "qctx.h"


__global__ void _QCtx_rng_sequence_0(qctx* ctx, float* rs, unsigned num_items )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_items) return;
    curandState rng = *(ctx->r + id) ; 
    float u = curand_uniform(&rng) ;
    if(id % 100000 == 0) printf("//_QCtx_rng_sequence id %d u %10.4f    \n", id, u  ); 
    rs[id] = u ; 
}

extern "C" void QCtx_rng_sequence_0(dim3 numBlocks, dim3 threadsPerBlock, qctx* ctx, float* rs, unsigned num_items )
{
    printf("//QCtx_rng_sequence_0 num_items %d \n", num_items ); 
    _QCtx_rng_sequence_0<<<numBlocks,threadsPerBlock>>>( ctx, rs, num_items );
} 






__global__ void _QCtx_rng_sequence_f(qctx* ctx, float* seq, unsigned ni, unsigned nv, unsigned ioffset )
{
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ni) return;
    curandState rng = *(ctx->r + i + ioffset) ; 
    unsigned ibase = i*nv ; 

    for(unsigned v=0 ; v < nv ; v++)
    {
        float u = curand_uniform(&rng) ;
        seq[ibase+v] = u ;
    } 
}
extern "C" void QCtx_rng_sequence_f(dim3 numBlocks, dim3 threadsPerBlock, qctx* ctx, float*  seq, unsigned ni, unsigned nv, unsigned ioffset )
{
    printf("//QCtx_rng_sequence_f ni %d nv %d ioffset %d  \n", ni, nv, ioffset ); 
    _QCtx_rng_sequence_f<<<numBlocks,threadsPerBlock>>>( ctx, seq, ni, nv, ioffset );

}

__global__ void _QCtx_rng_sequence_d(qctx* ctx, double* seq, unsigned ni, unsigned nv, unsigned ioffset )
{
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ni) return;
    curandState rng = *(ctx->r + i + ioffset) ; 
    unsigned ibase = i*nv ; 

    for(unsigned v=0 ; v < nv ; v++)
    {
        double u = curand_uniform_double(&rng) ;
        seq[ibase+v] = u ;
    } 
}
extern "C" void QCtx_rng_sequence_d(dim3 numBlocks, dim3 threadsPerBlock, qctx* ctx, double* seq, unsigned ni, unsigned nv, unsigned ioffset )
{
    printf("//QCtx_rng_sequence_d ni %d nv %d ioffset %d  \n", ni, nv, ioffset ); 
    _QCtx_rng_sequence_d<<<numBlocks,threadsPerBlock>>>( ctx, seq, ni, nv, ioffset );
}





/**
HMM hd_factor is more appropriate as a property of the uploaded texture than it is an input argument 
TODO: rearrange hd_factor 
**/

__global__ void _QCtx_generate_scint_wavelength(qctx* ctx, float* wavelength, unsigned num_wavelength, unsigned hd_factor )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_wavelength) return;

    curandState rng = *(ctx->r + id) ; 

    float wl ; 
    switch(hd_factor)
    {
        case 0:  wl = ctx->scint_wavelength_hd0(rng)  ; break ; 
        case 10: wl = ctx->scint_wavelength_hd10(rng) ; break ; 
        case 20: wl = ctx->scint_wavelength_hd20(rng) ; break ; 
        default: wl = 0.f ; 
    }
    if(id % 100000 == 0) printf("//_QCtx_generate_scint_wavelength id %d hd_factor %d wl %10.4f    \n", id, hd_factor, wl  ); 
    wavelength[id] = wl ; 
}

extern "C" void QCtx_generate_scint_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qctx* ctx, float* wavelength, unsigned num_wavelength, unsigned hd_factor ) 
{
    printf("//QCtx_generate_scint_wavelength num_wavelength %d \n", num_wavelength ); 
    _QCtx_generate_scint_wavelength<<<numBlocks,threadsPerBlock>>>( ctx, wavelength, num_wavelength, hd_factor );
} 

/**
genstep provisioning ? gensteps need to be uploaded with pointer held in qctx 
but for testing need to be able to manually fabricate a genstep
**/

__global__ void _QCtx_generate_cerenkov_wavelength(qctx* ctx, float* wavelength, unsigned num_wavelength )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_wavelength) return;

    curandState rng = *(ctx->r + id) ; 

    float wl = ctx->cerenkov_wavelength(id, rng);   

    if(id % 100000 == 0) printf("//_QCtx_generate_cerenkov_wavelength id %d wl %10.4f    \n", id, wl  ); 
    wavelength[id] = wl ; 
}


extern "C" void QCtx_generate_cerenkov_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qctx* ctx, float* wavelength, unsigned num_wavelength ) 
{
    printf("//QCtx_generate_cerenkov_wavelength num_wavelength %d \n", num_wavelength ); 
    _QCtx_generate_cerenkov_wavelength<<<numBlocks,threadsPerBlock>>>( ctx, wavelength, num_wavelength );
} 





__global__ void _QCtx_generate_cerenkov_photon(qctx* ctx, quad4* photon, unsigned num_photon, int print_id )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;

    curandState rng = *(ctx->r + id) ; 

    quad4 p ;   
    ctx->cerenkov_photon(p, id, rng, print_id);   

    if(id % 100000 == 0) printf("//_QCtx_generate_cerenkov_photon id %d \n", id  ); 
    photon[id] = p ; 
}

extern "C" void QCtx_generate_cerenkov_photon(dim3 numBlocks, dim3 threadsPerBlock, qctx* ctx, quad4* photon, unsigned num_photon, int print_id ) 
{
    printf("//QCtx_generate_cerenkov_photon num_photon %d \n", num_photon ); 
    _QCtx_generate_cerenkov_photon<<<numBlocks,threadsPerBlock>>>( ctx, photon, num_photon, print_id );
} 








__global__ void _QCtx_generate_photon(qctx* ctx, quad4* photon, unsigned num_photon )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;
    
    //ctx->r += id ;   
    //  could be problematic, do not want to change the the rng_states in global mem and get interference between threads

    curandState rng = *(ctx->r + id) ; 

    quad4 p ;   
    ctx->scint_photon(p, rng); 

    photon[id] = p ; 
}

extern "C" void QCtx_generate_photon(dim3 numBlocks, dim3 threadsPerBlock, qctx* ctx, quad4* photon, unsigned num_photon ) 
{
    printf("//QCtx_generate_photon num_photon %d \n", num_photon ); 
    _QCtx_generate_photon<<<numBlocks,threadsPerBlock>>>( ctx, photon, num_photon );
} 




__global__ void _QCtx_boundary_lookup_all(qctx* ctx, quad* lookup, unsigned width, unsigned height )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned index = iy * width + ix ;
    if (ix >= width | iy >= height ) return;

    quad q ; 
    q.f = ctx->boundary_lookup( ix, iy ); 
    lookup[index] = q ; 
}

extern "C" void QCtx_boundary_lookup_all(dim3 numBlocks, dim3 threadsPerBlock, qctx* ctx, quad* lookup, unsigned width, unsigned height )
{
    printf("//QCtx_boundary_lookup width %d  height %d \n", width, height ); 
    _QCtx_boundary_lookup_all<<<numBlocks,threadsPerBlock>>>( ctx, lookup, width, height );
}



__global__ void _QCtx_boundary_lookup_line(qctx* ctx, quad* lookup, float* domain, unsigned num_lookup, unsigned line, unsigned k )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_lookup) return;
    float wavelength = domain[id] ;  
    quad q ; 
    q.f = ctx->boundary_lookup( wavelength, line, k ); 
    lookup[id] = q ; 
}


extern "C" void QCtx_boundary_lookup_line(dim3 numBlocks, dim3 threadsPerBlock, qctx* ctx, quad* lookup, float* domain, unsigned num_lookup, unsigned line, unsigned k )
{
    printf("//QCtx_boundary_lookup_line num_lookup %d line %d k %d  \n", num_lookup, line, k ); 
    _QCtx_boundary_lookup_line<<<numBlocks,threadsPerBlock>>>( ctx, lookup, domain, num_lookup, line, k );
}


