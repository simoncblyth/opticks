/**

REMEMBER TO KEEP CODE HERE TO A MINIMUM : PUT AS MUCH AS POSSIBLE INTO THE 
MORE EASILY TESTED FROM MULTIPLE ENVIRONMENTS HEADERS 

**/

#include "stdio.h"
#include "curand_kernel.h"
#include "scuda.h"
#include "qgs.h"
#include "qctx.h"

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

    QG qg ;      
    qg.zero();  

    GS& g = qg.g ; 

    // fabricate some values for the genstep
    g.st.Id = 0 ; 
    g.st.ParentId = 0 ; 
    g.st.MaterialIndex = 0 ; 
    g.st.NumPhotons = 0 ; 

    g.st.x0.x = 100.f ; 
    g.st.x0.y = 100.f ; 
    g.st.x0.z = 100.f ; 
    g.st.t0 = 20.f ; 

    g.st.DeltaPosition.x = 1000.f ; 
    g.st.DeltaPosition.y = 1000.f ; 
    g.st.DeltaPosition.z = 1000.f ; 
    g.st.step_length = 1000.f ; 

    g.ck1.code = 0 ; 
    g.ck1.charge = 1.f ; 
    g.ck1.weight = 1.f ; 
    g.ck1.preVelocity = 0.f ; 

    g.ck1.BetaInverse = 1.2f ; 
    g.ck1.Wmin = 300.f ; 
    g.ck1.Wmax = 600.f ; 
    g.ck1.maxCos = 0.f ; 

    g.ck1.maxSin2 = 0.f ; 
    g.ck1.MeanNumberOfPhotons1 = 0.f ; 
    g.ck1.MeanNumberOfPhotons2 = 0.f ; 
    g.ck1.postVelocity = 0.f ; 

    float wl = ctx->cerenkov_wavelength(g, rng);   

    if(id % 100000 == 0) printf("//_QCtx_generate_cerenkov_wavelength id %d wl %10.4f    \n", id, wl  ); 
    wavelength[id] = wl ; 
}


extern "C" void QCtx_generate_cerenkov_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qctx* ctx, float* wavelength, unsigned num_wavelength ) 
{
    printf("//QCtx_generate_cerenkov_wavelength num_wavelength %d \n", num_wavelength ); 
    _QCtx_generate_cerenkov_wavelength<<<numBlocks,threadsPerBlock>>>( ctx, wavelength, num_wavelength );
} 



__global__ void _QCtx_generate_photon(qctx* ctx, quad4* photon, unsigned num_photon )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;
    
    //ctx->r += id ;   
    //  could be problematic, do not want to change the the rng_states in global mem and get interference between threads

    curandState rng = *(ctx->r + id) ; 

    // TODO: this kinda stuff, all non-glue stuff,  should be in headers, not here 

    QG qg ;      
    qg.zero();  

    GS& g = qg.g ; 

    // fabricate some values for the genstep
    g.st.Id = 0 ; 
    g.st.ParentId = 0 ; 
    g.st.MaterialIndex = 0 ; 
    g.st.NumPhotons = 0 ; 

    g.st.x0.x = 100.f ; 
    g.st.x0.y = 100.f ; 
    g.st.x0.z = 100.f ; 
    g.st.t0 = 20.f ; 

    g.st.DeltaPosition.x = 1000.f ; 
    g.st.DeltaPosition.y = 1000.f ; 
    g.st.DeltaPosition.z = 1000.f ; 
    g.st.step_length = 1000.f ; 

    g.sc1.code = 1 ; 
    g.sc1.charge = 1.f ;
    g.sc1.weight = 1.f ;
    g.sc1.midVelocity = 0.f ; 

    g.sc1.scnt = 0 ;
    g.sc1.f41 = 0.f ;   
    g.sc1.f42 = 0.f ;   
    g.sc1.f43 = 0.f ;   

    g.sc1.ScintillationTime = 10.f ;
    g.sc1.f51 = 0.f ;
    g.sc1.f52 = 0.f ;
    g.sc1.f53 = 0.f ;


    quad4 p ;   
    ctx->scint_photon(p, g, rng); 

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


