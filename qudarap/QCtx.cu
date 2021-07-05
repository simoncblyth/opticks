
#include "stdio.h"
#include "curand_kernel.h"
#include "scuda.h"
#include "qgs.h"
#include "qctx.h"

__global__ void _QCtx_generate_wavelength(qctx* ctx, float* wavelength, unsigned num_wavelength, unsigned hd_factor )
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
    if(id % 100000 == 0) printf("//_QCtx_generate_wavelength id %d hd_factor %d wl %10.4f    \n", id, hd_factor, wl  ); 
    wavelength[id] = wl ; 
}

extern "C" void QCtx_generate_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qctx* ctx, float* wavelength, unsigned num_wavelength, unsigned hd_factor ) 
{
    printf("//QCtx_generate_wavelength num_wavelength %d \n", num_wavelength ); 
    _QCtx_generate_wavelength<<<numBlocks,threadsPerBlock>>>( ctx, wavelength, num_wavelength, hd_factor );
} 

__global__ void _QCtx_generate_photon(qctx* ctx, quad4* photon, unsigned num_photon )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;
    
    //ctx->r += id ;   
    //  could be problematic, do not want to change the the rng_states in global mem and get interference between threads

    curandState rng = *(ctx->r + id) ; 

    QG qg ;      
    qg.zero();  

    GS& g = qg.g ; 

    // fabricate some values for the genstep
    g.st.x0.x = 100.f ; 
    g.st.x0.y = 100.f ; 
    g.st.x0.z = 100.f ; 
    g.st.t0 = 20.f ; 
    g.st.DeltaPosition.x = 1000.f ; 
    g.st.DeltaPosition.y = 1000.f ; 
    g.st.DeltaPosition.z = 1000.f ; 
    g.st.step_length = 1000.f ; 
    g.sc1.midVelocity = 1.f ; 
    g.sc1.ScintillationTime = 10.f ; 

    quad4 p ;   
    ctx->scint_photon(p, g, rng); 

    photon[id] = p ; 
}

extern "C" void QCtx_generate_photon(dim3 numBlocks, dim3 threadsPerBlock, qctx* ctx, quad4* photon, unsigned num_photon ) 
{
    printf("//QCtx_generate_photon num_photon %d \n", num_photon ); 
    _QCtx_generate_photon<<<numBlocks,threadsPerBlock>>>( ctx, photon, num_photon );
} 

