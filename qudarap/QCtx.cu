

#include "curand_kernel.h"
#include "scuda.h"
#include "qgs.h"
#include "qctx.h"


__global__ void _QCtx_generate_wavelength(curandState* rng_states, cudaTextureObject_t texObj, float* wavelength, unsigned num_wavelength )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_wavelength) return;

    qctx ctx ; 
    ctx.r         = rng_states+id ; 
    ctx.scint_tex = texObj ;  
    wavelength[id] = ctx.scint_wavelength(); 
}

extern "C" void QCtx_generate_wavelength(dim3 numBlocks, dim3 threadsPerBlock, curandState* rng_states, cudaTextureObject_t texObj, float* wavelength, unsigned num_wavelength ) 
{
    _QCtx_generate_wavelength<<<numBlocks,threadsPerBlock>>>( rng_states, texObj, wavelength, num_wavelength );
} 


__global__ void _QCtx_generate_photon(curandState* rng_states, cudaTextureObject_t texObj, quad4* photon, unsigned num_photon )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;

    qctx ctx ; 
    ctx.r         = rng_states+id ; 
    ctx.scint_tex = texObj ;  
 
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
    ctx.scint_photon(p, g); 

    photon[id] = p ; 
}

extern "C" void QCtx_generate_photon(dim3 numBlocks, dim3 threadsPerBlock, curandState* rng_states, cudaTextureObject_t texObj, quad4* photon, unsigned num_photon ) 
{
    _QCtx_generate_photon<<<numBlocks,threadsPerBlock>>>( rng_states, texObj, photon, num_photon );
} 




