#include "stdio.h"
#include "qrng.h"

#include "scuda.h"
#include "squad.h"
#include "srec.h"
#include "scerenkov.h"
#include "sstate.h"

#include "qprop.h"
#include "qbnd.h"
#include "qsim.h"
#include "qcerenkov_dev.h"




/**
genstep provisioning ? gensteps need to be uploaded with pointer held in qsim 
but for testing need to be able to manually fabricate a genstep
**/

__global__ void _QSim_cerenkov_dev_wavelength_rejection_sampled(qsim* sim, float* wavelength, unsigned num_wavelength )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_wavelength) return;

    RNG rng ;  sim->rng->init(rng, 0, idx); 

    float wl = qcerenkov_dev::wavelength_rejection_sampled(sim, idx, rng);   

    if(idx % 100000 == 0) printf("//_QSim_cerenkov_dev_wavelength_rejection_sampled idx %d wl %10.4f    \n", idx, wl  ); 
    wavelength[idx] = wl ; 
}


extern void QSim_cerenkov_dev_wavelength_rejection_sampled(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, float* wavelength, unsigned num_wavelength ) 
{
    printf("//QSim_cerenkov_dev_wavelength_rejection_sampled num_wavelength %d \n", num_wavelength ); 
    _QSim_cerenkov_dev_wavelength_rejection_sampled<<<numBlocks,threadsPerBlock>>>( sim, wavelength, num_wavelength );
} 

__global__ void _QSim_cerenkov_dev_generate(qsim* sim, quad4* photon, unsigned num_photon )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_photon) return;

    RNG rng ; sim->rng->init(rng, 0, idx); 

    quad4 p ;   
    qcerenkov_dev::generate(sim, p, idx, rng);   

    if(idx % 100000 == 0) printf("//_QSim_cerenkov_dev_generate idx %d \n", idx  ); 
    photon[idx] = p ; 
}

extern void QSim_cerenkov_dev_generate(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad4* photon, unsigned num_photon ) 
{
    printf("//QSim_cerenkov_dev_generate num_photon %d \n", num_photon ); 
    _QSim_cerenkov_dev_generate<<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon );
} 

template<typename T>
__global__ void _QSim_cerenkov_dev_generate_enprop(qsim* sim, quad4* photon, unsigned num_photon )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_photon) return;

    RNG rng ;  sim->rng->init(rng, 0, idx); 

    quad4 p ;   
    qcerenkov_dev::generate_enprop<T>(sim, p, idx, rng);   

    if(idx % 100000 == 0) printf("//_QSim_cerenkov_dev_generate_enprop idx %d \n", idx  ); 
    photon[idx] = p ; 
}


template<typename T>
extern void QSim_cerenkov_dev_generate_enprop(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad4* photon, unsigned num_photon) 
{
    printf("//QSim_cerenkov_dev_generate_enprop num_photon %d \n", num_photon ); 
    _QSim_cerenkov_dev_generate_enprop<T><<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon );
} 


template void QSim_cerenkov_dev_generate_enprop<float>(dim3, dim3, qsim*, quad4*, unsigned ); 
template void QSim_cerenkov_dev_generate_enprop<double>(dim3, dim3, qsim*, quad4*, unsigned ); 






__global__ void _QSim_cerenkov_dev_generate_expt_double(qsim* sim, quad4* photon, unsigned num_photon )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_photon) return;

    RNG rng ;  
    sim->rng->init(rng, 0, idx); 

    quad4 p ;   
    qcerenkov_dev::generate_expt_double(sim, p, idx, rng);   

    if(idx % 100000 == 0) printf("//_QSim_cerenkov_dev_generate_expt_double idx %d \n", idx  ); 
    photon[idx] = p ; 
}

extern void QSim_cerenkov_dev_generate_expt_double(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad4* photon, unsigned num_photon ) 
{
    printf("//QSim_cerenkov_dev_generate_expt_double num_photon %d \n", num_photon ); 
    _QSim_cerenkov_dev_generate_expt_double<<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon );
} 


