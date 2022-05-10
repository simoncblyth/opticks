
/**
genstep provisioning ? gensteps need to be uploaded with pointer held in qsim 
but for testing need to be able to manually fabricate a genstep
**/

__global__ void _QSim_cerenkov_wavelength_rejection_sampled(qsim* sim, float* wavelength, unsigned num_wavelength )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_wavelength) return;

    curandState rng = sim->rngstate[id]; 

    float wl = qcerenkov::cerenkov_wavelength_rejection_sampled(sim, id, rng);   

    if(id % 100000 == 0) printf("//_QSim_cerenkov_wavelength_rejection_sampled id %d wl %10.4f    \n", id, wl  ); 
    wavelength[id] = wl ; 
}


extern void QSim_cerenkov_wavelength_rejection_sampled(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, float* wavelength, unsigned num_wavelength ) 
{
    printf("//QSim_cerenkov_wavelength_rejection_sampled num_wavelength %d \n", num_wavelength ); 
    _QSim_cerenkov_wavelength_rejection_sampled<<<numBlocks,threadsPerBlock>>>( sim, wavelength, num_wavelength );
} 

__global__ void _QSim_cerenkov_generate(qsim* sim, quad4* photon, unsigned num_photon )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_photon) return;

    curandState rng = sim->rngstate[idx]; 

    quad4 p ;   
    qcerenkov::cerenkov_generate(sim, p, idx, rng);   

    if(idx % 100000 == 0) printf("//_QSim_cerenkov_generate idx %d \n", idx  ); 
    photon[idx] = p ; 
}

extern void QSim_cerenkov_generate(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad4* photon, unsigned num_photon ) 
{
    printf("//QSim_cerenkov_generate num_photon %d \n", num_photon ); 
    _QSim_cerenkov_generate<<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon );
} 

template<typename T>
__global__ void _QSim_cerenkov_generate_enprop(qsim* sim, quad4* photon, unsigned num_photon )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_photon) return;

    curandState rng = sim->rngstate[idx]; 

    quad4 p ;   
    qcerenkov::cerenkov_generate_enprop<T>(sim, p, idx, rng);   

    if(idx % 100000 == 0) printf("//_QSim_cerenkov_generate_enprop idx %d \n", idx  ); 
    photon[idx] = p ; 
}


template<typename T>
extern void QSim_cerenkov_generate_enprop(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad4* photon, unsigned num_photon) 
{
    printf("//QSim_cerenkov_generate_enprop num_photon %d \n", num_photon ); 
    _QSim_cerenkov_generate_enprop<T><<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon );
} 


template void QSim_cerenkov_generate_enprop<float>(dim3, dim3, qsim*, quad4*, unsigned ); 
template void QSim_cerenkov_generate_enprop<double>(dim3, dim3, qsim*, quad4*, unsigned ); 

__global__ void _QSim_cerenkov_generate_expt(qsim* sim, quad4* photon, unsigned num_photon )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_photon) return;

    curandState rng = sim->rngstate[idx]; 

    quad4 p ;   
    qcerenkov::cerenkov_generate_expt(sim, p, idx, rng);   

    if(idx % 100000 == 0) printf("//_QSim_cerenkov_generate_expt idx %d \n", idx  ); 
    photon[idx] = p ; 
}

extern void QSim_cerenkov_generate_expt(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad4* photon, unsigned num_photon ) 
{
    printf("//QSim_cerenkov_generate_expt num_photon %d \n", num_photon ); 
    _QSim_cerenkov_generate_expt<<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon );
} 




