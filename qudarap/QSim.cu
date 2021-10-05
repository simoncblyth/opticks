/**

REMEMBER TO KEEP CODE HERE TO A MINIMUM : PUT AS MUCH AS POSSIBLE INTO THE 
MORE EASILY TESTED FROM MULTIPLE ENVIRONMENTS HEADERS 

**/

#include "stdio.h"
#include "curand_kernel.h"
#include "scuda.h"
#include "qgs.h"
#include "qprop.h"
#include "qsim.h"
#include "qcurand.h"
#include "qevent.h"


/**
_QSim_rng_sequence
--------------------

id_offset : applies to sim.rngstate array controlling which curandState to use

**/

template <typename T>
__global__ void _QSim_rng_sequence(qsim<T>* sim, T* seq, unsigned ni, unsigned nv, unsigned id_offset )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= ni) return;
    curandState rng = sim->rngstate[id+id_offset]; 
    unsigned ibase = id*nv ; 

    for(unsigned v=0 ; v < nv ; v++)
    {
        T u = qcurand<T>::uniform(&rng) ;
        seq[ibase+v] = u ;
    } 
}


template <typename T>
extern void QSim_rng_sequence(dim3 numBlocks, dim3 threadsPerBlock, qsim<T>* sim, T*  seq, unsigned ni, unsigned nv, unsigned id_offset )
{
    printf("//QSim_rng_sequence_f ni %d nv %d id_offset %d  \n", ni, nv, id_offset ); 
    _QSim_rng_sequence<T><<<numBlocks,threadsPerBlock>>>( sim, seq, ni, nv, id_offset );

}

template void QSim_rng_sequence(dim3, dim3, qsim<float>*, float*, unsigned, unsigned, unsigned); 
template void QSim_rng_sequence(dim3, dim3, qsim<double>*, double*, unsigned, unsigned, unsigned); 






/**
HMM hd_factor is more appropriate as a property of the uploaded texture than it is an input argument 
TODO: rearrange hd_factor 
**/

template<typename T>
__global__ void _QSim_scint_wavelength(qsim<T>* sim, T* wavelength, unsigned num_wavelength, unsigned hd_factor )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_wavelength) return;

    curandState rng = sim->rngstate[id]; 

    T wl ; 
    switch(hd_factor)
    {
        case 0:  wl = sim->scint_wavelength_hd0(rng)  ; break ; 
        case 10: wl = sim->scint_wavelength_hd10(rng) ; break ; 
        case 20: wl = sim->scint_wavelength_hd20(rng) ; break ; 
        default: wl = 0.f ; 
    }
    if(id % 100000 == 0) printf("//_QSim_scint_wavelength id %d hd_factor %d wl %10.4f    \n", id, hd_factor, wl  ); 
    wavelength[id] = wl ; 
}

template <typename T>
extern void QSim_scint_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qsim<T>* sim, T* wavelength, unsigned num_wavelength, unsigned hd_factor ) 
{
    printf("//QSim_scint_wavelength num_wavelength %d \n", num_wavelength ); 
    _QSim_scint_wavelength<T><<<numBlocks,threadsPerBlock>>>( sim, wavelength, num_wavelength, hd_factor );
} 


template void QSim_scint_wavelength(dim3, dim3, qsim<double>*, double*, unsigned, unsigned ); 
template void QSim_scint_wavelength(dim3, dim3, qsim<float>*, float*, unsigned, unsigned ); 






/**
genstep provisioning ? gensteps need to be uploaded with pointer held in qsim 
but for testing need to be able to manually fabricate a genstep
**/

template <typename T>
__global__ void _QSim_cerenkov_wavelength_rejection_sampled(qsim<T>* sim, T* wavelength, unsigned num_wavelength )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_wavelength) return;

    curandState rng = sim->rngstate[id]; 

    T wl = sim->cerenkov_wavelength_rejection_sampled(id, rng);   

    if(id % 100000 == 0) printf("//_QSim_cerenkov_wavelength_rejection_sampled id %d wl %10.4f    \n", id, wl  ); 
    wavelength[id] = wl ; 
}


template <typename T>
extern void QSim_cerenkov_wavelength_rejection_sampled(dim3 numBlocks, dim3 threadsPerBlock, qsim<T>* sim, T* wavelength, unsigned num_wavelength ) 
{
    printf("//QSim_cerenkov_wavelength_rejection_sampled num_wavelength %d \n", num_wavelength ); 
    _QSim_cerenkov_wavelength_rejection_sampled<T><<<numBlocks,threadsPerBlock>>>( sim, wavelength, num_wavelength );
} 

template void QSim_cerenkov_wavelength_rejection_sampled(dim3, dim3, qsim<double>*, double*, unsigned ); 
template void QSim_cerenkov_wavelength_rejection_sampled(dim3, dim3, qsim<float>*, float*, unsigned ); 




template <typename T>
__global__ void _QSim_cerenkov_photon(qsim<T>* sim, quad4* photon, unsigned num_photon, int print_id )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;

    curandState rng = sim->rngstate[id]; 

    quad4 p ;   
    sim->cerenkov_photon(p, id, rng, print_id);   

    if(id % 100000 == 0) printf("//_QSim_cerenkov_photon id %d \n", id  ); 
    photon[id] = p ; 
}

template <typename T>
extern void QSim_cerenkov_photon(dim3 numBlocks, dim3 threadsPerBlock, qsim<T>* sim, quad4* photon, unsigned num_photon, int print_id ) 
{
    printf("//QSim_cerenkov_photon num_photon %d \n", num_photon ); 
    _QSim_cerenkov_photon<T><<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon, print_id );
} 

template void QSim_cerenkov_photon(dim3, dim3, qsim<double>*, quad4*, unsigned, int ); 
template void QSim_cerenkov_photon(dim3, dim3, qsim<float>*, quad4*, unsigned, int ); 








template <typename T>
__global__ void _QSim_cerenkov_photon_enprop(qsim<T>* sim, quad4* photon, unsigned num_photon, int print_id )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;

    curandState rng = sim->rngstate[id]; 

    quad4 p ;   
    sim->cerenkov_photon_enprop(p, id, rng, print_id);   

    if(id % 100000 == 0) printf("//_QSim_cerenkov_photon_enprop id %d \n", id  ); 
    photon[id] = p ; 
}

template <typename T>
extern void QSim_cerenkov_photon_enprop(dim3 numBlocks, dim3 threadsPerBlock, qsim<T>* sim, quad4* photon, unsigned num_photon, int print_id ) 
{
    printf("//QSim_cerenkov_photon_enprop num_photon %d \n", num_photon ); 
    _QSim_cerenkov_photon_enprop<T><<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon, print_id );
} 

template void QSim_cerenkov_photon_enprop(dim3, dim3, qsim<double>*, quad4*, unsigned, int ); 
template void QSim_cerenkov_photon_enprop(dim3, dim3, qsim<float>*, quad4*, unsigned, int ); 








template <typename T>
__global__ void _QSim_cerenkov_photon_expt(qsim<T>* sim, quad4* photon, unsigned num_photon, int print_id )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;

    curandState rng = sim->rngstate[id]; 

    quad4 p ;   
    sim->cerenkov_photon_expt(p, id, rng, print_id);   

    if(id % 100000 == 0) printf("//_QSim_cerenkov_photon_expt id %d \n", id  ); 
    photon[id] = p ; 
}

template <typename T>
extern void QSim_cerenkov_photon_expt(dim3 numBlocks, dim3 threadsPerBlock, qsim<T>* sim, quad4* photon, unsigned num_photon, int print_id ) 
{
    printf("//QSim_cerenkov_photon_expt num_photon %d \n", num_photon ); 
    _QSim_cerenkov_photon_expt<T><<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon, print_id );
} 

template void QSim_cerenkov_photon_expt(dim3, dim3, qsim<double>*, quad4*, unsigned, int ); 
template void QSim_cerenkov_photon_expt(dim3, dim3, qsim<float>*, quad4*, unsigned, int ); 













template <typename T>
__global__ void _QSim_scint_photon(qsim<T>* sim, quad4* photon, unsigned num_photon )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;
    
    //sim->r += id ;   
    //  would be problematic, do not want to change the the rng_states in global mem and get interference between threads

    curandState rng = sim->rngstate[id] ; 

    quad4 p ;   
    sim->scint_photon(p, rng); 

    photon[id] = p ; 
}

template <typename T>
extern void QSim_scint_photon(dim3 numBlocks, dim3 threadsPerBlock, qsim<T>* sim, quad4* photon, unsigned num_photon ) 
{
    printf("//QSim_scint_photon num_photon %d \n", num_photon ); 
    _QSim_scint_photon<T><<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon );
} 

template void QSim_scint_photon(dim3, dim3, qsim<double>*, quad4*, unsigned ); 
template void QSim_scint_photon(dim3, dim3, qsim<float>*, quad4*, unsigned ); 



template <typename T>
__global__ void _QSim_generate_photon(qsim<T>* sim, qevent* evt )
{
    unsigned photon_id = blockIdx.x*blockDim.x + threadIdx.x;
    
   if (photon_id >= evt->num_photon) return;
    
    curandState rng = sim->rngstate[photon_id] ; 
    unsigned genstep_id = evt->seed[photon_id] ; 
    const quad6& gs     = evt->genstep[genstep_id] ; 

    //printf("//_QSim_generate_photon photon_id %d evt->num_photon %d genstep_id %d  \n", photon_id, evt->num_photon, genstep_id );  

    quad4 p ;   
    sim->generate_photon(p, rng, gs, photon_id, genstep_id ); 

    evt->photon[photon_id] = p ; 

}

template <typename T>
extern void QSim_generate_photon(dim3 numBlocks, dim3 threadsPerBlock, qsim<T>* sim, qevent* evt ) 
{
    printf("//QSim_generate_photon sim %p evt %p \n", sim, evt ); 
    // NB trying to use the the sim and evt pointers here gives "Bus error" 
    // thats because this is not yet on GPU, despite being compiled by nvcc
    _QSim_generate_photon<T><<<numBlocks,threadsPerBlock>>>( sim, evt );
} 

template void QSim_generate_photon(dim3, dim3, qsim<double>*, qevent* ); 
template void QSim_generate_photon(dim3, dim3, qsim<float>*,  qevent* ); 












template <typename T>
__global__ void _QSim_boundary_lookup_all(qsim<T>* sim, quad* lookup, unsigned width, unsigned height )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned index = iy * width + ix ;
    if (ix >= width | iy >= height ) return;

    quad q ; 
    q.f = sim->boundary_lookup( ix, iy ); 
    lookup[index] = q ; 
}

template <typename T>
extern void QSim_boundary_lookup_all(dim3 numBlocks, dim3 threadsPerBlock, qsim<T>* sim, quad* lookup, unsigned width, unsigned height )
{
    printf("//QSim_boundary_lookup width %d  height %d \n", width, height ); 
    _QSim_boundary_lookup_all<T><<<numBlocks,threadsPerBlock>>>( sim, lookup, width, height );
}

template void QSim_boundary_lookup_all(dim3, dim3, qsim<double>*, quad*, unsigned, unsigned ); 
template void QSim_boundary_lookup_all(dim3, dim3, qsim<float>*, quad*, unsigned, unsigned ); 


template <typename T>
__global__ void _QSim_boundary_lookup_line(qsim<T>* sim, quad* lookup, T* domain, unsigned num_lookup, unsigned line, unsigned k )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_lookup) return;
    T wavelength = domain[id] ;  
    quad q ; 
    q.f = sim->boundary_lookup( wavelength, line, k ); 
    lookup[id] = q ; 
}


template <typename T>
extern void QSim_boundary_lookup_line(dim3 numBlocks, dim3 threadsPerBlock, qsim<T>* sim, quad* lookup, T* domain, unsigned num_lookup, unsigned line, unsigned k )
{
    printf("//QSim_boundary_lookup_line num_lookup %d line %d k %d  \n", num_lookup, line, k ); 
    _QSim_boundary_lookup_line<T><<<numBlocks,threadsPerBlock>>>( sim, lookup, domain, num_lookup, line, k );
}

template void QSim_boundary_lookup_line(dim3, dim3, qsim<double>*, quad*, double*, unsigned, unsigned, unsigned ); 
template void QSim_boundary_lookup_line(dim3, dim3, qsim<float>*, quad*, float*, unsigned, unsigned, unsigned ); 


template <typename T>
__global__ void _QSim_prop_lookup(qsim<T>* sim, T* lookup, const T* domain, unsigned domain_width, unsigned* pids, unsigned num_pids )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= domain_width || iy >= num_pids  ) return;

    T x = domain[ix] ;  
    unsigned pid = pids[iy] ; 

    T y = sim->prop->interpolate( pid, x ); 
    lookup[iy*domain_width + ix] = y ; 
}

template <typename T>
extern void QSim_prop_lookup( dim3 numBlocks, dim3 threadsPerBlock, qsim<T>* sim, T* lookup, const T* domain, unsigned domain_width, unsigned* pids, unsigned num_pids )
{
    printf("//QSim_prop_lookup domain_width %d num_pids %d  \n", domain_width, num_pids ); 
    _QSim_prop_lookup<T><<<numBlocks,threadsPerBlock>>>( sim, lookup, domain, domain_width, pids, num_pids );
}


template void QSim_prop_lookup(dim3, dim3, qsim<double>*, double*, double const*, unsigned, unsigned*, unsigned) ; 
template void QSim_prop_lookup(dim3, dim3, qsim<float>*,  float*,  float const*, unsigned, unsigned*, unsigned ) ; 







/**
ipid : index of the lookup outputs for that pid, which may differ from index of the pid   
**/

template <typename T>
__global__ void _QSim_prop_lookup_one(qsim<T>* sim, T* lookup, const T* domain, unsigned domain_width, unsigned num_pids, unsigned pid, unsigned ipid )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width || pid >= num_pids  ) return;

    T x = domain[ix] ;  
    T y = sim->prop->interpolate( pid, x ); 

    lookup[ipid*domain_width + ix] = y ; 
}

template <typename T>
extern  void QSim_prop_lookup_one(dim3 numBlocks, dim3 threadsPerBlock, qsim<T>* sim, T* lookup, const T* domain, unsigned domain_width, unsigned num_pids, unsigned pid, unsigned ipid )
{
    printf("//QSim_prop_lookup_one domain_width %d num_pids %d pid %d ipid %d \n", domain_width, num_pids, pid, ipid ); 
    _QSim_prop_lookup_one<T><<<numBlocks,threadsPerBlock>>>( sim, lookup, domain, domain_width, num_pids, pid, ipid );
}

template void QSim_prop_lookup_one(dim3, dim3, qsim<double>*, double*, const double*, unsigned, unsigned, unsigned, unsigned ) ; 
template void QSim_prop_lookup_one(dim3, dim3, qsim<float>*, float*, const float*, unsigned, unsigned, unsigned, unsigned ) ; 




