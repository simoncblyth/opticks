/**
QSim.cu : extern void CUDA launch functions testing qsim.h methods
-------------------------------------------------------------------------------------

The launch functions are all invoked from QSim.cc methods with corresponding names.   


**/


#include "stdio.h"

#include "scuda.h"
#include "squad.h"
#include "scurand.h"
#include "sphoton.h"
#include "srec.h"


#include "qgs.h"
#include "qprop.h"
#include "qsim.h"
#include "qcerenkov.h"


#include "qevent.h"
#include "qdebug.h"

#include "QSimLaunch.hh"


/**
_QSim_rng_sequence
--------------------

id_offset : applies to sim.rngstate array controlling which curandState to use

**/

template <typename T>
__global__ void _QSim_rng_sequence(qsim* sim, T* seq, unsigned ni, unsigned nv, unsigned id_offset )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= ni) return;
    curandState rng = sim->rngstate[id+id_offset]; 
    unsigned ibase = id*nv ; 

    for(unsigned v=0 ; v < nv ; v++)
    {
        T u = scurand<T>::uniform(&rng) ;
        seq[ibase+v] = u ;
    } 
}


template <typename T>
extern void QSim_rng_sequence(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, T*  seq, unsigned ni, unsigned nv, unsigned id_offset )
{
    printf("//QSim_rng_sequence_f ni %d nv %d id_offset %d  \n", ni, nv, id_offset ); 
    _QSim_rng_sequence<T><<<numBlocks,threadsPerBlock>>>( sim, seq, ni, nv, id_offset );

}

template void QSim_rng_sequence(dim3, dim3, qsim*, float* , unsigned, unsigned, unsigned); 
template void QSim_rng_sequence(dim3, dim3, qsim*, double*, unsigned, unsigned, unsigned); 






/**
HMM hd_factor is more appropriate as a property of the uploaded texture than it is an input argument 
TODO: rearrange hd_factor 
**/

__global__ void _QSim_scint_wavelength(qsim* sim, float* wavelength, unsigned num_wavelength, unsigned hd_factor )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_wavelength) return;

    curandState rng = sim->rngstate[id]; 

    float wl ; 
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

extern void QSim_scint_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, float* wavelength, unsigned num_wavelength, unsigned hd_factor ) 
{
    printf("//QSim_scint_wavelength num_wavelength %d \n", num_wavelength ); 
    _QSim_scint_wavelength<<<numBlocks,threadsPerBlock>>>( sim, wavelength, num_wavelength, hd_factor );
} 








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





__global__ void _QSim_cerenkov_photon(qsim* sim, quad4* photon, unsigned num_photon )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_photon) return;

    curandState rng = sim->rngstate[idx]; 

    quad4 p ;   
    qcerenkov::cerenkov_photon(sim, p, idx, rng);   

    if(idx % 100000 == 0) printf("//_QSim_cerenkov_photon idx %d \n", idx  ); 
    photon[idx] = p ; 
}

extern void QSim_cerenkov_photon(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad4* photon, unsigned num_photon ) 
{
    printf("//QSim_cerenkov_photon num_photon %d \n", num_photon ); 
    _QSim_cerenkov_photon<<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon );
} 









template<typename T>
__global__ void _QSim_cerenkov_photon_enprop(qsim* sim, quad4* photon, unsigned num_photon )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_photon) return;

    curandState rng = sim->rngstate[idx]; 

    quad4 p ;   
    qcerenkov::cerenkov_photon_enprop<T>(sim, p, idx, rng);   

    if(idx % 100000 == 0) printf("//_QSim_cerenkov_photon_enprop idx %d \n", idx  ); 
    photon[idx] = p ; 
}


template<typename T>
extern void QSim_cerenkov_photon_enprop(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad4* photon, unsigned num_photon) 
{
    printf("//QSim_cerenkov_photon_enprop num_photon %d \n", num_photon ); 
    _QSim_cerenkov_photon_enprop<T><<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon );
} 


template void QSim_cerenkov_photon_enprop<float>(dim3, dim3, qsim*, quad4*, unsigned ); 
template void QSim_cerenkov_photon_enprop<double>(dim3, dim3, qsim*, quad4*, unsigned ); 



__global__ void _QSim_cerenkov_photon_expt(qsim* sim, quad4* photon, unsigned num_photon )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_photon) return;

    curandState rng = sim->rngstate[idx]; 

    quad4 p ;   
    qcerenkov::cerenkov_photon_expt(sim, p, idx, rng);   

    if(idx % 100000 == 0) printf("//_QSim_cerenkov_photon_expt idx %d \n", idx  ); 
    photon[idx] = p ; 
}

extern void QSim_cerenkov_photon_expt(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad4* photon, unsigned num_photon ) 
{
    printf("//QSim_cerenkov_photon_expt num_photon %d \n", num_photon ); 
    _QSim_cerenkov_photon_expt<<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon );
} 














__global__ void _QSim_scint_photon(qsim* sim, quad4* photon, unsigned num_photon )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_photon) return;
    
    //sim->r += id ;   
    //  would be problematic, do not want to change the the rng_states in global mem and get interference between threads

    curandState rng = sim->rngstate[idx] ; 

    quad4 p ;   
    sim->scint_photon(p, rng); 

    photon[idx] = p ; 
}

extern void QSim_scint_photon(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad4* photon, unsigned num_photon ) 
{
    printf("//QSim_scint_photon num_photon %d \n", num_photon ); 
    _QSim_scint_photon<<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon );
} 




__global__ void _QSim_generate_photon(qsim* sim, qevent* evt )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= evt->num_photon) return;
    
    curandState rng = sim->rngstate[idx] ; 
    unsigned genstep_id = evt->seed[idx] ; 
    const quad6& gs     = evt->genstep[genstep_id] ; 

    //printf("//_QSim_generate_photon idx %4d evt->num_photon %4d genstep_id %4d  \n", idx, evt->num_photon, genstep_id );  

    sphoton p ;   
    sim->generate_photon(p, rng, gs, idx, genstep_id ); 

    evt->photon[idx] = p ;  
}

extern void QSim_generate_photon(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, qevent* evt ) 
{
    printf("//QSim_generate_photon sim %p evt %p \n", sim, evt ); 
    // NB trying to de-reference the sim and evt pointers here gives "Bus error" 
    // because the code of this function is not running on device despite being compiled by nvcc
    _QSim_generate_photon<<<numBlocks,threadsPerBlock>>>( sim, evt );
} 




__global__ void _QSim_fill_state_0(qsim* sim, quad6* state,  unsigned num_state, qdebug* dbg )
{
    unsigned state_id = blockIdx.x*blockDim.x + threadIdx.x;
    printf("//_QSim_fill_state_0 state_id %d \n", state_id ); 

    if (state_id >= num_state) return;

    qstate s ; 

    float wavelength = dbg->wavelength ; 
    float cosTheta = dbg->cosTheta ;  
    int boundary = state_id + 1 ; 

    printf("//_QSim_fill_state_0 state_id %d  boundary %d wavelength %10.4f cosTheta %10.4f   \n", state_id, boundary, wavelength, cosTheta );  

    sim->fill_state(s, boundary, wavelength, cosTheta, state_id ); 

    state[state_id].q0.f = s.material1 ; 
    state[state_id].q1.f = s.m1group2 ; 
    state[state_id].q2.f = s.material2 ; 
    state[state_id].q3.f = s.surface ; 
    state[state_id].q4.u = s.optical ; 
    state[state_id].q5.u = s.index ; 

    //printf("//_QSim_fill_state_0 s.material1 %10.4f %10.4f %10.4f %10.4f \n", s.material1.x, s.material1.y, s.material1.z, s.material1.w ); 
}

extern void QSim_fill_state_0(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad6* state, unsigned num_state, qdebug* dbg )
{
    printf("//QSim_fill_state_0 sim %p state %p num_state %d dbg %p \n", sim, state, num_state, dbg ); 
    _QSim_fill_state_0<<<numBlocks,threadsPerBlock>>>( sim, state, num_state, dbg  );
} 




__global__ void _QSim_fill_state_1( qsim* sim, qstate* state,  unsigned num_state, qdebug* dbg )
{
    unsigned state_id = blockIdx.x*blockDim.x + threadIdx.x;
    printf("//_QSim_fill_state_1 blockIdx.x %d blockDim.x %d threadIdx.x %d state_id %d num_state %d \n", blockIdx.x, blockDim.x, threadIdx.x, state_id, num_state ); 

    if (state_id >= num_state) return;


    const float& wavelength = dbg->wavelength ; 
    const float& cosTheta = dbg->cosTheta ;  
    int boundary = state_id + 1 ; // boundary is 1-based  TOFIX: boundary now 0-based

    printf("//_QSim_fill_state_1 state_id %d  boundary %d wavelength %10.4f cosTheta %10.4f   \n", state_id, boundary, wavelength, cosTheta );  

    qstate s ; 
    sim->fill_state(s, boundary, wavelength, cosTheta, state_id ); 

    state[state_id] = s ; 

    //printf("//_QSim_fill_state_1 s.material1 %10.4f %10.4f %10.4f %10.4f \n", s.material1.x, s.material1.y, s.material1.z, s.material1.w ); 
}

extern void QSim_fill_state_1(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, qstate* state, unsigned num_state, qdebug* dbg )
{
    printf("//QSim_fill_state_1 sim %p state %p num_state %d dbg %p \n", sim, state, num_state, dbg ); 
    _QSim_fill_state_1<<<numBlocks,threadsPerBlock>>>( sim, state, num_state, dbg  );
} 




__global__ void _QSim_rayleigh_scatter_align( qsim* sim, sphoton* photon,  unsigned num_photon, qdebug* dbg )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("//_QSim_rayleigh_scatter_align blockIdx.x %d blockDim.x %d threadIdx.x %d id %d num_photon %d \n", blockIdx.x, blockDim.x, threadIdx.x, id, num_photon ); 

    if (idx >= num_photon) return;

    sphoton p = dbg->p ;    // need local copy of photon otherwise would have write interference between threads
    curandState rng = sim->rngstate[idx] ; 

    sim->rayleigh_scatter(p, rng);  

    photon[idx] = p ; 
}

__global__ void _QSim_propagate_to_boundary( qsim* sim, sphoton* photon, unsigned num_photon, qdebug* dbg )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("//_QSim_propagate_to_boundary blockIdx.x %d blockDim.x %d threadIdx.x %d propagate_id %d \n", blockIdx.x, blockDim.x, threadIdx.x, propagate_id ); 

    if (idx >= num_photon) return;

    const quad2* prd = &dbg->prd ;  // no need for local copy when readonly   
    const qstate& s  = dbg->s ;     
    sphoton p        = dbg->p ;    // need local copy of photon otherwise will have write interference between threads

    curandState rng = sim->rngstate[idx] ; 

    unsigned flag = 0u ;  
    sim->propagate_to_boundary( flag, p, prd, s, rng, idx );  
    p.set_flag(flag); 

    photon[idx] = p ; 

    //const float3* position = (float3*)&p.q0.f.x ; 
    //const float& time = p.q0.f.w ; 

    printf("//_QSim_propagate_to_boundary flag %d position %10.4f %10.4f %10.4f  time %10.4f  \n", flag, p.pos.x, p.pos.y, p.pos.z, p.time ); 

}

__global__ void _QSim_propagate_at_boundary_generate( qsim* sim, sphoton* photon, unsigned num_photon, qdebug* dbg )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("//_QSim_propagate_at_boundary_generate blockIdx.x %d blockDim.x %d threadIdx.x %d propagate_id %d \n", blockIdx.x, blockDim.x, threadIdx.x, propagate_id ); 

    if (idx >= num_photon) return;

    const quad2* prd = &dbg->prd ;  // no need for local copy when readonly   
    const qstate& s = dbg->s ;     

    sphoton p = dbg->p ;    // need local copy of photon otherwise will have write interference between threads
    quad4& q = (quad4&)p ; 

    curandState rng = sim->rngstate[idx] ; 

    q.q0.f = q.q1.f ;   // non-standard record initial mom and pol into q0, q3
    q.q3.f = q.q2.f ; 

    unsigned flag = 0 ; 
    sim->propagate_at_boundary( flag, p, prd, s, rng, idx );  

    q.q3.u.w = flag ;  // non-standard

    photon[idx] = p ; 
}


__global__ void _QSim_propagate_at_boundary_mutate( qsim* sim, sphoton* photon, unsigned num_photon, qdebug* dbg )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("//_QSim_propagate_at_boundary_mutate blockIdx.x %d blockDim.x %d threadIdx.x %d id %d \n", blockIdx.x, blockDim.x, threadIdx.x, id ); 

    if (idx >= num_photon) return;

    const quad2* prd = &dbg->prd ; 
    const qstate& s = dbg->s ;     

    sphoton p  = photon[idx] ; 
    quad4&  q  = (quad4&)p ; 

    curandState rng = sim->rngstate[idx] ; 

    q.q0.f = q.q1.f ;   // non-standard record initial mom and pol into q0, q3
    q.q3.f = q.q2.f ;
 
    unsigned flag = 0 ; 
    sim->propagate_at_boundary( flag, p, prd, s, rng, idx );  

    q.q3.u.w = flag ;  // non-standard

    photon[idx] = p ; 
}



__global__ void _QSim_hemisphere_polarized( qsim* sim, sphoton* photon, unsigned num_photon, qdebug* dbg, unsigned polz )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_photon) return;

    curandState rng = sim->rngstate[idx] ; 
    const quad2* prd = &dbg->prd ;  
    sphoton p        = dbg->p ;   
    bool inwards = true ; 

    sim->hemisphere_polarized( p, polz, inwards,  prd, rng );  

    photon[idx] = p ; 
}



__global__ void _QSim_reflect_generate( qsim* sim, sphoton* photon, unsigned num_photon, qdebug* dbg, unsigned type )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_photon) return;

    curandState rng = sim->rngstate[idx] ; 
    const quad2* prd = &dbg->prd ;  

    sphoton p = dbg->p ;   
    quad4& q = (quad4&)p ; 

    q.q0.f = q.q1.f ;   // non-standard record initial mom into p0 and initial pol into q3
    q.q3.f = q.q2.f ; 

    float u_decision_burn = curand_uniform(&rng);   // aligns consumption 
    //printf("//_QSim_reflect_generate id %d u_decision_burn %10.4f \n", id, u_decision_burn );  

    switch(type)
    {
        case REFLECT_DIFFUSE:   sim->reflect_diffuse(  p, prd, rng, idx) ;  break ;  
        case REFLECT_SPECULAR:  sim->reflect_specular( p, prd, rng, idx) ;  break ;  
    }
    photon[idx] = p ; 
}














__global__ void _QSim_random_direction_marsaglia( qsim* sim, quad* q, unsigned num_quad )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_quad ) return;

    curandState rng = sim->rngstate[idx] ; 

    float3* dir = (float3*)&q[idx].f.x ;  
    sim->random_direction_marsaglia( dir, rng, idx );  
    q[idx].u.w = idx ; 
}

__global__ void _QSim_lambertian_direction( qsim* sim, quad* q, unsigned num_quad, qdebug* dbg )
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= num_quad ) return;

    curandState rng = sim->rngstate[idx] ; 

    float3* dir = (float3*)&q[idx].f.x ;  
    const float orient = -1.f ; 

    sim->lambertian_direction( dir, &dbg->normal, orient, rng, idx );  

    q[idx].u.w = idx ; 
}


extern void QSim_quad_launch(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad* q, unsigned num_quad, qdebug* dbg, unsigned type  )
{
    const char* name = QSimLaunch::Name(type) ; 
    printf("//QSim_quad_launch sim %p quad %p num_quad %d dbg %p type %d name %s \n", sim, q, num_quad, dbg, type, name ); 

    switch(type)
    {
        case RANDOM_DIRECTION_MARSAGLIA: _QSim_random_direction_marsaglia<<<numBlocks,threadsPerBlock>>>(  sim, q, num_quad )        ; break ;
        case LAMBERTIAN_DIRECTION:       _QSim_lambertian_direction<<<numBlocks,threadsPerBlock>>>(        sim, q, num_quad, dbg )   ; break ;
    }
}


extern void QSim_photon_launch(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, sphoton* photon, unsigned num_photon, qdebug* dbg, unsigned type  )
{
    const char* name = QSimLaunch::Name(type) ; 
    printf("//QSim_photon_launch sim %p photon %p num_photon %d dbg %p type %d name %s \n", sim, photon, num_photon, dbg, type, name ); 
    switch(type)
    {
        case PROPAGATE_TO_BOUNDARY:  _QSim_propagate_to_boundary<<<numBlocks,threadsPerBlock>>>(  sim, photon, num_photon, dbg  )   ; break ;

        case RAYLEIGH_SCATTER_ALIGN: _QSim_rayleigh_scatter_align<<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon, dbg  )   ; break ;

        case HEMISPHERE_S_POLARIZED: _QSim_hemisphere_polarized<<<numBlocks,threadsPerBlock>>>(   sim, photon, num_photon, dbg, 0u  ) ; break ; 
        case HEMISPHERE_P_POLARIZED: _QSim_hemisphere_polarized<<<numBlocks,threadsPerBlock>>>(   sim, photon, num_photon, dbg, 1u  ) ; break ; 
        case HEMISPHERE_X_POLARIZED: _QSim_hemisphere_polarized<<<numBlocks,threadsPerBlock>>>(   sim, photon, num_photon, dbg, 2u  ) ; break ; 

        case PROPAGATE_AT_BOUNDARY:        
        case PROPAGATE_AT_BOUNDARY_NORMAL_INCIDENCE:        
                             _QSim_propagate_at_boundary_generate<<<numBlocks,threadsPerBlock>>>(  sim, photon, num_photon, dbg  )   ; break ;

        case PROPAGATE_AT_BOUNDARY_S_POLARIZED: 
        case PROPAGATE_AT_BOUNDARY_P_POLARIZED:
        case PROPAGATE_AT_BOUNDARY_X_POLARIZED:  
                             _QSim_propagate_at_boundary_mutate<<<numBlocks,threadsPerBlock>>>(    sim, photon, num_photon, dbg  ) ; break ;

        case REFLECT_DIFFUSE:  
        case REFLECT_SPECULAR:  
                            _QSim_reflect_generate<<<numBlocks,threadsPerBlock>>>( sim, photon, num_photon, dbg, type ) ; break ;  


    }
}


/**
_QSim_mock_propagate
-----------------------

TODO: compare performance using reference or pointer into global mem here rather than local stack copy    

**/

__global__ void _QSim_mock_propagate( qsim* sim, quad2* prd )
{
    qevent* evt = sim->evt ; 
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= evt->num_photon ) return;

    printf("//_QSim_mock_propagate idx %d evt.num_photon %d evt.max_record %d  \n", idx, evt->num_photon, evt->max_record ); 

    curandState rng = sim->rngstate[idx] ; 
    sphoton p = evt->photon[idx] ;   
    p.set_idx(idx); 

    sim->mock_propagate( p, prd, rng, idx );  

    evt->photon[idx] = p ; 
}


extern void QSim_mock_propagate_launch(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad2* prd )
{
    _QSim_mock_propagate<<<numBlocks,threadsPerBlock>>>( sim, prd ); 
}




__global__ void _QSim_boundary_lookup_all(qsim* sim, quad* lookup, unsigned width, unsigned height )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned index = iy * width + ix ;
    if (ix >= width | iy >= height ) return;

    quad q ; 
    q.f = sim->boundary_lookup( ix, iy ); 
    lookup[index] = q ; 
}

extern void QSim_boundary_lookup_all(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad* lookup, unsigned width, unsigned height )
{
    printf("//QSim_boundary_lookup width %d  height %d \n", width, height ); 
    _QSim_boundary_lookup_all<<<numBlocks,threadsPerBlock>>>( sim, lookup, width, height );
}



__global__ void _QSim_boundary_lookup_line(qsim* sim, quad* lookup, float* domain, unsigned num_lookup, unsigned line, unsigned k )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_lookup) return;
    float wavelength = domain[id] ;  
    quad q ; 
    q.f = sim->boundary_lookup( wavelength, line, k ); 
    lookup[id] = q ; 
}


extern void QSim_boundary_lookup_line(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad* lookup, float* domain, unsigned num_lookup, unsigned line, unsigned k )
{
    printf("//QSim_boundary_lookup_line num_lookup %d line %d k %d  \n", num_lookup, line, k ); 
    _QSim_boundary_lookup_line<<<numBlocks,threadsPerBlock>>>( sim, lookup, domain, num_lookup, line, k );
}



template <typename T>
__global__ void _QSim_prop_lookup(qsim* sim, T* lookup, const T* domain, unsigned domain_width, unsigned* pids, unsigned num_pids )
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
extern void QSim_prop_lookup( dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, T* lookup, const T* domain, unsigned domain_width, unsigned* pids, unsigned num_pids )
{
    printf("//QSim_prop_lookup domain_width %d num_pids %d  \n", domain_width, num_pids ); 
    _QSim_prop_lookup<<<numBlocks,threadsPerBlock>>>( sim, lookup, domain, domain_width, pids, num_pids );
}


template void QSim_prop_lookup(dim3, dim3, qsim*, double*, double const*, unsigned, unsigned*, unsigned) ; 
template void QSim_prop_lookup(dim3, dim3, qsim*,  float*,  float const*, unsigned, unsigned*, unsigned ) ; 







/**
ipid : index of the lookup outputs for that pid, which may differ from index of the pid   
**/

template <typename T>
__global__ void _QSim_prop_lookup_one(qsim* sim, T* lookup, const T* domain, unsigned domain_width, unsigned num_pids, unsigned pid, unsigned ipid )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width || pid >= num_pids  ) return;

    T x = domain[ix] ;  
    T y = sim->prop->interpolate( pid, x ); 

    lookup[ipid*domain_width + ix] = y ; 
}

template <typename T>
extern  void QSim_prop_lookup_one(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, T* lookup, const T* domain, unsigned domain_width, unsigned num_pids, unsigned pid, unsigned ipid )
{
    printf("//QSim_prop_lookup_one domain_width %d num_pids %d pid %d ipid %d \n", domain_width, num_pids, pid, ipid ); 
    _QSim_prop_lookup_one<<<numBlocks,threadsPerBlock>>>( sim, lookup, domain, domain_width, num_pids, pid, ipid );
}

template void QSim_prop_lookup_one(dim3, dim3, qsim*, double*, const double*, unsigned, unsigned, unsigned, unsigned ) ; 
template void QSim_prop_lookup_one(dim3, dim3, qsim*, float*, const float*, unsigned, unsigned, unsigned, unsigned ) ; 



__global__ void _QSim_multifilm_lookup_all(qsim* sim, quad2* sample,  quad2* result,  unsigned width, unsigned height )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned index = iy * width + ix ;
    if (ix >= width | iy >= height ) return;
    
    unsigned pmtType = sample[index].q0.u.x;
    unsigned bnd =     sample[index].q0.u.y;
    
    float    wv      = sample[index].q0.f.z;
    float    aoi     = sample[index].q0.f.w;

    float4 res = sim->multifilm_lookup( pmtType , bnd , wv, aoi );
     
    result[index].q0.u.x = pmtType;
    result[index].q0.u.y = bnd ;
    result[index].q0.f.z = wv ;
    result[index].q0.f.w = aoi;
    result[index].q1.f.x = res.x;
    result[index].q1.f.y = res.y;
    result[index].q1.f.z = res.z;
    result[index].q1.f.w = res.w;

    if(index < 100)
    {printf( "//index %d res.x %10.4f res.y %10.4f res.z %10.4f res.w %10.4f sample.x %10.4f sample.y %10.4f sample.z %10.4f sample.w %10.4f\n ",index,  res.x, res.y, res.z, res.w,  sample[index].q1.f.x, sample[index].q1.f.y, sample[index].q1.f.z, sample[index].q1.f.w); 
     }    

}


extern void QSim_multifilm_lookup_all(dim3 numBlocks, dim3 threadsPerBlock, qsim* sim, quad2* sample, quad2* result,  unsigned width, unsigned height )
{
    printf("//QSim_multifilm_lookup width %d  height %d \n", width, height ); 
    _QSim_multifilm_lookup_all<<<numBlocks,threadsPerBlock>>>( sim, sample,result , width, height );
}



    


