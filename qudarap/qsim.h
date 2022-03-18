#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QSIM_METHOD __device__
#else
   #define QSIM_METHOD 
#endif 

#include "OpticksGenstep.h"
#include "sqat4.h"
#include "sc4u.h"
#include "sevent.h"

#include "qgs.h"
#include "qprop.h"
#include "qcurand.h"
#include "qbnd.h"
#include "qstate.h"

/**
qsim.h : GPU side struct prepared CPU side by QSim.hh
========================================================

Canonical use is from CSGOptiX/OptiX7Test.cu:simulate 

The qsim.h instance is uploaded once only at CSGOptiX instanciation 
as this encompasses the physics not the event-by-event info.


This is aiming to replace the OptiX 6 context in a CUDA-centric way.

* qsim encompasses global info relevant to to all photons, meaning that any changes
  make to the qsim instance from single photon threads must be into thread-owned slots 
  to avoid interference 
 
* temporary working state local to each photon is currently being passed by reference args, 
  would be cleaner to use a collective state struct to hold this local structs 

**/

struct curandStateXORWOW ; 
template <typename T> struct qprop ; 


template <typename T>
struct qsim
{
    curandStateXORWOW*  rngstate ; 

    cudaTextureObject_t scint_tex ; 
    quad4*              scint_meta ;
    // hmm could encapsulate the above group into a qscint ? 
    // and follow a similar approach for qcerenkov 
    // ... hmm there is commonality between the icdf textures with hd_factor on top 
    // that needs to be profited from 


    

    static constexpr T one = T(1.) ;   

    // boundary tex is created/configured CPU side in QBnd
    cudaTextureObject_t boundary_tex ; 
    quad4*              boundary_meta ; 
    unsigned            boundary_tex_MaterialLine_Water ;
    unsigned            boundary_tex_MaterialLine_LS ; 
    // hmm could encapsulate the above group into a qbnd ?

    quad*               optical ;  
    qprop<T>*           prop ;  

    static constexpr float hc_eVnm = 1239.8418754200f ; // G4: h_Planck*c_light/(eV*nm) 
 
#if defined(__CUDACC__) || defined(__CUDABE__)

    QSIM_METHOD float4  boundary_lookup( unsigned ix, unsigned iy ); 
    QSIM_METHOD float4  boundary_lookup( float nm, unsigned line, unsigned k ); 

    QSIM_METHOD float   scint_wavelength_hd0(curandStateXORWOW& rng);  
    QSIM_METHOD float   scint_wavelength_hd10(curandStateXORWOW& rng);
    QSIM_METHOD float   scint_wavelength_hd20(curandStateXORWOW& rng);
    QSIM_METHOD void    scint_dirpol(quad4& p, curandStateXORWOW& rng); 
    QSIM_METHOD void    reemit_photon(quad4& p, float scintillationTime, curandStateXORWOW& rng);
    QSIM_METHOD void    scint_photon( quad4& p, GS& g, curandStateXORWOW& rng);
    QSIM_METHOD void    scint_photon( quad4& p, curandStateXORWOW& rng);

    QSIM_METHOD void    cerenkov_fabricate_genstep(GS& g, bool energy_range );

    QSIM_METHOD float   cerenkov_wavelength_rejection_sampled(unsigned id, curandStateXORWOW& rng, const GS& g);
    QSIM_METHOD float   cerenkov_wavelength_rejection_sampled(unsigned id, curandStateXORWOW& rng) ; 

    QSIM_METHOD void    cerenkov_photon(quad4& p, unsigned id, curandStateXORWOW& rng, const GS& g, int print_id = -1 ) ; 
    QSIM_METHOD void    cerenkov_photon(quad4& p, unsigned id, curandStateXORWOW& rng, int print_id = -1 ) ; 

    QSIM_METHOD void    cerenkov_photon_enprop(quad4& p, unsigned id, curandStateXORWOW& rng, const GS& g, int print_id = -1 ) ; 
    QSIM_METHOD void    cerenkov_photon_enprop(quad4& p, unsigned id, curandStateXORWOW& rng, int print_id = -1 ) ; 

    QSIM_METHOD void    cerenkov_photon_expt(  quad4& p, unsigned id, curandStateXORWOW& rng, int print_id = -1 ); 

    QSIM_METHOD void    generate_photon(      quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id  ); 
    QSIM_METHOD void    generate_photon_dummy(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id  ); 
    QSIM_METHOD void    generate_photon_torch(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id  ); 

    QSIM_METHOD void    fill_state(qstate& s, int boundary, float wavelength ); 

#else
    qsim()
        :
        rngstate(nullptr),
        scint_tex(0),
        scint_meta(nullptr),
        boundary_tex(0),
        boundary_meta(nullptr),
        optical(nullptr),
        prop(nullptr)
    {
    }
#endif

}; 


// TODO: get the below to work on CPU with mocked curand and tex2D

#if defined(__CUDACC__) || defined(__CUDABE__)

/**
qsim::boundary_lookup ix iy : Low level integer addressing lookup
--------------------------------------------------------------------

**/

template <typename T>
inline QSIM_METHOD float4 qsim<T>::boundary_lookup( unsigned ix, unsigned iy )
{
    const unsigned& nx = boundary_meta->q0.u.x  ; 
    const unsigned& ny = boundary_meta->q0.u.y  ; 
    float x = (float(ix)+0.5f)/float(nx) ;
    float y = (float(iy)+0.5f)/float(ny) ;
    float4 props = tex2D<float4>( boundary_tex, x, y );     
    return props ; 
}

/**
qsim::boundary_lookup nm line k 
----------------------------------

nm:    float wavelength 
line:  4*boundary_index + OMAT/OSUR/ISUR/IMAT   (0/1/2/3)
k   :  property group index 0/1 

return float4 props 

**/
template <typename T>
inline QSIM_METHOD float4 qsim<T>::boundary_lookup( float nm, unsigned line, unsigned k )
{
    const unsigned& nx = boundary_meta->q0.u.x  ; 
    const unsigned& ny = boundary_meta->q0.u.y  ; 
    const float& nm0 = boundary_meta->q1.f.x ; 
    const float& nms = boundary_meta->q1.f.z ; 

    float fx = (nm - nm0)/nms ;  
    float x = (fx+0.5f)/float(nx) ;   // ?? +0.5f ??

    unsigned iy = _BOUNDARY_NUM_FLOAT4*line + k ;   
    float y = (float(iy)+0.5f)/float(ny) ; 

    float4 props = tex2D<float4>( boundary_tex, x, y );     
    return props ; 
}

/**
qsim::fill_state
-------------------

pick relevant boundary_tex lines depening on boundary sign, ie photon direction relative to normal

For example consider photons arriving at PMT cathode surface geometry normals point outwards 
so boundary sign will be -ve making line+OSUR the relevant surface

boundary 
   1 based code, signed by cos_theta of photon direction to outward geometric normal
   >0 outward going photon
   <0 inward going photon

NB the line is above the details of the payload (ie how many float4 per matsur) it is just::
 
    boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::fill_state(qstate& s, int boundary, float wavelength )
{
    const int line = ( boundary > 0 ? (boundary - 1) : (-boundary - 1) )*_BOUNDARY_NUM_MATSUR ;   
    const int m1_line = boundary > 0 ? line + IMAT : line + OMAT ;   
    const int m2_line = boundary > 0 ? line + OMAT : line + IMAT ;   
    const int su_line = boundary > 0 ? line + ISUR : line + OSUR ;   

    s.material1 = boundary_lookup( wavelength, m1_line, 0);  
    s.m1group2  = boundary_lookup( wavelength, m1_line, 1);  
    s.material2 = boundary_lookup( wavelength, m2_line, 0); 
    s.surface   = boundary_lookup( wavelength, su_line, 0);    

    s.optical = optical[su_line] ;   // index/type/finish/value
    s.index.x = optical[m1_line].x ; // m1 index
    s.index.y = optical[m2_line].x ; // m2 index 
    s.index.z = optical[su_line].x ; // su index

    //s.identity = identity ;   // feels pointless holding identity here, as already in callers scope : eg OptiX7Test.cu:simulate : so remove ?
}









template <typename T>
inline QSIM_METHOD float qsim<T>::scint_wavelength_hd0(curandStateXORWOW& rng) 
{
    constexpr float y0 = 0.5f/3.f ; 
    float u0 = curand_uniform(&rng); 
    return tex2D<float>(scint_tex, u0, y0 ); 
}

/**
qsim::scint_wavelength_hd10
--------------------------------------------------

Idea is to improve handling of extremes by throwing ten times the bins
at those regions, using simple and cheap linear mappings.

TODO: move hd "layers" into float4 payload so the 2d cerenkov and 1d scintillation
icdf texture can share some of teh implementation

**/

template <typename T>
inline QSIM_METHOD float qsim<T>::scint_wavelength_hd10(curandStateXORWOW& rng) 
{
    float u0 = curand_uniform(&rng); 
    float wl ; 

    constexpr float y0 = 0.5f/3.f ; 
    constexpr float y1 = 1.5f/3.f ; 
    constexpr float y2 = 2.5f/3.f ; 

    if( u0 < 0.1f )
    {
        wl = tex2D<float>(scint_tex, u0*10.f , y1 );    
    }
    else if ( u0 > 0.9f )
    {
        wl = tex2D<float>(scint_tex, (u0 - 0.9f)*10.f , y2 );    
    }
    else
    {
        wl = tex2D<float>(scint_tex, u0,  y0 ); 
    }
    return wl ; 
}



template <typename T>
inline QSIM_METHOD float qsim<T>::scint_wavelength_hd20(curandStateXORWOW& rng) 
{
    float u0 = curand_uniform(&rng); 
    float wl ; 

    constexpr float y0 = 0.5f/3.f ; 
    constexpr float y1 = 1.5f/3.f ; 
    constexpr float y2 = 2.5f/3.f ; 

    if( u0 < 0.05f )
    {
        wl = tex2D<float>(scint_tex, u0*20.f , y1 );    
    }
    else if ( u0 > 0.95f )
    {
        wl = tex2D<float>(scint_tex, (u0 - 0.95f)*20.f , y2 );    
    }
    else
    {
        wl = tex2D<float>(scint_tex, u0,  y0 ); 
    }
    return wl ; 
}

/**
qsim::cerenkov_wavelength_rejection_sampled
---------------------------------------------

wavelength between Wmin and Wmax is uniform-reciprocal-sampled 
to mimic uniform energy range sampling without taking reciprocals
twice

g4-cls G4Cerenkov::

    251   G4double Pmin = Rindex->GetMinLowEdgeEnergy();
    252   G4double Pmax = Rindex->GetMaxLowEdgeEnergy();
    253   G4double dp = Pmax - Pmin;
    254 
    255   G4double nMax = Rindex->GetMaxValue();
    256 
    257   G4double BetaInverse = 1./beta;
    258 
    259   G4double maxCos = BetaInverse / nMax;
    260   G4double maxSin2 = (1.0 - maxCos) * (1.0 + maxCos);
    261 
    ...
    270   for (G4int i = 0; i < fNumPhotons; i++) {
    271   
    272       // Determine photon energy
    273   
    274       G4double rand;
    275       G4double sampledEnergy, sampledRI;
    276       G4double cosTheta, sin2Theta;
    277 
    278       // sample an energy
    279 
    280       do {
    281          rand = G4UniformRand();
    282          sampledEnergy = Pmin + rand * dp;                // linear energy sample in Pmin -> Pmax
    283          sampledRI = Rindex->Value(sampledEnergy);
    284          cosTheta = BetaInverse / sampledRI;              // what cone angle that energy sample corresponds to 
    285 
    286          sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);   
    287          rand = G4UniformRand();
    288 
    289         // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
    290       } while (rand*maxSin2 > sin2Theta);                // constrain   
    291 

::

                        
                        
                  \    B    /                        
              \    .   |   .    /                            AC     ct / n          1         i       BetaInverse 
          \    C       |       C    /             cos th =  ---- =  --------   =  ------ =   ---  =  -------------
      \    .    \      |      /    .     /                   AB       bct           b n       n        sampledRI
       .         \    bct    /          .
                  \    |    /                                  BetaInverse
                   \   |   /  ct                  maxCos  =  ----------------- 
                    \  |th/  ----                                nMax                                                
                     \ | /    n
                      \|/
                       A

    Particle travels AB, light travels AC,  ACB is right angle 


     Only get Cerenkov radiation when   

            cos th <= 1 , 

            beta >= beta_min = 1/n        BetaInverse <= BetaInverse_max = n 


     At the small beta threshold AB = AC,   beta = beta_min = 1/n     eg for n = 1.333, beta_min = 0.75  

            cos th = 1,  th = 0         light is in direction of the particle 


     For ultra relativistic particle beta = 1, there is a maximum angle 

            th = arccos( 1/n )      

    In [5]: np.arccos(0.75)*180./np.pi
    Out[5]: 41.40962210927086


     So the beta range to have Cerenkov is  : 

                1/n       slowest, cos th = 1, th = 0   

          ->    1         ultra-relativistic, maximum cone angle th  arccos(1/n)     
 

     Corresponds to BetaInverse range 

           BetaInverse =  n            slowest, cos th = 1, th = 0    cone in particle direction 
    
           BetaInverse  = 1           



                
     The above considers a fixed refractive index.
     Actually refractive index varies with wavelength resulting in 
     a range of cone angles for a fixed particle beta.


    * https://www2.physics.ox.ac.uk/sites/default/files/2013-08-20/external_pergamon_jelley_pdf_18410.pdf
    * ~/opticks_refs/external_pergamon_jelley_pdf_18410.pdf


**/

template <typename T>
inline QSIM_METHOD float qsim<T>::cerenkov_wavelength_rejection_sampled(unsigned id, curandStateXORWOW& rng, const GS& g) 
{
    float u0 ;
    float u1 ; 
    float w ; 
    float wavelength ;

    float sampledRI ;
    float cosTheta ;
    float sin2Theta ;
    float u_maxSin2 ;

    // should be MaterialLine no ?
    unsigned line = g.st.MaterialIndex ; //   line :  4*boundary_idx + OMAT/IMAT (0/3)

    do {
        u0 = curand_uniform(&rng) ;

        w = g.ck1.Wmin + u0*(g.ck1.Wmax - g.ck1.Wmin) ; 

        wavelength = g.ck1.Wmin*g.ck1.Wmax/w ;  

        float4 props = boundary_lookup(wavelength, line, 0u); 

        sampledRI = props.x ;


        cosTheta = g.ck1.BetaInverse / sampledRI ;

        sin2Theta = fmaxf( 0.0001f, (1.f - cosTheta)*(1.f + cosTheta));  // avoid going -ve 

        u1 = curand_uniform(&rng) ;

        u_maxSin2 = u1*g.ck1.maxSin2 ;

    } while ( u_maxSin2 > sin2Theta);


    if( id == 0u )
    {
        printf("// qsim::cerenkov_wavelength_rejection_sampled id %d sampledRI %7.3f cosTheta %7.3f sin2Theta %7.3f wavelength %7.3f \n", id, sampledRI, cosTheta, sin2Theta, wavelength );  
    }

    return wavelength ; 
}

/**
FOR NOW NOT THE USUAL PHOTON : BUT DEBUGGING THE WAVELENGTH SAMPLING 
**/



template <typename T>
inline QSIM_METHOD void qsim<T>::cerenkov_photon(quad4& p, unsigned id, curandStateXORWOW& rng, const GS& g, int print_id )
{
    float u0 ;
    float u1 ; 


    float w_linear ; 
    float wavelength ;

    float sampledRI ;
    float cosTheta ;
    float sin2Theta ;
    float u_mxs2_s2 ;

    // should be MaterialLine no ?
    unsigned line = g.st.MaterialIndex ; //   line :  4*boundary_idx + OMAT/IMAT (0/3)

    unsigned loop = 0u ; 

    do {

#ifdef FLIP_RANDOM
        u0 = 1.f - curand_uniform(&rng) ;
#else
        u0 = curand_uniform(&rng) ;
#endif

        w_linear = g.ck1.Wmin + u0*(g.ck1.Wmax - g.ck1.Wmin) ; 

        wavelength = g.ck1.Wmin*g.ck1.Wmax/w_linear ;  

        float4 props = boundary_lookup( wavelength, line, 0u); 

        sampledRI = props.x ;

        cosTheta = g.ck1.BetaInverse / sampledRI ;

        sin2Theta = (1.f - cosTheta)*(1.f + cosTheta);  

#ifdef FLIP_RANDOM
        u1 = 1.f - curand_uniform(&rng) ;
#else
        u1 = curand_uniform(&rng) ;
#endif

        u_mxs2_s2 = u1*g.ck1.maxSin2 - sin2Theta ;

        loop += 1 ; 

        if( id == print_id )
        {
            printf("//qsim::cerenkov_photon id %d loop %3d u0 %10.5f ri %10.5f ct %10.5f s2 %10.5f u_mxs2_s2 %10.5f \n", id, loop, u0, sampledRI, cosTheta, sin2Theta, u_mxs2_s2 );
        }


    } while ( u_mxs2_s2 > 0.f );

    float energy = hc_eVnm/wavelength ; 

    p.q0.f.x = energy ; 
    p.q0.f.y = wavelength ; 
    p.q0.f.z = sampledRI ; 
    p.q0.f.w = cosTheta ; 

    p.q1.f.x = sin2Theta ; 
    p.q1.u.y = 0u ; 
    p.q1.u.z = 0u ; 
    p.q1.f.w = g.ck1.BetaInverse ; 

    p.q2.f.x = w_linear ;    // linear sampled wavelenth
    p.q2.f.y = wavelength ;  // reciprocalized trick : does it really work  
    p.q2.f.z = u0 ; 
    p.q2.f.w = u1 ; 

    p.q3.u.x = line ; 
    p.q3.u.y = loop ; 
    p.q3.f.z = 0.f ; 
    p.q3.f.w = 0.f ; 
} 



/**
qsim::cerenkov_photon_enprop
------------------------------

Variation assuming Wmin, Wmax contain Pmin Pmax and using qprop::interpolate 
to sample the RINDEX

**/




template <typename T>
inline QSIM_METHOD void qsim<T>::cerenkov_photon_enprop(quad4& p, unsigned id, curandStateXORWOW& rng, const GS& g, int print_id )
{
    T u0 ;
    T u1 ; 
    T energy ; 
    T sampledRI ;
    T cosTheta ;
    T sin2Theta ;
    T u_mxs2_s2 ;

    // should be MaterialLine no ?
    unsigned line = g.st.MaterialIndex ; //   line :  4*boundary_idx + OMAT/IMAT (0/3)

    unsigned loop = 0u ; 

    do {

        u0 = qcurand<T>::uniform(&rng) ;

        energy = g.ck1.Wmin + u0*(g.ck1.Wmax - g.ck1.Wmin) ; 

        sampledRI = prop->interpolate( 0u, energy ); 

        cosTheta = g.ck1.BetaInverse / sampledRI ;

        sin2Theta = (one - cosTheta)*(one + cosTheta);  

        u1 = qcurand<T>::uniform(&rng) ;

        u_mxs2_s2 = u1*g.ck1.maxSin2 - sin2Theta ;

        loop += 1 ; 

        if( id == print_id )
        {
            printf("//qsim::cerenkov_photon_enprop id %d loop %3d u0 %10.5f ri %10.5f ct %10.5f s2 %10.5f u_mxs2_s2 %10.5f \n", id, loop, u0, sampledRI, cosTheta, sin2Theta, u_mxs2_s2 );
        }


    } while ( u_mxs2_s2 > 0.f );


    float wavelength = hc_eVnm/energy ; 

    p.q0.f.x = energy ; 
    p.q0.f.y = wavelength ; 
    p.q0.f.z = sampledRI ; 
    p.q0.f.w = cosTheta ; 

    p.q1.f.x = sin2Theta ; 
    p.q1.u.y = 0u ; 
    p.q1.u.z = 0u ; 
    p.q1.f.w = g.ck1.BetaInverse ; 

    p.q2.f.x = 0.f ; 
    p.q2.f.y = 0.f ; 
    p.q2.f.z = u0 ; 
    p.q2.f.w = u1 ; 

    p.q3.u.x = line ; 
    p.q3.u.y = loop ; 
    p.q3.f.z = 0.f ; 
    p.q3.f.w = 0.f ; 
} 









/**
qsim::cerenkov_photon_expt
-------------------------------------

This does the sampling all in double, narrowing to 
float just for the photon output.

Note that this is not using a genstep.

Which things have most need to be  double to make any difference ?

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::cerenkov_photon_expt(quad4& p, unsigned id, curandStateXORWOW& rng, int print_id )
{
    double BetaInverse = 1.5 ; 
    double Pmin = 1.55 ; 
    double Pmax = 15.5 ; 
    double nMax = 1.793 ; 

    //double maxOneMinusCosTheta = (nMax - BetaInverse) / nMax;   

    double maxCos = BetaInverse / nMax;
    double maxSin2 = ( 1. - maxCos )*( 1. + maxCos ); 
    double cosTheta ;
    double sin2Theta ;

    double reject ;
    double u0 ;
    double u1 ; 
    double energy ; 
    double sampledRI ;

    //double oneMinusCosTheta ;

    unsigned loop = 0u ; 

    do {
        u0 = curand_uniform_double(&rng) ;
        u1 = curand_uniform_double(&rng) ;
        energy = Pmin + u0*(Pmax - Pmin) ; 
        sampledRI = prop->interpolate( 0u, energy ); 
        //oneMinusCosTheta = (sampledRI - BetaInverse) / sampledRI ; 
        //reject = u1*maxOneMinusCosTheta - oneMinusCosTheta ;
        loop += 1 ; 

        cosTheta = BetaInverse / sampledRI ;
        sin2Theta = (1. - cosTheta)*(1. + cosTheta);  
        reject = u1*maxSin2 - sin2Theta ;

    } while ( reject > 0. );


    // narrowing for output 
    p.q0.f.x = energy ; 
    p.q0.f.y = hc_eVnm/energy ;
    p.q0.f.z = sampledRI ; 
    //p.q0.f.w = 1. - oneMinusCosTheta ; 
    p.q0.f.w = cosTheta ; 

    p.q1.f.x = sin2Theta ; 
    p.q1.u.y = 0u ; 
    p.q1.u.z = 0u ; 
    p.q1.f.w = BetaInverse ; 

    p.q2.f.x = reject ; 
    p.q2.f.y = 0.f ; 
    p.q2.f.z = u0 ; 
    p.q2.f.w = u1 ; 

    p.q3.f.x = 0.f ; 
    p.q3.u.y = loop ; 
    p.q3.f.z = 0.f ; 
    p.q3.f.w = 0.f ; 
} 



/**
HMM ? state struct to collect this thread locals ?
**/

template <typename T>
inline QSIM_METHOD void qsim<T>::generate_photon(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id )
{
    int gencode = gs.q0.i.x ; 
    //printf("//qsim.generate_photon gencode %d \n", gencode); 
    switch(gencode)
    {
        case OpticksGenstep_TORCH: generate_photon_torch(p, rng, gs, photon_id, genstep_id) ; break ; 
        default:                   generate_photon_dummy(p, rng, gs, photon_id, genstep_id) ; break ; 
    }
}


/**
qsim::generate_photon_torch
-----------------------------

Acting on the gensteps created eg in QEvent 

Contrast with CPU implementation sysrap SEvent::GenerateCenterExtentGenstepsPhotons

The gensteps are for example configured in SEvent::MakeCenterExtentGensteps

NB the sevent.h enum order is different to the python one  eg XYZ=0 


TODO: this has not been updated following the cxs torch genstep rejig with transforms into the genstep ?


**/

template <typename T>
inline QSIM_METHOD void qsim<T>::generate_photon_torch(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id )
{

    C4U gsid ;  

    //int gencode          = gs.q0.i.x ;   
    int gridaxes           = gs.q0.i.y ;  // { XYZ, YZ, XZ, XY }
    gsid.u                 = gs.q0.i.z ; 
    //unsigned num_photons = gs.q0.u.w ; 


    p.q0.f.x = gs.q1.f.x ;   // start with genstep local frame position, typically origin  (0,0,0)   
    p.q0.f.y = gs.q1.f.y ; 
    p.q0.f.z = gs.q1.f.z ; 
    p.q0.f.w = 1.f ;        

    //printf("//qsim.generate_photon_torch gridaxes %d gs.q1 (%10.4f %10.4f %10.4f %10.4f) \n", gridaxes, gs.q1.f.x, gs.q1.f.y, gs.q1.f.z, gs.q1.f.w ); 

    float u0 = curand_uniform(&rng); 

    float sinPhi, cosPhi;
    sincosf(2.f*M_PIf*u0,&sinPhi,&cosPhi);

    float u1 = curand_uniform(&rng); 
    float cosTheta = 2.f*u1 - 1.f ; 
    float sinTheta = sqrtf(1.f-cosTheta*cosTheta) ; 

    //printf("//qsim.generate_photon_torch u0 %10.4f sinPhi   %10.4f cosPhi   %10.4f \n", u0, sinPhi, cosPhi ); 
    //printf("//qsim.generate_photon_torch u1 %10.4f sinTheta %10.4f cosTheta %10.4f \n", u1, sinTheta, cosTheta ); 
    //printf("//qsim.generate_photon_torch  u0 %10.4f sinPhi   %10.4f cosPhi   %10.4f u1 %10.4f sinTheta %10.4f cosTheta %10.4f \n",  u0, sinPhi, cosPhi, u1, sinTheta, cosTheta ); 

    switch( gridaxes )
    { 
        case YZ:  { p.q1.f.x = 0.f    ;  p.q1.f.y = cosPhi ;  p.q1.f.z = sinPhi ;  p.q1.f.w = 0.f ; } ; break ; 
        case XZ:  { p.q1.f.x = cosPhi ;  p.q1.f.y = 0.f    ;  p.q1.f.z = sinPhi ;  p.q1.f.w = 0.f ; } ; break ; 
        case XY:  { p.q1.f.x = cosPhi ;  p.q1.f.y = sinPhi ;  p.q1.f.z = 0.f    ;  p.q1.f.w = 0.f ; } ; break ; 
        case XYZ: { p.q1.f.x = sinTheta*cosPhi ;  
                    p.q1.f.y = sinTheta*sinPhi ;  
                    p.q1.f.z = cosTheta        ;  
                    p.q1.f.w = 0.f ; } ; break ;   // previously used XZ
    }


    qat4 qt(gs) ; // copy 4x4 transform from last 4 quads of genstep 
    qt.right_multiply_inplace( p.q0.f, 1.f );   // position 
    qt.right_multiply_inplace( p.q1.f, 0.f );   // direction 


    // HMM:  photon_id is global to the launch 
    // but having a  local to the photons of this genstep index is more useful for this id
    // so the genstep needs to carry the photon_id offset in order convert into a local index

    unsigned char ucj = (photon_id < 255 ? photon_id : 255 ) ;
    gsid.c4.w = ucj ; 
    p.q3.u.w = gsid.u ;

} 



template <typename T>
inline QSIM_METHOD void qsim<T>::generate_photon_dummy(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id )
{
    printf("//qsim::generate_photon_dummy  photon_id %3d genstep_id %3d  gs.q0.i ( gencode:%3d %3d %3d %3d ) \n", 
       photon_id, 
       genstep_id, 
       gs.q0.i.x, 
       gs.q0.i.y,
       gs.q0.i.z, 
       gs.q0.i.w
      );  

    p.q0.i.x = 1 ; p.q0.i.y = 2 ; p.q0.i.z = 3 ; p.q0.i.w = 4 ; 
    p.q1.i.x = 1 ; p.q1.i.y = 2 ; p.q1.i.z = 3 ; p.q1.i.w = 4 ; 
    p.q2.i.x = 1 ; p.q2.i.y = 2 ; p.q2.i.z = 3 ; p.q2.i.w = 4 ; 
    p.q3.i.x = 1 ; p.q3.i.y = 2 ; p.q3.i.z = 3 ; p.q3.i.w = 4 ; 
}






/**
qsim::cerenkov_wavelength with a fabricated genstep for testing
-----------------------------------------------------------------

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::cerenkov_fabricate_genstep(GS& g, bool energy_range )
{
    // picks the material line from which to get RINDEX
    unsigned MaterialLine = boundary_tex_MaterialLine_LS ;  
    float nMax = 1.793f ; 
    float BetaInverse = 1.500f ; 

    float maxCos = BetaInverse / nMax;
    float maxSin2 = (1.f - maxCos) * (1.f + maxCos) ;

    g.st.Id = 0 ; 
    g.st.ParentId = 0 ; 
    g.st.MaterialIndex = MaterialLine ; 
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

    float Pmin = 1.55f ; 
    float Pmax = 15.5f ; 

    g.ck1.BetaInverse = BetaInverse ;      //  g.ck1.BetaInverse/sampledRI  : yields the cone angle cosTheta

    if(energy_range)
    {
        g.ck1.Wmin = Pmin ;    
        g.ck1.Wmax = Pmax ; 
    }
    else
    {
        g.ck1.Wmin = hc_eVnm/Pmax ;            // close to: 1240./15.5 = 80.               
        g.ck1.Wmax = hc_eVnm/Pmin ;            // close to: 1240./1.55 = 800.              
    }

    g.ck1.maxCos = maxCos  ;               //  is this used?          

    g.ck1.maxSin2 = maxSin2 ;              // constrains cone angle rejection sampling   
    g.ck1.MeanNumberOfPhotons1 = 0.f ; 
    g.ck1.MeanNumberOfPhotons2 = 0.f ; 
    g.ck1.postVelocity = 0.f ; 

} 


template <typename T>
inline QSIM_METHOD float qsim<T>::cerenkov_wavelength_rejection_sampled(unsigned id, curandStateXORWOW& rng ) 
{
    QG qg ;      
    qg.zero();  
    GS& g = qg.g ; 
    bool energy_range = false ; 
    cerenkov_fabricate_genstep(g, energy_range); 
    float wavelength = cerenkov_wavelength_rejection_sampled(id, rng, g);   
    return wavelength ; 
}

template <typename T>
inline QSIM_METHOD void qsim<T>::cerenkov_photon(quad4& p, unsigned id, curandStateXORWOW& rng, int print_id ) 
{
    QG qg ;      
    qg.zero();  
    GS& g = qg.g ; 
    bool energy_range = false ; 
    cerenkov_fabricate_genstep(g, energy_range); 
    cerenkov_photon(p, id, rng, g, print_id); 
}

template <typename T>
inline QSIM_METHOD void qsim<T>::cerenkov_photon_enprop(quad4& p, unsigned id, curandStateXORWOW& rng, int print_id ) 
{
    QG qg ;      
    qg.zero();  
    GS& g = qg.g ; 
    bool energy_range = true ; 
    cerenkov_fabricate_genstep(g, energy_range); 
    cerenkov_photon_enprop(p, id, rng, g, print_id); 
}







/**
qsim::scint_dirpol
--------------------

Fills the photon quad4 struct with the below:

* direction, weight
* polarization, wavelength 

NB no position, time.

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::scint_dirpol(quad4& p, curandStateXORWOW& rng)
{
    float u0 = curand_uniform(&rng) ; 
    float u1 = curand_uniform(&rng) ; 
    float u2 = curand_uniform(&rng) ;   
    float u3 = curand_uniform(&rng) ;   

    float ct = 1.0f - 2.0f*u1 ;                 // -1.: 1. 
    float st = sqrtf( (1.0f-ct)*(1.0f+ct)) ; 
    float phi = 2.f*M_PIf*u2 ;
    float sp = sinf(phi); 
    float cp = cosf(phi); 
    float3 dir0 = make_float3( st*cp, st*sp,  ct ); 

    p.q1.f.x = dir0.x ; 
    p.q1.f.y = dir0.y ; 
    p.q1.f.z = dir0.z ; 
    p.q1.f.w = 1.f ;    // weight   

    float3 pol0 = make_float3( ct*cp, ct*sp, -st );
    float3 perp = cross( dir0, pol0 ); 
    float az =  2.f*M_PIf*u3 ; 
    float sz = sin(az);
    float cz = cos(az);
    float3 pol1 = normalize( cz*pol0 + sz*perp ) ; 

    p.q2.f.x = pol1.x ; 
    p.q2.f.y = pol1.y ; 
    p.q2.f.z = pol1.z ; 
    p.q2.f.w = scint_wavelength_hd20(rng); // hmm should this switch on hd_factor  
}

/**
Because reemission is possible (inside scintillators) for photons arising from Cerenkov (or Torch) 
gensteps need to special case handle the scintillationTime somehow ? 

Could carry the single float (could be domain compressed, it is eg 1.5 ns) in other gensteps ? 
But it is material specific just like REEMISSIONPROB so its more appropriate 
to live in the boundary_tex alongside the REEMISSIONPROB. 
But it could be carried in the genstep(or anywhere) as its use is "gated" by a non-zero REEMISSIONPROB.

Prefer to just hold it in the context, and provide G4Opticks::setReemissionScintillationTime API 
for setting it (default 0.) that is used from detector specific code which can read from 
the Geant4 properties directly.  What about geocache ? Can hold/persist with GScintillatorLib metadata.

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::reemit_photon(quad4& p, float scintillationTime, curandStateXORWOW& rng)
{
    scint_dirpol(p, rng); 
    float u4 = curand_uniform(&rng) ; 
    p.q0.f.w += -scintillationTime*logf(u4) ;
}

template <typename T>
inline QSIM_METHOD void qsim<T>::scint_photon(quad4& p, GS& g, curandStateXORWOW& rng)
{
    p.zero(); 
    scint_dirpol(p, rng); 

    float fraction = g.sc1.charge == 0.f  ? 1.f : curand_uniform(&rng) ;   
    float u4 = curand_uniform(&rng) ; 

    p.q0.f.x = g.st.x0.x + fraction*g.st.DeltaPosition.x ; 
    p.q0.f.y = g.st.x0.y + fraction*g.st.DeltaPosition.y ; 
    p.q0.f.z = g.st.x0.z + fraction*g.st.DeltaPosition.z ; 
    p.q0.f.w = g.st.t0   + fraction*g.st.step_length/g.sc1.midVelocity - g.sc1.ScintillationTime*logf(u4) ;
}


template <typename T>
inline QSIM_METHOD void qsim<T>::scint_photon(quad4& p, curandStateXORWOW& rng)
{
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

    scint_photon(p, g, rng); 
}





#endif

