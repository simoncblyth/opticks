#pragma once
/**
qscint.h
==================


**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QSCINT_METHOD __device__
#else
   #define QSCINT_METHOD 
#endif 

struct quad4 ; 
struct curandStateXORWOW ; 
struct quad6 ; 
struct sphoton ; 


//#include "stdio.h"


struct qscint
{
    cudaTextureObject_t scint_tex ; 
    quad4*              scint_meta ; // HUH: not used ? 
    unsigned            hd_factor ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
    QSCINT_METHOD void    generate( sphoton& p, curandStateXORWOW& rng, const quad6& gs, int photon_id, int genstep_id ) const ; 
    QSCINT_METHOD void    reemit(   sphoton& p, curandStateXORWOW& rng, float scintillationTime) const ;
    QSCINT_METHOD void    momw_polw(sphoton& p, curandStateXORWOW& rng) const ; 
    // sets direction, polarization and wavelength as needed by both generate and reemit

    QSCINT_METHOD float   wavelength(     curandStateXORWOW& rng) const ; 
    QSCINT_METHOD float   wavelength_hd0( curandStateXORWOW& rng) const ;  
    QSCINT_METHOD float   wavelength_hd10(curandStateXORWOW& rng) const ;
    QSCINT_METHOD float   wavelength_hd20(curandStateXORWOW& rng) const ;

#endif

}; 


#if defined(__CUDACC__) || defined(__CUDABE__)

#include "sscint.h"



/**
qscint::generate_photon
------------------------

**/

inline QSCINT_METHOD void qscint::generate(sphoton& p, curandStateXORWOW& rng, const quad6& _gs, int photon_id, int genstep_id ) const 
{
    momw_polw(p, rng ); 

    const sscint& gs = (const sscint&)_gs ; 

    float fraction = gs.charge == 0.f  ? 1.f : curand_uniform(&rng) ;   
    p.pos = gs.pos + fraction*gs.DeltaPosition ; 

    float u4 = curand_uniform(&rng) ; 
    float deltaTime = fraction*gs.step_length/gs.meanVelocity - gs.ScintillationTime*logf(u4) ;

    p.time = gs.time + deltaTime ; 
}


/**
qscint::reemit_photon
------------------------

OLD NOTES IN NEED OF REVIST : HOW TO HANDLE REEMISSION scintillationTime ?

As reemission happens inside scintillators for photons arising from Cerenkov (or Torch) 
gensteps need to special case the handing of the reemission scintillationTime somehow
because do not have access to scintillation gensteps when handling cerenkov or torch photons. 

Could carry the single float (could be domain compressed, it is eg 1.5 ns) in other gensteps ? 
But it is material specific (if you had more than one scintillator) 
just like REEMISSIONPROB so its more appropriate 
to live in the boundary_tex alongside the REEMISSIONPROB ?

But it could be carried in the genstep(or anywhere) as its use is "gated" by a non-zero REEMISSIONPROB.

Prefer to just hold it in the context, and provide G4Opticks::setReemissionScintillationTime API 
for setting it (default 0.) that is used from detector specific code which can read from 
the Geant4 properties directly.  What about geocache ? Can hold/persist with GScintillatorLib metadata.

**/

inline QSCINT_METHOD void qscint::reemit(sphoton& p, curandStateXORWOW& rng, float scintillationTime) const 
{
    momw_polw(p, rng); 
    float u3 = curand_uniform(&rng) ; 
    p.time += -scintillationTime*logf(u3) ;
}


/**
qscint::momw_polw : dir,pol and wavelength do not depend on genstep param
--------------------------------------------------------------------------------

Translation of "jcv DsG4Scintillation"

**/

inline QSCINT_METHOD void qscint::momw_polw(sphoton& p, curandStateXORWOW& rng) const 
{
    float u0 = curand_uniform(&rng); 
    float u1 = curand_uniform(&rng); 
    float u2 = curand_uniform(&rng); 

    float cost = 1.f - 2.f*u0;
    float sint = sqrt((1.f-cost)*(1.f+cost));
    float phi = 2.f*M_PIf*u1;
    float sinp = sin(phi);
    float cosp = cos(phi);

    p.mom.x = sint*cosp;  
    p.mom.y = sint*sinp;
    p.mom.z = cost ;  
    p.weight = 1.f ; 

    // Determine polarization of new photon 
    p.pol.x = cost*cosp ; 
    p.pol.y = cost*sinp ; 
    p.pol.z = -sint ;

    phi = 2.f*M_PIf*u2 ;
    sinp = sin(phi); 
    cosp = cos(phi); 

    p.pol = normalize( cosp*p.pol + sinp*cross(p.mom, p.pol) ) ;   
    p.wavelength = wavelength(rng);
}



inline QSCINT_METHOD float qscint::wavelength(curandStateXORWOW& rng) const 
{
    float wl ;  
    switch(hd_factor)
    {   
        case 0:  wl = wavelength_hd0(rng)  ; break ; 
        case 10: wl = wavelength_hd10(rng) ; break ; 
        case 20: wl = wavelength_hd20(rng) ; break ; 
        default: wl = 0.f ; 
    }   
    //printf("//qscint::wavelength wl %10.4f hd %d \n", wl, hd_factor ); 
    return wl ; 
}


inline QSCINT_METHOD float qscint::wavelength_hd0(curandStateXORWOW& rng) const 
{
    constexpr float y0 = 0.5f/3.f ; 
    float u0 = curand_uniform(&rng); 
    return tex2D<float>(scint_tex, u0, y0 ); 
}

/**
qscint::wavelength_hd10
--------------------------------------------------

Idea is to improve handling of extremes by throwing ten times the bins
at those regions, using simple and cheap linear mappings.

TODO: move hd "layers" into float4 payload so the 2d cerenkov and 1d scint
icdf texture can share some of teh implementation

**/

inline QSCINT_METHOD float qscint::wavelength_hd10(curandStateXORWOW& rng) const 
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



inline QSCINT_METHOD float qscint::wavelength_hd20(curandStateXORWOW& rng) const 
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


#endif


