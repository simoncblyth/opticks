#pragma once
/**
qcerenkov.h
==============

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QCERENKOV_METHOD __device__
#else
   #define QCERENKOV_METHOD 
#endif 

#include "scurand.h"
#include "smath.h"


struct scerenkov ; 
struct qbase ; 
struct qbnd ; 
struct quad6 ; 
struct curandStateXORWOW ; 
template <typename T> struct qprop ; 

/**

HMM: the qsim member causes chicken-egg setup problem, 
as want a cerenkov member of qsim 

* using it for sim->boundary_lookup, sim->prop 
  so need to break things up at finer level for easier reuse

**/

struct qcerenkov
{
    qbase* base ; 
    qbnd*  bnd ;  
    qprop<float>*  prop ;  

#if defined(__CUDACC__) || defined(__CUDABE__)
    QCERENKOV_METHOD void generate( sphoton& p,  curandStateXORWOW& rng, const quad6& gs    , int idx, int genstep_id ) const ;

    template<typename T>
    QCERENKOV_METHOD void wavelength_sampled_enprop( float& wavelength, float& cosTheta, float& sin2Theta, curandStateXORWOW& rng, const scerenkov& gs, int idx, int genstep_id ) const ;  
    QCERENKOV_METHOD void wavelength_sampled_bndtex( float& wavelength, float& cosTheta, float& sin2Theta, curandStateXORWOW& rng, const scerenkov& gs, int idx, int genstep_id ) const ; 

    QCERENKOV_METHOD void fraction_sampled(float& fraction, float& delta, curandStateXORWOW& rng, const scerenkov& gs, int idx, int gsid ) const ; 
#endif

};

#if defined(__CUDACC__) || defined(__CUDABE__)
inline QCERENKOV_METHOD void qcerenkov::generate( sphoton& p, curandStateXORWOW& rng, const quad6& _gs, int idx, int gsid ) const 
{
    const scerenkov& gs = (const scerenkov&)_gs ;
    const float3 p0 = normalize(gs.DeltaPosition) ;   // TODO: see of can normalize inside the genstep at collection  

    float wavelength ; 
    float cosTheta ; 
    float sin2Theta ; 

    wavelength_sampled_bndtex(wavelength, cosTheta, sin2Theta, rng, gs, idx, gsid) ;  
    //wavelength_sampled_enprop<float>(wavelength, cosTheta, sin2Theta, rng, gs, idx, gsid) ;  
    //wavelength_sampled_enprop<double>(wavelength, cosTheta, sin2Theta, rng, gs, idx, gsid) ;  

    float sinTheta = sqrtf(sin2Theta);

    // Generate random position of photon on cone surface 
    // defined by Theta 

    float u0 = curand_uniform(&rng); 
    float phi = 2.f*M_PIf*u0 ;
    float sinPhi = sin(phi); 
    float cosPhi = cos(phi); 

    // calculate x,y, and z components of photon energy
    // (in coord system with primary particle direction 
    //  aligned with the z axis)

    p.mom.x = sinTheta*cosPhi ; 
    p.mom.y = sinTheta*sinPhi ; 
    p.mom.z = cosTheta ; 

    // Rotate momentum direction back to global reference system 
    smath::rotateUz(p.mom, p0 ); 

    // Determine polarization of new photon 

    p.pol.x = cosTheta*cosPhi ; 
    p.pol.y = cosTheta*sinPhi ;
    p.pol.z = -sinTheta ;

   
    // Rotate back to original coord system 
    smath::rotateUz(p.pol, p0 ); 

    p.wavelength = wavelength ;  
    p.weight = 1.f ; 

    float fraction ; 
    float delta ; 

    fraction_sampled( fraction, delta, rng, gs, idx, gsid ); 

    float midVelocity = gs.preVelocity + fraction*( gs.postVelocity - gs.preVelocity )*0.5f ;   

    p.time = gs.time + delta / midVelocity ;
    p.pos = gs.pos + fraction * gs.DeltaPosition ;   // NB here gs.DeltaPosition must not be normalized 
} 

/**
qcerenkov::fraction_sampled
------------------------------

Note that N and NumberOfPhotons are never used below.
The point of the the rejection sampling loop is to come up with a 
*fraction* and *delta* that fulfils the theoretical constraint.  
This *fraction* controls where the photon gets generated along the 
genstep line segment with the rejection sampling serving 
to get the appropriate distribution of generation along that line. 

**/

inline QCERENKOV_METHOD void qcerenkov::fraction_sampled(float& fraction, float& delta, curandStateXORWOW& rng, const scerenkov& gs, int idx, int gsid ) const 
{
    float NumberOfPhotons ;   
    float N ; 
    float u ; 

    float MeanNumberOfPhotonsMax = fmaxf( gs.MeanNumberOfPhotons1, gs.MeanNumberOfPhotons2 );  
    float DeltaN = (gs.MeanNumberOfPhotons1-gs.MeanNumberOfPhotons2) ; 
    do  
    {   
        fraction = curand_uniform(&rng) ;

        delta = fraction * gs.step_length ;

        NumberOfPhotons = gs.MeanNumberOfPhotons1 - fraction * DeltaN  ;

        u = curand_uniform(&rng) ; 

        N = u * MeanNumberOfPhotonsMax ;

    } while (N > NumberOfPhotons);
}







/**
qcerenkov::wavelength_sampled_bndtex
-----------------------------------------------------

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

inline QCERENKOV_METHOD void qcerenkov::wavelength_sampled_bndtex(float& wavelength, float& cosTheta, float& sin2Theta, curandStateXORWOW& rng, const scerenkov& gs, int idx, int gsid ) const 
{
    float u0 ;
    float u1 ; 
    float w ; 
    float sampledRI ;
    float u_maxSin2 ;

    do {
        u0 = curand_uniform(&rng) ;

        w = gs.Wmin + u0*(gs.Wmax - gs.Wmin) ; 

        wavelength = gs.Wmin*gs.Wmax/w ; // arranges flat energy distribution, expressed in wavelength 

        float4 props = bnd->boundary_lookup(wavelength, gs.matline, 0u); 

        sampledRI = props.x ;

        cosTheta = gs.BetaInverse / sampledRI ;

        sin2Theta = fmaxf( 0.f, (1.f - cosTheta)*(1.f + cosTheta));  

        u1 = curand_uniform(&rng) ;

        u_maxSin2 = u1*gs.maxSin2 ;

    } while ( u_maxSin2 > sin2Theta );


    if( idx == 0u )
    {
        printf("// qcerenkov::cerenkov_wavelength_rejection_sampled idx %d sampledRI %7.3f cosTheta %7.3f sin2Theta %7.3f wavelength %7.3f \n", 
              idx , sampledRI, cosTheta, sin2Theta, wavelength );  
    }
}


/**
qcerenkov::wavelength_sampled_enprop
--------------------------------------

template type controls the type used for the rejection sampling, not the return type 

**/
template<typename T>
inline QCERENKOV_METHOD void qcerenkov::wavelength_sampled_enprop(float& f_wavelength, float& f_cosTheta, float& f_sin2Theta, curandStateXORWOW& rng, const scerenkov& gs, int idx, int gsid ) const 
{
    T u0 ;
    T u1 ; 
    T energy ; 
    T sampledRI ;
    T cosTheta ;
    T sin2Theta ;
    T u_mxs2_s2 ;

    T one(1.) ; 
    T zero(0.) ; 

    T pmin = gs.Pmin() ; 
    T pmax = gs.Pmax() ; 

    unsigned loop = 0u ; 

    do {

        u0 = scurand<T>::uniform(&rng) ;

        energy = pmin + u0*(pmax - pmin) ; 

        sampledRI = prop->interpolate( 0u, energy ); 

        cosTheta = gs.BetaInverse / sampledRI ;

        sin2Theta = (one - cosTheta)*(one + cosTheta);  

        u1 = scurand<T>::uniform(&rng) ;

        u_mxs2_s2 = u1*gs.maxSin2 - sin2Theta ;

        loop += 1 ; 

        if( idx == base->pidx )
        {
            printf("//qcerenkov::cerenkov_generate_enprop idx %d loop %3d u0 %10.5f ri %10.5f ct %10.5f s2 %10.5f u_mxs2_s2 %10.5f \n", idx, loop, u0, sampledRI, cosTheta, sin2Theta, u_mxs2_s2 );
        }


    } while ( u_mxs2_s2 > zero );

    f_wavelength = smath::hc_eVnm/energy ; 
    f_cosTheta = cosTheta ;  
    f_sin2Theta = sin2Theta ;  
}

#endif


















