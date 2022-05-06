#pragma once

/**
qcerenkov.h
==============

FOR NOW NOT THE USUAL PHOTON : BUT DEBUGGING THE WAVELENGTH SAMPLING 
**/


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QCERENKOV_METHOD __device__
#else
   #define QCERENKOV_METHOD 
#endif 

#include "scurand.h"

struct qsim ; 
struct quad6 ; 
struct curandStateXORWOW ; 
#include "qgs.h"

struct qcerenkov
{
    // so far not using sphoton as generate nothing like a real photon 
    QCERENKOV_METHOD static void cerenkov_fabricate_genstep(               qsim* sim, GS& g, bool energy_range );

    // fabricating genstep every time !!
    QCERENKOV_METHOD static float   cerenkov_wavelength_rejection_sampled( qsim* sim, unsigned id, curandStateXORWOW& rng) ; 
    QCERENKOV_METHOD static float   cerenkov_wavelength_rejection_sampled( qsim* sim, unsigned id, curandStateXORWOW& rng, const GS& g);

    QCERENKOV_METHOD static void    cerenkov_photon(                       qsim* sim, quad4& p, unsigned id, curandStateXORWOW& rng ) ; 

    template<typename T>
    QCERENKOV_METHOD static void    cerenkov_photon_enprop(                qsim* sim, quad4& p, unsigned id, curandStateXORWOW& rng ) ; 

    template<typename T>
    QCERENKOV_METHOD static void    cerenkov_photon_enprop(                qsim* sim, quad4& p, unsigned id, curandStateXORWOW& rng, const GS& g ) ; 


    QCERENKOV_METHOD static void    cerenkov_photon(                       qsim* sim, quad4& p, unsigned id, curandStateXORWOW& rng, const GS& g ) ; 
    QCERENKOV_METHOD static void    cerenkov_photon_expt(                  qsim* sim, quad4& p, unsigned id, curandStateXORWOW& rng ); 

};



/**
qcerenkov::cerenkov_fabricate_genstep 
---------------------------------------

* currently uses hard coded values depending on RINDEX of material (LS) that will be used : ie fixing the cone angle
* a better way of doing this would use the MaterialLine as input and obtain the values from the RINDEX
* as this code could be arranged to be used on CPU only that is perfectly feasible  
* the focus of this fabricated genstep is wavelength generation 

TODO: move this down to storch for typical use on CPU only 
   
**/

inline QCERENKOV_METHOD void qcerenkov::cerenkov_fabricate_genstep(qsim* sim,  GS& g, bool energy_range )
{
    // picks the material line from which to get RINDEX
    unsigned MaterialLine = sim->boundary_tex_MaterialLine_LS ;  
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

    float Pmin = 1.55f ;    // eV
    float Pmax = 15.5f ; 

    g.ck1.BetaInverse = BetaInverse ;      //  g.ck1.BetaInverse/sampledRI  : yields the cone angle cosTheta


    // Wmin Wmax are poorly named as they atre used for energy when energy_range:true and wavelenth for energy_range:false
    if(energy_range)   
    {
        g.ck1.Wmin = Pmin ;   
        g.ck1.Wmax = Pmax ; 
    }
    else
    {
        g.ck1.Wmin = smath::hc_eVnm/Pmax ;            // close to: 1240./15.5 = 80.               
        g.ck1.Wmax = smath::hc_eVnm/Pmin ;            // close to: 1240./1.55 = 800.              
    }

    g.ck1.maxCos = maxCos  ;               //  is this used?          

    g.ck1.maxSin2 = maxSin2 ;              // constrains cone angle rejection sampling   
    g.ck1.MeanNumberOfPhotons1 = 0.f ; 
    g.ck1.MeanNumberOfPhotons2 = 0.f ; 
    g.ck1.postVelocity = 0.f ; 

} 








/**
qcerenkov::cerenkov_wavelength_rejection_sampled
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

inline QCERENKOV_METHOD float qcerenkov::cerenkov_wavelength_rejection_sampled(qsim* sim, unsigned id, curandStateXORWOW& rng, const GS& g) 
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

        float4 props = sim->boundary_lookup(wavelength, line, 0u); 

        sampledRI = props.x ;


        cosTheta = g.ck1.BetaInverse / sampledRI ;

        sin2Theta = fmaxf( 0.0001f, (1.f - cosTheta)*(1.f + cosTheta));  // avoid going -ve 

        u1 = curand_uniform(&rng) ;

        u_maxSin2 = u1*g.ck1.maxSin2 ;

    } while ( u_maxSin2 > sin2Theta);


    if( id == 0u )
    {
        printf("// qcerenkov::cerenkov_wavelength_rejection_sampled id %d sampledRI %7.3f cosTheta %7.3f sin2Theta %7.3f wavelength %7.3f \n", id, sampledRI, cosTheta, sin2Theta, wavelength );  
    }

    return wavelength ; 
}




inline QCERENKOV_METHOD void qcerenkov::cerenkov_photon(qsim* sim, quad4& q, unsigned id, curandStateXORWOW& rng, const GS& g )
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

        float4 props = sim->boundary_lookup( wavelength, line, 0u); 

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

        if( id == sim->pidx )
        {
            printf("//qcerenkov::cerenkov_photon id %d loop %3d u0 %10.5f ri %10.5f ct %10.5f s2 %10.5f u_mxs2_s2 %10.5f \n", id, loop, u0, sampledRI, cosTheta, sin2Theta, u_mxs2_s2 );
        }


    } while ( u_mxs2_s2 > 0.f );

    float energy = smath::hc_eVnm/wavelength ; 

    q.q0.f.x = energy ; 
    q.q0.f.y = wavelength ; 
    q.q0.f.z = sampledRI ; 
    q.q0.f.w = cosTheta ; 

    q.q1.f.x = sin2Theta ; 
    q.q1.u.y = 0u ; 
    q.q1.u.z = 0u ; 
    q.q1.f.w = g.ck1.BetaInverse ; 

    q.q2.f.x = w_linear ;    // linear sampled wavelenth
    q.q2.f.y = wavelength ;  // reciprocalized trick : does it really work  
    q.q2.f.z = u0 ; 
    q.q2.f.w = u1 ; 

    q.q3.u.x = line ; 
    q.q3.u.y = loop ; 
    q.q3.f.z = 0.f ; 
    q.q3.f.w = 0.f ; 
} 



/**
qcerenkov::cerenkov_photon_enprop
-----------------------------------

Variation assuming Wmin, Wmax contain Pmin Pmax and using qprop::interpolate 
to sample the RINDEX

**/



template<typename T>
inline QCERENKOV_METHOD void qcerenkov::cerenkov_photon_enprop(qsim* sim, quad4& q, unsigned id, curandStateXORWOW& rng, const GS& g )
{
    T u0 ;
    T u1 ; 
    T energy ; 
    T sampledRI ;
    T cosTheta ;
    T sin2Theta ;
    T u_mxs2_s2 ;

    T one(1.) ; 

    // should be MaterialLine no ?
    unsigned line = g.st.MaterialIndex ; //   line :  4*boundary_idx + OMAT/IMAT (0/3)
    unsigned loop = 0u ; 

    do {

        u0 = scurand<T>::uniform(&rng) ;

        energy = g.ck1.Wmin + u0*(g.ck1.Wmax - g.ck1.Wmin) ; 

        sampledRI = sim->prop->interpolate( 0u, energy ); 

        cosTheta = g.ck1.BetaInverse / sampledRI ;

        sin2Theta = (one - cosTheta)*(one + cosTheta);  

        u1 = scurand<T>::uniform(&rng) ;

        u_mxs2_s2 = u1*g.ck1.maxSin2 - sin2Theta ;

        loop += 1 ; 

        if( id == sim->pidx )
        {
            printf("//qcerenkov::cerenkov_photon_enprop id %d loop %3d u0 %10.5f ri %10.5f ct %10.5f s2 %10.5f u_mxs2_s2 %10.5f \n", id, loop, u0, sampledRI, cosTheta, sin2Theta, u_mxs2_s2 );
        }


    } while ( u_mxs2_s2 > 0.f );


    float wavelength = smath::hc_eVnm/energy ; 



    q.q0.f.x = energy ; 
    q.q0.f.y = wavelength ; 
    q.q0.f.z = sampledRI ; 
    q.q0.f.w = cosTheta ; 

    q.q1.f.x = sin2Theta ; 
    q.q1.u.y = 0u ; 
    q.q1.u.z = 0u ; 
    q.q1.f.w = g.ck1.BetaInverse ; 

    q.q2.f.x = 0.f ; 
    q.q2.f.y = 0.f ; 
    q.q2.f.z = u0 ; 
    q.q2.f.w = u1 ; 

    q.q3.u.x = line ; 
    q.q3.u.y = loop ; 
    q.q3.f.z = 0.f ; 
    q.q3.f.w = 0.f ; 
} 









/**
qcerenkov::cerenkov_photon_expt
-------------------------------------

This does the sampling all in double, narrowing to 
float just for the photon output.

Note that this is not using a genstep.

Which things have most need to be  double to make any difference ?

**/

inline QCERENKOV_METHOD void qcerenkov::cerenkov_photon_expt(qsim* sim, quad4& q, unsigned id, curandStateXORWOW& rng )
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
        sampledRI = sim->prop->interpolate( 0u, energy ); 
        //oneMinusCosTheta = (sampledRI - BetaInverse) / sampledRI ; 
        //reject = u1*maxOneMinusCosTheta - oneMinusCosTheta ;
        loop += 1 ; 

        cosTheta = BetaInverse / sampledRI ;
        sin2Theta = (1. - cosTheta)*(1. + cosTheta);  
        reject = u1*maxSin2 - sin2Theta ;

    } while ( reject > 0. );


    // narrowing for output 
    q.q0.f.x = energy ; 
    q.q0.f.y = smath::hc_eVnm/energy ;
    q.q0.f.z = sampledRI ; 
    //p.q0.f.w = 1. - oneMinusCosTheta ; 
    q.q0.f.w = cosTheta ; 

    q.q1.f.x = sin2Theta ; 
    q.q1.u.y = 0u ; 
    q.q1.u.z = 0u ; 
    q.q1.f.w = BetaInverse ; 

    q.q2.f.x = reject ; 
    q.q2.f.y = 0.f ; 
    q.q2.f.z = u0 ; 
    q.q2.f.w = u1 ; 

    q.q3.f.x = 0.f ; 
    q.q3.u.y = loop ; 
    q.q3.f.z = 0.f ; 
    q.q3.f.w = 0.f ; 
} 





/**
qcerenkov:cerenkov_wavelength_rejection_sampled
--------------------------------------------

HUH: this is using a GPU fabricated genstep everytime : that is kinda crazy approach.
Makes much more sense to fabricate genstep on CPU and upload it. 

**/


inline QCERENKOV_METHOD float qcerenkov::cerenkov_wavelength_rejection_sampled(qsim* sim, unsigned id, curandStateXORWOW& rng ) 
{
    QG qg ;      
    qg.zero();  
    GS& g = qg.g ; 
    bool energy_range = false ; 
    cerenkov_fabricate_genstep(sim, g, energy_range); 
    float wavelength = cerenkov_wavelength_rejection_sampled(sim, id, rng, g);   
    return wavelength ; 
}

inline QCERENKOV_METHOD void qcerenkov::cerenkov_photon(qsim* sim, quad4& p, unsigned id, curandStateXORWOW& rng ) 
{
    QG qg ;      
    qg.zero();  
    GS& g = qg.g ; 
    bool energy_range = false ; 
    cerenkov_fabricate_genstep(sim, g, energy_range); 
    cerenkov_photon(sim, p, id, rng, g ); 
}

template<typename T>
inline QCERENKOV_METHOD void qcerenkov::cerenkov_photon_enprop(qsim* sim, quad4& p, unsigned id, curandStateXORWOW& rng) 
{
    QG qg ;      
    qg.zero();  
    GS& g = qg.g ; 
    bool energy_range = true ; 
    cerenkov_fabricate_genstep(sim, g, energy_range); 

    cerenkov_photon_enprop<T>(sim, p, id, rng, g ); 
}

