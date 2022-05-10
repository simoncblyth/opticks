





inline QCERENKOV_METHOD void qcerenkov::cerenkov_generate(qsim* sim, quad4& q, unsigned id, curandStateXORWOW& rng, const GS& g )
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
            printf("//qcerenkov::cerenkov_generate id %d loop %3d u0 %10.5f ri %10.5f ct %10.5f s2 %10.5f u_mxs2_s2 %10.5f \n", id, loop, u0, sampledRI, cosTheta, sin2Theta, u_mxs2_s2 );
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
qcerenkov::cerenkov_generate_enprop
-----------------------------------

Variation assuming Wmin, Wmax contain Pmin Pmax and using qprop::interpolate 
to sample the RINDEX

**/



template<typename T>
inline QCERENKOV_METHOD void qcerenkov::cerenkov_generate_enprop(qsim* sim, quad4& q, unsigned id, curandStateXORWOW& rng, const GS& g )
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
            printf("//qcerenkov::cerenkov_generate_enprop id %d loop %3d u0 %10.5f ri %10.5f ct %10.5f s2 %10.5f u_mxs2_s2 %10.5f \n", id, loop, u0, sampledRI, cosTheta, sin2Theta, u_mxs2_s2 );
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
qcerenkov::cerenkov_generate_expt
-------------------------------------

This does the sampling all in double, narrowing to 
float just for the photon output.

Note that this is not using a genstep.

Which things have most need to be  double to make any difference ?

**/

inline QCERENKOV_METHOD void qcerenkov::cerenkov_generate_expt(qsim* sim, quad4& q, unsigned id, curandStateXORWOW& rng )
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
    GS& g = qg.g ; 
    bool energy_range = false ; 
    cerenkov_fabricate_genstep(sim, g, energy_range); 
    float wavelength = cerenkov_wavelength_rejection_sampled(sim, id, rng, g);   
    return wavelength ; 
}

inline QCERENKOV_METHOD void qcerenkov::cerenkov_generate(qsim* sim, quad4& p, unsigned id, curandStateXORWOW& rng ) 
{
    QG qg ;      
    GS& g = qg.g ; 
    bool energy_range = false ; 
    cerenkov_fabricate_genstep(sim, g, energy_range); 
    cerenkov_generate(sim, p, id, rng, g ); 
}

template<typename T>
inline QCERENKOV_METHOD void qcerenkov::cerenkov_generate_enprop(qsim* sim, quad4& p, unsigned id, curandStateXORWOW& rng) 
{
    QG qg ;      
    GS& g = qg.g ; 
    bool energy_range = true ; 
    cerenkov_fabricate_genstep(sim, g, energy_range); 

    cerenkov_generate_enprop<T>(sim, p, id, rng, g ); 
}

