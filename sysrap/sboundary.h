#pragma once
/**
sboundary.h
=============

Riffing on qsim.h:propagate_at_boundary to assist with comparison with stmm.h 

comparing stmm.h with sboundary.h 
-------------------------------------

Q: pol comes in much later with stmm.h, how does it manage that ? 
A: By calculating the s and p coeffs and then only applying them to the actual pol as the final step 

   * in the sboundary.h expressions below that is factoring off the E1.x E1.y 
   * PERHAPS CAN FACTORIZE sboundary.h ANALOGOUSLY ?


From u4/CustomBoundary.h:doIt::

    267     const G4ThreeVector& surface_normal = theRecoveredNormal ;
    268     const G4ThreeVector& direction      = OldMomentum ;
    269     const G4ThreeVector& polarization   = OldPolarization ;
    ...
    271     G4double minus_cos_theta = direction*surface_normal ;
    272     G4double orientation = minus_cos_theta < 0. ? 1. : -1.  ;
    273     G4ThreeVector oriented_normal = orientation*surface_normal ;
    ...
    315     const double _si = stack.ll[0].st.real() ;
    ...
    327     // E_s2 : S-vs-P power fraction : signs make no difference as squared
    328     double E_s2 = _si > 0. ? (polarization*direction.cross(oriented_normal))/_si : 0. ;
  
    ///
    ///                                                                              ^^^  trans_length
    ///                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        trans
    ///                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ A_trans
    ///                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ E1_perp
    ///

    329     E_s2 *= E_s2;

    ////    ^^^^^   E1_perp*E1_perp

    ...
    340     double fT_s = stack.art.T_s ;
    341     double fT_p = stack.art.T_p ;
    342     double fR_s = stack.art.R_s ;
    343     double fR_p = stack.art.R_p ;
    344     double one = 1.0 ;
    345     double T = fT_s*E_s2 + fT_p*(one-E_s2);  // THIS MATCHES TransCoeff from sboundary.h 
    346     double R = fR_s*E_s2 + fR_p*(one-E_s2);
    347     double A = one - (T+R);

**/

#include "scurand.h"

struct sboundary
{
    sphoton& p ; 

    const sstate& s ; 
    const float& n1 ;
    const float& n2 ;    
    const float eta ;  
    const float3* normal ;     // geometrical outwards normal 
    const float mct ; 
    const float orient ; 
    const float c1 ; 
    const float c2c2 ; 
    const bool tir ;
    const float c2 ; 
    const float n1c1 ; 
    const float n2c2 ;
    const float n2c1 ;
    const float n1c2 ;
    const float2 _E2_t ;

    const float3 transverse ; 
    const float transverse_length ; 
    const bool  normal_incidence ; 
    const float3 A_transverse ; 
    const float E1_perp ; 

    const float2 E1   ;
    const float2 E1_chk   ;
    const float2 E2_t ;
    const float2 _E2_r ;
    const float2 E2_r ;

    const float2 RR ;
    const float2 TT ; 

    const float TransCoeff ; 
    const float ReflectCoeff ; 
    const float u_reflect ;
    const bool reflect ; 
    const unsigned flag ; 
    const float Coeff ; 
    const float EdotN ; 

    float3 A_parallel ; 

    sboundary(curandStateXORWOW& rng, sctx& ctx );  
};

inline sboundary::sboundary( curandStateXORWOW& rng, sctx& ctx ) 
    :
    p(ctx.p),
    s(ctx.s),
    n1(s.material1.x),
    n2(s.material2.x),
    eta(n1/n2),
    normal((float3*)&ctx.prd->q0.f.x),
    mct(dot(p.mom, *normal )),
    orient( mct < 0.f ? 1.f : -1.f ),
    c1(fabs(mct)),
    c2c2(1.f - eta*eta*(1.f - c1 * c1 )),   // Snells law and trig identity 
    tir(c2c2 < 0.f),
    c2( tir ? 0.f : sqrtf(c2c2)),
    n1c1(n1*c1),
    n2c2(n2*c2),
    n2c1(n2*c1),
    n1c2(n1*c2),
    _E2_t(make_float2( 2.f*n1c1/(n1c1+n2c2)   , 2.f*n1c1/(n2c1+n1c2) )),  // (ts,tp)  matches: real(c.stack.comp.ts,tp) 
    _E2_r(make_float2(  _E2_t.x - 1.f         , n2*_E2_t.y/n1 - 1.f  )),  // (rs,rp)  matches: real(c.stack.comp.rs,rp) 
    transverse(cross(p.mom, orient*(*normal))),
    transverse_length(length(transverse)),
    normal_incidence(transverse_length < 1e-6f ),
    A_transverse(normal_incidence ? p.pol : transverse/transverse_length),
    E1_perp(dot(p.pol, A_transverse)),
    E1(normal_incidence ? make_float2( 0.f, 1.f) : make_float2( E1_perp , length( p.pol - (E1_perp*A_transverse) ) )),   // ( S, P )
    E1_chk(make_float2( E1_perp, sqrtf(1.f - E1_perp*E1_perp) )),  // HMM: .x can be -ve, .y chosen +ve 
    E2_t(_E2_t*E1),        // deferred using pol to make closer to stmm.h 
    E2_r(_E2_r*E1),
    RR(normalize(E2_r)),   // elementwise multiplication is like non-uniform scaling, so cannot factor out E1 here 
    TT(normalize(E2_t)), 
    TransCoeff(tir || n1c1 == 0.f ? 0.f : n2c2*dot(E2_t,E2_t)/n1c1),
    ReflectCoeff(1.f - TransCoeff),
    u_reflect(curand_uniform(&rng)),
    reflect(u_reflect > TransCoeff), 
    flag(reflect ? BOUNDARY_REFLECT : BOUNDARY_TRANSMIT),
    Coeff(reflect ? ReflectCoeff : TransCoeff),
    EdotN(orient*dot(p.pol, *normal))
{
    p.mom = reflect
                    ?
                       p.mom + 2.0f*c1*orient*(*normal)
                    :
                       eta*(p.mom) + (eta*c1 - c2)*orient*(*normal)
                    ;

    A_parallel = normalize(cross(p.mom, A_transverse));   // A_paral is P-pol direction using new p.mom

    p.pol =  normal_incidence 
                    ?
                       ( reflect ?  p.pol*(n2>n1? -1.f:1.f) : p.pol )
                    :
                       ( reflect ?
                                    ( tir ?  -p.pol + 2.f*EdotN*orient*(*normal) : RR.x*A_transverse + RR.y*A_parallel )
                                 :
                                    TT.x*A_transverse + TT.y*A_parallel
                       )
                    ;
}

inline std::ostream& operator<<(std::ostream& os, const sboundary& b)  
{   
    os 
       << std::setw(20) << " n1 "           << std::setw(10) << std::fixed << std::setprecision(4) << b.n1  << std::endl 
       << std::setw(20) << " n2 "           << std::setw(10) << std::fixed << std::setprecision(4) << b.n2  << std::endl 
       << std::setw(20) << " eta "          << std::setw(10) << std::fixed << std::setprecision(4) << b.eta  << std::endl 
       << std::setw(20) << " normal "       << std::setw(10) << std::fixed << std::setprecision(4) << *b.normal  << std::endl
       << std::setw(20) << " mct "           << std::setw(10) << std::fixed << std::setprecision(4) << b.mct  << std::endl 
       << std::setw(20) << " orient "           << std::setw(10) << std::fixed << std::setprecision(4) << b.orient  << std::endl 
       << std::setw(20) << " c1 "           << std::setw(10) << std::fixed << std::setprecision(4) << b.c1  << std::endl 
       << std::setw(20) << " c2c2 "           << std::setw(10) << std::fixed << std::setprecision(4) << b.c2c2  << std::endl 
       << std::setw(20) << " tir "           << std::setw(10) << b.tir  << std::endl 
       << std::setw(20) << " c2 "           << std::setw(10) << std::fixed << std::setprecision(4) << b.c2  
       << std::endl 
       << std::setw(20) << " _E2_t "        <<  std::setw(10) << b._E2_t  
       << std::setw(20) << " length(_E2_t) " << std::setw(10) << std::fixed << std::setprecision(4) << length(b._E2_t)
       << std::endl
       << std::setw(20) << " _E2_r "        <<  std::setw(10) << b._E2_r 
       << std::setw(20) << " length(_E2_r) " << std::setw(10) << std::fixed << std::setprecision(4) << length(b._E2_r)
       << std::endl
       << std::setw(20) << " transverse "        << std::setw(10) << b.transverse
       << std::setw(20) << " length(transverse) " << std::setw(10) << std::fixed << std::setprecision(4) << length(b.transverse)  
       << std::endl
       << std::setw(20) << " transverse_length " << std::setw(10) << std::fixed << std::setprecision(4) <<  b.transverse_length  << std::endl
       << std::setw(20) << " normal_incidence " << std::setw(10) << std::fixed << std::setprecision(4) <<  b.normal_incidence  
       << std::endl
       << std::setw(20) << " A_transverse "      << std::setw(10) <<  b.A_transverse  
       << std::setw(20) << " length(A_transverse) " << std::setw(10) << std::fixed << std::setprecision(4) <<  length(b.A_transverse)  
       << std::endl
       << std::setw(20) << " E1_perp "      << std::setw(10) << std::fixed << std::setprecision(4) <<  b.E1_perp  
       << std::endl
       << std::setw(20) << " E1 "           <<  std::setw(10) << b.E1  
       << std::setw(20) << " length(E1) "   <<  std::setw(10) << std::fixed << std::setprecision(4) << length(b.E1)
       << std::endl
       << std::setw(20) << " E1_chk "       << std::setw(10) << b.E1_chk  
       << std::setw(20) << " length(E1) "   <<  std::setw(10) << std::fixed << std::setprecision(4) << length(b.E1_chk)
       << std::endl
       << std::setw(20) << " E2_t "         << std::setw(10)<<  b.E2_t  
       << std::setw(20) << " length(E2_t) " << std::setw(10) << std::fixed << std::setprecision(4) <<  length(b.E2_t)  
       << std::endl
       << std::setw(20) << " E2_r "         << std::setw(10) << b.E2_r  
       << std::setw(20) << " length(E2_r) " << std::setw(10) << std::fixed << std::setprecision(4) <<  length(b.E2_r)  
       << std::endl
       << std::setw(20) << " RR "           << std::setw(10) <<  b.RR 
       << std::setw(20) << " length(RR) "   << std::setw(10) << std::fixed << std::setprecision(4) <<  length(b.RR)  
       << std::endl
       << std::setw(20) << " TT "           << std::setw(10) <<  b.TT  
       << std::setw(20) << " length(TT) "   << std::setw(10) << std::fixed << std::setprecision(4) <<  length(b.TT)  
       << std::endl
       << std::setw(20) << " TransCoeff "   << std::setw(10) << std::fixed << std::setprecision(4) << b.TransCoeff  << std::endl
       << std::setw(20) << " ReflectCoeff " << std::setw(10) << std::fixed << std::setprecision(4) << b.ReflectCoeff  << std::endl
       << std::setw(20) << " u_reflect "    << std::setw(10) << std::fixed << std::setprecision(4) << b.u_reflect  << std::endl
       << std::setw(20) << " reflect "      << std::setw(10) << std::fixed << std::setprecision(4) << b.reflect  << std::endl
       << std::setw(20) << " flag "         << std::setw(10) << std::fixed << std::setprecision(4) << OpticksPhoton::Flag(b.flag)  << std::endl 
       << std::setw(20) << " Coeff "        << std::setw(10) << std::fixed << std::setprecision(4) << b.Coeff  << std::endl
       << std::setw(20) << " EdotN "        << std::setw(10) << std::fixed << std::setprecision(4) << b.EdotN  
       << std::endl
       << std::setw(20) << " p.mom "        << std::setw(10) << b.p.mom  
       << std::setw(20) << " length(p.mom) " << std::setw(10) << std::fixed << std::setprecision(4) << length(b.p.mom)  
       << std::endl 
       << std::setw(20) << " A_parallel "   << std::setw(10) << b.A_parallel 
       << std::setw(20) << " length(A_parallel) " << std::setw(10) << std::fixed << std::setprecision(4) << length(b.A_parallel) 
       << std::endl
       << std::setw(20) << " p.pol "        << std::setw(10) << b.p.pol  
       << std::setw(20) << " length(p.pol) " << std::setw(10) << std::fixed << std::setprecision(4) << length(b.p.pol)  
       << std::endl 
       << std::setw(20) << " dot(p.mom,p.pol) " << std::setw(10) << std::fixed << std::setprecision(4) << dot(b.p.mom,b.p.pol)  
       << std::endl 
       ;
    return os; 
}   

