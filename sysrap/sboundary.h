#pragma once
/**
sboundary.h
=============

Riffing on qsim.h:propagate_at_boundary to assist with comparison with stmm.h 

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
    const float _c1 ; 
    const float orient ; 
    const float3 trans ; 
    const float trans_length ; 
    const bool  normal_incidence ; 
    const float3 A_trans ; 
    const float E1_perp ; 
    const float c1 ; 
    const float c2c2 ; 
    const bool tir ;
    const float EdotN ; 
    const float c2 ; 
    const float n1c1 ; 
    const float n2c2 ;
    const float n2c1 ;
    const float n1c2 ;
    const float2 E1   ;
    const float2 E2_t ;
    const float2 E2_r ;
    const float2 RR ;
    const float2 TT ; 
    const float TransCoeff ; 
    const float u_reflect ;
    const bool reflect ; 
    const unsigned flag ; 

    float3 A_paral ; 

    sboundary(curandStateXORWOW& rng, sctx& ctx );  
};

/**
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

inline sboundary::sboundary( curandStateXORWOW& rng, sctx& ctx ) 
    :
    p(ctx.p),
    s(ctx.s),
    n1(s.material1.x),
    n2(s.material2.x),
    eta(n1/n2),
    normal((float3*)&ctx.prd->q0.f.x),
    _c1(-dot(p.mom, *normal )),
    orient(_c1 < 0.f ? -1.f : 1.f ),
    trans(cross(p.mom, orient*(*normal))),
    trans_length(length(trans)),
    normal_incidence(trans_length < 1e-6f ),
    A_trans(normal_incidence ? p.pol : trans/trans_length),
    E1_perp(dot(p.pol, A_trans)),
    c1(fabs(_c1)),
    c2c2(1.f - eta*eta*(1.f - c1 * c1 )),   // Snells law and trig identity 
    tir(c2c2 < 0.f),
    EdotN(orient*dot(p.pol, *normal)),
    c2( tir ? 0.f : sqrtf(c2c2)),
    n1c1(n1*c1),
    n2c2(n2*c2),
    n2c1(n2*c1),
    n1c2(n1*c2),
    E1(normal_incidence ? make_float2( 0.f, 1.f) : make_float2( E1_perp , length( p.pol - (E1_perp*A_trans) ) )),   // ( S, P )
    E2_t(make_float2(  2.f*n1c1*E1.x/(n1c1+n2c2), 2.f*n1c1*E1.y/(n2c1+n1c2) )),  
    E2_r(make_float2( E2_t.x - E1.x             , (n2*E2_t.y/n1) - E1.y     )),
    RR(normalize(E2_r)),
    TT(normalize(E2_t)), 
    TransCoeff(tir || n1c1 == 0.f ? 0.f : n2c2*dot(E2_t,E2_t)/n1c1),
    u_reflect(curand_uniform(&rng)),
    reflect(u_reflect > TransCoeff), 
    flag(reflect ? BOUNDARY_REFLECT : BOUNDARY_TRANSMIT)
{
    p.mom = reflect
                    ?
                       p.mom + 2.0f*c1*orient*(*normal)
                    :
                       eta*(p.mom) + (eta*c1 - c2)*orient*(*normal)
                    ;

    A_paral = normalize(cross(p.mom, A_trans));   // A_paral is P-pol direction using new p.mom

    p.pol =  normal_incidence 
                    ?
                       ( reflect ?  p.pol*(n2>n1? -1.f:1.f) : p.pol )
                    :
                       ( reflect ?
                                    ( tir ?  -p.pol + 2.f*EdotN*orient*(*normal) : RR.x*A_trans + RR.y*A_paral )
                                 :
                                    TT.x*A_trans + TT.y*A_paral
                       )
                    ;
}

inline std::ostream& operator<<(std::ostream& os, const sboundary& b)  
{   
    os 
       << std::setw(20) << " n1 "           << std::setw(10) << std::fixed << std::setprecision(4) << b.n1  << std::endl 
       << std::setw(20) << " n2 "           << std::setw(10) << std::fixed << std::setprecision(4) << b.n2  << std::endl 
       << std::setw(20) << " eta "          << std::setw(10) << std::fixed << std::setprecision(4) << b.eta  << std::endl 
       << std::setw(20) << " _c1 "          << std::setw(10) << std::fixed << std::setprecision(4) << b._c1  << std::endl 
       << std::setw(20) << " normal "       << std::setw(10) << std::fixed << std::setprecision(4) << *b.normal  << std::endl
       << std::setw(20) << " trans "        << std::setw(10) << std::fixed << std::setprecision(4) <<  b.trans  << std::endl
       << std::setw(20) << " trans_length " << std::setw(10) << std::fixed << std::setprecision(4) <<  b.trans_length  << std::endl
       << std::setw(20) << " normal_incidence " << std::setw(10) << std::fixed << std::setprecision(4) <<  b.normal_incidence  << std::endl
       << std::setw(20) << " A_trans "      << std::setw(10) << std::fixed << std::setprecision(4) <<  b.A_trans  << std::endl
       << std::setw(20) << " E1_perp "      << std::setw(10) << std::fixed << std::setprecision(4) <<  b.E1_perp  << std::endl
       << std::setw(20) << " E1 "           << std::setw(10) << std::fixed << std::setprecision(4) <<  b.E1  << std::endl
       << std::setw(20) << " E2_t "         << std::setw(10) << std::fixed << std::setprecision(4) <<  b.E2_t  << std::endl
       << std::setw(20) << " TT "           << std::setw(10) << std::fixed << std::setprecision(4) <<  b.TT  << std::endl
       << std::setw(20) << " E2_r "         << std::setw(10) << std::fixed << std::setprecision(4) <<  b.E2_r  << std::endl
       << std::setw(20) << " RR "           << std::setw(10) << std::fixed << std::setprecision(4) <<  b.RR  << std::endl
       << std::setw(20) << " TransCoeff "   << std::setw(10) << std::fixed << std::setprecision(4) << b.TransCoeff  << std::endl
       << std::setw(20) << " u_reflect "    << std::setw(10) << std::fixed << std::setprecision(4) << b.u_reflect  << std::endl
       << std::setw(20) << " reflect "      << std::setw(10) << std::fixed << std::setprecision(4) << b.reflect  << std::endl
       << std::setw(20) << " p.mom "        << std::setw(10) << std::fixed << std::setprecision(4) << b.p.mom  << std::endl 
       << std::setw(20) << " A_paral "      << std::setw(10) << std::fixed << std::setprecision(4) <<  b.A_paral  << std::endl
       << std::setw(20) << " p.pol "        << std::setw(10) << std::fixed << std::setprecision(4) << b.p.pol  << std::endl 
       << std::setw(20) << " flag "         << std::setw(10) << std::fixed << std::setprecision(4) << OpticksPhoton::Flag(b.flag)  << std::endl 
       << std::endl 
       ;
    return os; 
}   

