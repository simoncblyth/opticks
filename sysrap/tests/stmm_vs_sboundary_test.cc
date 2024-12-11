/**
stmm_vs_sboundary_test.cc
============================

::

   ~/o/sysrap/tests/stmm_vs_sboundary_test.sh


Specialize the stmm.h stack calc to 2 non-thin layers and compare with sboundary.h 

Had to adapt the sboundary.h polarization calc as what is in junoPMTOpticalModel 
looks clearly wrong. 
sboundary.h is a riff on qsim:propagate_at_boundary which is based on G4OpBoundaryProcess. 

* But might be missing some subtlety ? 

BUT: can the leap be made for the mom and pol appropriate to the real usage 
of a stack with 4 layers (2 thin in the middle) ? 


:google:`refraction stack of layers`

* https://www.reading.ac.uk/infrared/technical-library/thin-film-multilayers

* https://physics.stackexchange.com/questions/420069/resulting-refractive-index-of-multiple-layers-with-different-indexes

Snell's Law::

    n1 s1 = n2 s2 = n3 s3 = n4 s4 = n5 s5 

To find the angle at which the ray emerges you only need to consider the 2
mediums in which the ray enters and emerges. It does not matter what layers it
has passed through in between. If these 2 mediums are the same (eg air) then
the beam emerges parallel to the direction from which it entered.

Neither can the theorem tell you whether the ray undergoes total internal
reflection at one of the interfaces between layers of material. In that case
the ray will emerge from the face which it entered, or a side face if the slab
of multi-layer material is not wide enough. To find out if this happens you
need to trace the ray through the layers, checking what happens at each
boundary. 



* https://repository.tudelft.nl/islandora/object/uuid:3a6b5efa-99d9-481a-9185-6ef14d13429c/datastream/OBJ/download
* ~/opticks_refs/multilayer_Thesis_Version_XII.pdf



**/

#include "srngcpu.h"
using RNG = srngcpu ; 

#include "sphoton.h"
#include "sstate.h"
#include "srec.h"
#include "sseq.h"
#include "stag.h"
#include "sevent.h"
#include "sctx.h"
//#include "scurand.h"
#include "sboundary.h"
#include "stmm.h"

const char* FOLD = getenv("FOLD") ; 

struct scf
{
    float frac_twopi ; 
    float spare01 ; 
    float spare02 ; 
    float spare03 ; 

    float c_T ; 
    float b_TransCoeff ; 
    float spare12 ; 
    float spare13 ; 

    float c_transverse_length ;
    float b_transverse_length ;
    float spare22 ; 
    float spare23 ; 

    float c_E1_perp ; 
    float b_E1_perp ; 
    float spare32 ; 
    float spare33 ; 
};

inline std::ostream& operator<<(std::ostream& os, const scf& cf)  
{
    os
        << std::setw(25) << " frac_twopi "          << std::setw(10) << std::fixed << std::setprecision(10) << cf.frac_twopi
        << std::endl 
        << std::setw(25) << " c_T "                  << std::setw(10) << std::fixed << std::setprecision(10) << cf.c_T 
        << std::setw(25) << " b_TransCoeff  "       << std::setw(10) << std::fixed << std::setprecision(10)  << cf.b_TransCoeff 
        << std::setw(25) << " c-b "                  << std::setw(10) << std::fixed << std::setprecision(10)  << ( cf.c_T-cf.b_TransCoeff )
        << std::endl
        << std::setw(25) << " c_transverse_length " << std::setw(10) << std::fixed << std::setprecision(10) << cf.c_transverse_length 
        << std::setw(25) << " b_transverse_length " << std::setw(10) << std::fixed << std::setprecision(10) << cf.b_transverse_length 
        << std::setw(25) << " c-b "                 << std::setw(10) << std::fixed << std::setprecision(10) << ( cf.c_transverse_length - cf.b_transverse_length )
        << std::endl
        << std::setw(25) << " c_E1_perp "  << std::setw(10) << std::fixed << std::setprecision(10) << cf.c_E1_perp
        << std::setw(25) << " b_E1_perp "  << std::setw(10) << std::fixed << std::setprecision(10) << cf.b_E1_perp
        << std::setw(25) << " c-b "         << std::setw(10) << std::fixed << std::setprecision(10) << ( cf.c_E1_perp - cf.b_E1_perp )
        << std::endl
        ;
    return os ; 
}


struct Custom   
{
    static constexpr const float one = 1.f ; 
    const float3& nrm ; 
    const float3& mom ;
    const float wl ; 
    const float mct ; 
    const float orient ; 
    const float3 _transverse ;  
    const float  transverse_length ;  // same as s1 (sine of incident angle)
    const bool   normal_incidence ; 
    const float3 transverse ;  

    StackSpec<float,2> ss ; 
    Stack<float,2> stack ; 
   
    const float c1 ; 
    const float c2 ; 
    const float n1 ; 
    const float n2 ; 
    const float eta ;  
    const float c2c2 ; 
    const bool tir ; 

    const float2 fT ; 
    const float2 fR ; 

    const float2 _E2_t ; 
    const float2 _E2_r ; 

    Custom(const float3& nrm, const float3& mom, float n1, float n2, float wl ); 
    void set_polarization(const float3& polarization_) ; 

    float3 polarization ; 

    float EdotN ; 
    float E1_perp ; 
    float E_s2 ; 

    float2 E1 ; 
    float2 E1E1 ; 

    float T ; 
    float R ; 
    float A ; 

    float2 E2_t ; 
    float2 E2_r ; 
 
    float2 TT ; 
    float2 RR ; 

    void set_reflect( bool reflect_ ); 

    bool reflect ; 
    float3 new_mom ; 
    float3 parallel ; 
    float3 new_pol ; 
}; 

inline Custom::Custom(const float3& nrm_, const float3& mom_, float n1_, float n2_, float wl_)
    :
    nrm(nrm_),
    mom(mom_),
    wl(wl_),
    mct(dot(mom, nrm)), 
    orient( mct < 0.f ? 1.f : -1.f),
    _transverse(orient*cross(mom,nrm)),
    transverse_length(length(_transverse)),
    normal_incidence(transverse_length < 1e-6f ),
    transverse(normal_incidence ? make_float3(0.f, 0.f, 0.f) : _transverse/transverse_length),
    ss(StackSpec<float,2>::Create2(n1_,n2_)),
    stack(wl, mct, ss ),
    c1(real(stack.ll[0].ct)),
    c2(real(stack.ll[1].ct)),
    n1(real(stack.ll[0].n)),
    n2(real(stack.ll[1].n)),
    c2c2(1.f - eta*eta*(1.f - c1 * c1 )),   // Snells law and trig identity 
    tir(c2c2 < 0.f),   // HMM: maybe can just check imag(stack.ll[1].ct) ? 
    eta(n1/n2),
    fT(make_float2(stack.art.T_s, stack.art.T_p)),
    fR(make_float2(stack.art.R_s, stack.art.R_p)),
    _E2_t(make_float2(real(stack.comp.ts), real(stack.comp.tp))), 
    _E2_r(make_float2(real(stack.comp.rs), real(stack.comp.rp)))
{
    assert( n1 == n1_ ); 
    assert( n2 == n2_ ); 
}

inline void Custom::set_polarization(const float3& polarization_)
{ 
    polarization = polarization_ ; 

    EdotN = orient*dot(polarization, nrm) ; 
    E1_perp = dot(polarization,transverse); 
    E_s2 = E1_perp*E1_perp ; 
    E1   = make_float2( E1_perp, sqrtf(1.f - E_s2)) ;
    E1E1 = make_float2( E_s2, 1.f - E_s2); 
    //E1E1 = E1*E1 ;  

    T = dot( fT, E1E1 );    // equiv to fT.x*E_s2 + fT.y*(one-E_s2)
    R = dot( fR, E1E1 );    // equiv to fR.x*E_s2 + fR.y*(one-E_s2)
    A = one - (T + R ); 

    E2_t = E1*_E2_t ; 
    E2_r = E1*_E2_r ; 

    TT  = normalize(E2_t) ; 
    RR  = normalize(E2_r) ; 
}

inline void Custom::set_reflect( bool reflect_ )
{
    reflect = reflect_ ; 

    new_mom = reflect
                    ?
                       mom + 2.0f*c1*orient*nrm
                    :
                       eta*mom + (eta*c1 - c2)*orient*nrm
                    ;

    parallel = normalize(cross(new_mom, transverse));   // A_paral is P-pol direction using new p.mom

    new_pol =  normal_incidence 
                    ?
                       ( reflect ?  polarization*(n2>n1? -1.f:1.f) : polarization )
                    :
                       ( reflect ?
                                    ( tir ?  -polarization + 2.f*EdotN*orient*nrm : RR.x*transverse + RR.y*parallel )
                                 :
                                    TT.x*transverse + TT.y*parallel
                       )
                    ;
}

/**
T matches TransCoeff from sboundary.h despite very different impl, hopefully acts as Rosetta stone 
-----------------------------------------------------------------------------------------------------

The below demonstrates the match in the expressions, as done numerically already:

stmm.h::

    fT.x = stack.art.T_s = (n2c2/n1c1)*ts*ts  
    fT.y = stack.art.T_p = (n2c2/n1c1)*tp*tp  

    E_s2 = E1_perp*E1_perp 

    T = fT.x*E_s2 + fT.y*(one-E_s2);
      = (n2c2/n1c1)[ ts*ts*E_s2 + tp*tp*(1-E_s2) ] 

sboundary.h::

    E1 = E1_chk(make_float2( E1_perp, sqrtf(1.f - E1_perp*E1_perp) ))
    E1 =    ( E1_perp, sqrt( 1-E1_perp*E1_perp )) 
    E1 =    ( E1_perp, sqrt( 1-E1_perp*E1_perp )) 

    _E2_t =  (ts,tp)
    E2_t  = (_E2_t*E1)  

    TransCoeff = n2c2*dot(E2_t,E2_t)/n1c1 
               = (n2c2/n1c1)*dot(E2_t, E2_t)
               = (n2c2/n1c1)*( ts*ts*E_perp*E_perp + tp*tp*(1-E1_perp*E1_perp) ) 
               = (n2c2/n1c1)*( ts*ts*E_s2 + tp*tp*(1-E_s2) ) 

**/


int main(int argc, char** argv)
{
    RNG rng ; 
    rng.set_fake(0.);     // 0.:force transmit 1.:force reflect 

    float3 mom = normalize(make_float3(1.f, 0.f, -1.f)) ; 
    float3 nrm = make_float3(0.f, 0.f, 1.f ); // surface normal in +z direction 

    quad2 prd ; 
    prd.q0.f.x = nrm.x ; 
    prd.q0.f.y = nrm.y ; 
    prd.q0.f.z = nrm.z ;   

    sctx ctx ; 
    ctx.prd = &prd ; 

    float n1 = 1.f ; 
    float n2 = 1.5f ; 
    float wl = 420.f ; 

    Custom c(nrm, mom, n1, n2, wl); 

    sstate& s = ctx.s ; 
    s.material1.x = n1 ; 
    s.material2.x = n2 ; 

    sphoton& p = ctx.p ; 

    const int N = 16 ; 
    std::vector<sphoton> pp(N*2) ; 
    std::vector<scf>    cfs(N); 

    for(int i=0 ; i < N ; i++)
    {   
        sphoton& p0 = pp[i*2+0] ; 
        sphoton& p1 = pp[i*2+1] ; 
        scf& cf = cfs[i] ; 

        float frac_twopi = float(i)/float(N)  ;

        p.zero(); 
        p.mom = mom ; 
        p.set_polarization(frac_twopi) ;
        p0 = p ;

        c.set_polarization(p.pol); 


        sboundary b(rng, ctx); 
        p1 = b.p ;

        c.set_reflect(b.reflect) ;  


        cf.frac_twopi = frac_twopi ; 

        cf.c_transverse_length = c.transverse_length ; 
        cf.b_transverse_length = b.transverse_length ; 

        cf.c_T = c.T ; 
        cf.b_TransCoeff = b.TransCoeff ; 

        cf.c_E1_perp = c.E1_perp ; 
        cf.b_E1_perp = b.E1_perp ; 


        std::cout 
           << " b._E2_t " << b._E2_t  
           << " length(b._E2_t) " << length(b._E2_t) 
           << std::endl 
           << " c._E2_t " << c._E2_t
           << " length(c._E2_t) " << length(c._E2_t) 
           << std::endl 
           << std::endl 
           << " b._E2_r " << b._E2_r  
           << " length(b._E2_r) " << length(b._E2_r) 
           << std::endl 
           << " c._E2_r " << c._E2_r
           << " length(c._E2_r) " << length(c._E2_r) 
           << std::endl 
           << std::endl 
           << " b.E2_r "  << b.E2_r  << std::endl 
           << " c.E2_r "  << c.E2_r  << std::endl 
           << std::endl 
           << " c.E2_t "  << c.E2_t  << std::endl 
           << " b.E2_t "  << b.E2_t  << std::endl 
           << std::endl 
           << " b.E1   " << b.E1 
           << " length(b.E1) " << length(b.E1) 
           << std::endl 
           << " c.E1   " << c.E1 
           << " length(c.E1) " << length(c.E1) 
           << std::endl 
           << " b.E1_chk   " << b.E1_chk 
           << std::endl 
           << std::endl 
           << " b.RR " << b.RR 
           << " length(b.RR) " << length(b.RR) 
           << std::endl 
           << " c.RR " << c.RR 
           << " length(c.RR) " << length(c.RR) 
           << std::endl 
           << std::endl 
           << " b.TT   " << b.TT 
           << " length(b.TT)   " << length(b.TT) 
           << std::endl 
           << " c.TT   " << c.TT 
           << " length(c.TT)   " << length(c.TT) 
           << std::endl 
           << std::endl 
           << " c.fT   " << c.fT  << std::endl 
           << " c.fR   " << c.fR  << std::endl 
           << std::endl 
           << " c.E1E1   " << c.E1E1 
           << " length(c.E1E1) " << length(c.E1E1)
           << std::endl 
           << " c.E_s2 " << c.E_s2 << std::endl 
           << " 1.f-c.E_s2 " << 1.f-c.E_s2 << std::endl 
           << " c.T   " << c.T  << std::endl 
           << " c.R   " << c.R  << std::endl 
           << " c.A   " << c.A  << std::endl 
           << " b.flag " << OpticksPhoton::Flag(b.flag)
           << std::endl
           << " p0 " << p0.descDir()
           << std::endl
           << " p1 " << p1.descDir()
           << std::endl
           << cf 
           << std::endl
           << " c.new_mom " << c.new_mom << std::endl 
           << " c.new_pol " << c.new_pol << std::endl 
           << std::endl
           ;

     }

     //std::cout << " stack " << stack << std::endl ; 

     NP* a = NP::Make<float>(N,2,4,4) ;
     a->read2<float>( (float*)pp.data() );
     a->save(FOLD, "pp.npy");
     std::cout << " save to " << FOLD << "/pp.npy" << std::endl;

     NP* a_cfs = NP::Make<float>(N, 4, 4 ); 
     a_cfs->read2<float>( (float*)cfs.data() );
     a_cfs->save(FOLD, "cf.npy");
     std::cout << " save to " << FOLD << "/cf.npy" << std::endl;


     return 0 ; 
}
