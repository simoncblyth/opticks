// ./stmm_vs_sboundary_test.sh

#include "sphoton.h"
#include "sstate.h"
#include "srec.h"
#include "sseq.h"
#include "stag.h"
#include "sevent.h"
#include "sctx.h"
#include "scurand.h"
#include "sboundary.h"
#include "stmm.h"

const char* FOLD = getenv("FOLD") ; 

struct Custom   
{
    static constexpr const float one = 1.f ; 
    const float3& nrm ; 
    const float3& mom ; 
    const float wl ; 
    const float mct ; 
    const float orient ; 
    const float3 _transverse ;  
    const float  transverse_length ; 
    const bool   normal_incidence ; 
    const float3 transverse ;  

    StackSpec<float,2> ss ; 
    Stack<float,2> stack ; 

    const float _si ; 
    const float2 fT ; 
    const float2 fR ; 

    Custom(const float3& nrm, const float3& mom, float n1, float n2, float wl ); 
    void set_polarization(const float3& polarization_) ; 

    float3 polarization ; 
    float E1_perp ; 
    float E_s2 ; 
    float T ; 
    float R ; 
    float A ; 

}; 

inline Custom::Custom(const float3& nrm_, const float3& mom_, float n1, float n2, float wl_)
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
    ss(StackSpec<float,2>::Create2(n1,n2)),
    stack(wl, mct, ss ),
    _si(stack.ll[0].st.real()),
    fT(make_float2(stack.art.T_s, stack.art.T_p)),
    fR(make_float2(stack.art.R_s, stack.art.R_p))
{
}

inline void Custom::set_polarization(const float3& polarization_)
{ 
    polarization = polarization_ ; 
    E1_perp = dot(polarization,transverse); 
    E_s2 = E1_perp*E1_perp ; 
    T = fT.x*E_s2 + fT.y*(one-E_s2);
    R = fR.x*E_s2 + fR.y*(one-E_s2);
    A = one - (T+R);
}


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






int main(int argc, char** argv)
{
    curandStateXORWOW rng(1u) ; 
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

        cf.frac_twopi = frac_twopi ; 

        cf.c_transverse_length = c.transverse_length ; 
        cf.b_transverse_length = b.transverse_length ; 

        cf.c_T = c.T ; 
        cf.b_TransCoeff = b.TransCoeff ; 

        cf.c_E1_perp = c.E1_perp ; 
        cf.b_E1_perp = b.E1_perp ; 


        std::cout
            << " b.flag " << OpticksPhoton::Flag(b.flag)
            << std::endl
            << " p0 " << p0.descDir()
            << std::endl
            << " p1 " << p1.descDir()
            << std::endl
            << cf 
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
