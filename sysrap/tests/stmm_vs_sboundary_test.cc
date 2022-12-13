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
    const float3& nrm ; 
    const float3& mom ; 
    const float wl ; 
    const float mct ; 

    StackSpec<float,2> ss ; 
    Stack<float,2> stack ; 

    const float _si ; 
    const float2 fT ; 
    const float2 fR ; 

    Custom(const float3& nrm, const float3& mom, float n1, float n2, float wl ); 
}; 

inline Custom::Custom(const float3& nrm_, const float3& mom_, float n1, float n2, float wl_)
    :
    nrm(nrm_),
    mom(mom_),
    wl(wl_),
    mct(dot(mom, nrm)), 
    ss(StackSpec<float,2>::Create2(n1,n2)),
    stack(wl, mct, ss ),
    _si(stack.ll[0].st.real()),
    fT(make_float2(stack.art.T_s, stack.art.T_p)),
    fR(make_float2(stack.art.R_s, stack.art.R_p))
{
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


    for(int i=0 ; i < N ; i++)
    {   
        sphoton& p0 = pp[i*2+0] ; 
        sphoton& p1 = pp[i*2+1] ; 

        float frac_twopi = float(i)/float(N)  ;   

        p.zero(); 
        p.mom = mom ; 
        p.set_polarization(frac_twopi) ;



        // using normal and initial mom and pol to get the S_vs_P fraction 
        const float3& polarization = p.pol ; 
        const float3& direction    = p.mom ; 
        const float3& oriented_normal = nrm ; 

        float E1_perp = c._si > 0. ? dot(polarization,cross(direction,oriented_normal))/c._si : 0. ; 
        float E_s2 = E1_perp*E1_perp ; 

        float one = 1.f ; 
        float T = c.fT.x*E_s2 + c.fT.y*(one-E_s2);
        float R = c.fR.x*E_s2 + c.fR.y*(one-E_s2);
        float A = one - (T+R);





        p0 = p ;
        sboundary b(rng, ctx); 
        p1 = b.p ;


        std::cout
            << " b.flag " << OpticksPhoton::Flag(b.flag)
            << std::endl
            << " p0 " << p0.descDir()
            << std::endl
            << " p1 " << p1.descDir()
            << std::endl
            << std::setw(25) << " T "                   << std::setw(10) << std::fixed << std::setprecision(10) << T 
            << std::setw(25) << " b.TransCoeff  "       << std::setw(10) << std::fixed << std::setprecision(10) << b.TransCoeff 
            << std::endl
            << std::setw(25) << " c._si "               << std::setw(10) << std::fixed << std::setprecision(10) << c._si
            << std::setw(25) << " b.trans_length  "     << std::setw(10) << std::fixed << std::setprecision(10) << b.trans_length 
            << std::endl
            << std::setw(25) << " E_s2 "                << std::setw(10) << std::fixed << std::setprecision(10) << E_s2
            << std::setw(25) << " b.E1_perp*b.E1_perp " << std::setw(10) << std::fixed << std::setprecision(10) << b.E1_perp*b.E1_perp
            << std::endl
            ;
     }

     //std::cout << " stack " << stack << std::endl ; 

     NP* a = NP::Make<float>(N,2,4,4) ;
     a->read2<float>( (float*)pp.data() );
     a->save(FOLD, "pp.npy");
     std::cout << " save to " << FOLD << "/pp.npy" << std::endl;

     return 0 ; 
}
