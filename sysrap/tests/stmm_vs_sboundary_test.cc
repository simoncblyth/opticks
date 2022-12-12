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

int main(int argc, char** argv)
{
    curandStateXORWOW rng(1u) ; 
    rng.set_fake(0.);     // 0.:forced transmit 1.:forced reflect 

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
    float mct = dot(mom, nrm) ; 

    StackSpec<float,2> ss ; 

    ss.ls[0].nr = n1 ; 
    ss.ls[0].ni = 0.0f ; 
    ss.ls[0].d  = 0.0f ; 

    ss.ls[1].nr = n2 ; 
    ss.ls[1].ni = 0.0f ; 
    ss.ls[1].d  = 0.0f ; 

    Stack<float,2> stack(wl, mct, ss ); // ART calc done in ctor    
    float _si = stack.ll[0].st.real() ; 


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


        const float3& polarization = p.pol ; 
        const float3& direction = p.mom ; 
        const float3& oriented_normal = nrm ; 

        float E_s2 = _si > 0. ? dot(polarization,cross(direction,oriented_normal))/_si : 0. ; 
        E_s2 *= E_s2 ; 

        float TransCoeff2 = E_s2*stack.art.T_s + (1.f-E_s2)*stack.art.T_p ; 

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
            << " b.TransCoeff  " << std::setw(10) << std::fixed << std::setprecision(10) << b.TransCoeff 
            << std::endl
            << " b.TransCoeff2 " << std::setw(10) << std::fixed << std::setprecision(10) << TransCoeff2 
            << std::endl
            << " E_s2 " << std::setw(10) << std::fixed << std::setprecision(10) << E_s2
            << std::endl
            ;
     }

     std::cout << " stack " << stack << std::endl ; 


     NP* a = NP::Make<float>(N,2,4,4) ;
     a->read2<float>( (float*)pp.data() );
     a->save(FOLD, "pp.npy");
     std::cout << " save to " << FOLD << "/pp.npy" << std::endl;

     return 0 ; 
}
