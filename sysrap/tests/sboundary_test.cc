// ./sboundary_test.sh

#include "sphoton.h"
#include "sstate.h"
#include "srec.h"
#include "sseq.h"
#include "stag.h"
#include "sevent.h"
#include "sctx.h"
#include "scurand.h"
#include "sboundary.h"

const char* FOLD = getenv("FOLD") ; 

int main(int argc, char** argv)
{
    curandStateXORWOW rng(1u) ; 
    rng.set_fake(0.); // 0.:forced transmit 1.:forced reflect 

    float3 mom = normalize(make_float3(1.f, 0.f, -1.f)) ; 
    float3 nrm = make_float3(0.f, 0.f, 1.f ); // surface normal in +z direction 

    quad2 prd ; 
    prd.q0.f.x = nrm.x ; 
    prd.q0.f.y = nrm.y ; 
    prd.q0.f.z = nrm.z ;   

    sctx ctx ; 
    ctx.prd = &prd ; 

    sstate& s = ctx.s ; 
    s.material1.x = 1.0f ; 
    s.material2.x = 1.5f ; 

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
            ;
     }

     NP* a = NP::Make<float>(N,2,4,4) ;
     a->read2<float>( (float*)pp.data() );
     a->save(FOLD, "pp.npy");
     std::cout << " save to " << FOLD << "/pp.npy" << std::endl;

     return 0 ; 
}
