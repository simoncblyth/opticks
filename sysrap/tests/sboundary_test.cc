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
    const float force_reflect = 1.f ; 
    const float force_transmit = 0.f ; 
    const char force = U::GetE<char>("FORCE", 'N') ; 
    switch(force)
    {
        case 'R':rng.set_fake(force_reflect) ; break ;  
        case 'T':rng.set_fake(force_transmit) ; break ;  
    }
    std::cout << " FORCE " << force << std::endl ;    


    /*
              
           +-st-+ 
            \   :
             \  ct
              \ :
               \:
         -------+--------

    */
    float3 nrm = make_float3(0.f, 0.f, 1.f ); // surface normal in +z direction 

    float n1 = 1.f ; 
    float n2 = 1.5f ; 

    const char* BREWSTER = "BREWSTER" ; 
    const char* AOI = U::GetEnv("AOI", BREWSTER ); 
    bool is_brewster = strcmp(AOI, BREWSTER) == 0  ; 
    float aoi = is_brewster ? atanf(n2/n1) : std::atof(AOI)*M_PIf/180.f ;   

    std::cout 
        << " AOI " << AOI 
        << " aoi " << aoi 
        << " aoi/M_PIf " << aoi/M_PIf
        << " aoi/M_PIf*180 " << aoi/M_PIf*180.f
        << std::endl 
        ;

    float3 mom = normalize(make_float3(sinf(aoi), 0.f, -cosf(aoi))) ; 

    quad2 prd ; 
    prd.q0.f.x = nrm.x ; 
    prd.q0.f.y = nrm.y ; 
    prd.q0.f.z = nrm.z ;   

    sctx ctx ; 
    ctx.prd = &prd ; 

    sstate& s = ctx.s ; 
    s.material1.x = n1 ; 
    s.material2.x = n2 ; 

    sphoton& p = ctx.p ; 

    const int N = U::GetEnvInt("N",16) ; 
    std::vector<sphoton> pp(N*2) ; 

    for(int i=0 ; i < N ; i++)
    {   
        sphoton& p0 = pp[i*2+0] ; 
        sphoton& p1 = pp[i*2+1] ; 

        float frac_twopi = float(i)/float(N)  ;   

        p.zero(); 

        p.mom = mom ; 
        p.set_polarization(frac_twopi) ;
        p.time = frac_twopi ; 

        p0 = p ;
        p0.wavelength = 0.1f ;  // used for scaling the pol vector

        sboundary b(rng, ctx); // changes ctx.p, p, b.p  (all the same reference)
 
        p1 = b.p ;
        p1.wavelength = b.Coeff ; 


        std::cout
            << " b.flag " << OpticksPhoton::Flag(b.flag)
            << std::endl
            << " p0 " << p0.descDir()
            << std::endl
            << " p1 " << p1.descDir()
            << std::endl
            ;

        std::cout << b ; 

     }

     NP* a = NP::Make<float>(N,2,4,4) ;
     a->read2<float>( (float*)pp.data() );
     a->save(FOLD, "pp.npy");
     std::cout << " save to " << FOLD << "/pp.npy" << std::endl;

     return 0 ; 
}
