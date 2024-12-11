/**
sboundary_test.cc
====================

Build and run::

    ~/o/sysrap/tests/sboundary_test.sh


              
           +-st-+ 
            \   :
             \  ct
              \ :
               \:
         -------+--------

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

#include "sboundary.h"

const char* FOLD = getenv("FOLD") ; 



int main(int argc, char** argv)
{
    RNG rng ;   

    float3 nrm = make_float3(0.f, 0.f, 1.f ); // surface normal in +z direction 

    const int N = U::GetEnvInt("N",16) ; 
    float n1 = U::GetE<float>("N1", 1.f) ; 
    float n2 = U::GetE<float>("N2", 1.5f) ; 
    const char* AOI = U::GetEnv("AOI", "45" ); 
    const char force = U::GetE<char>("FORCE", 'N') ; 
    switch(force)
    {
        case 'R':rng.set_fake(1.f) ; break ;  
        case 'T':rng.set_fake(0.f) ; break ;  
    }

    float aoi = 0.f ; 
    if(strcmp(AOI, "BREWSTER") == 0)
    {
        aoi =  atanf(n2/n1) ; 
    }
    else if(strcmp(AOI, "CRITICAL") == 0)
    {
        assert( n2/n1 <= 1.f ); 
        aoi = asinf(n2/n1);  
    }
    else
    {
        aoi = std::atof(AOI)*M_PIf/180.f ; 
    }
    float3 mom = normalize(make_float3(sinf(aoi), 0.f, -cosf(aoi))) ; 


    std::cout 
        << " N " << N 
        << " N1 " << n1 
        << " N2 " << n2
        << " AOI " << AOI 
        << " FORCE " << force 
        << std::endl 
        << " aoi " << aoi 
        << " aoi/M_PIf " << aoi/M_PIf
        << " aoi/M_PIf*180 " << aoi/M_PIf*180.f
        << " mom " << mom 
        << std::endl 
        ;


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

    const int pnum = 3 ; 
    std::vector<sphoton> pp(N*pnum) ; 

    for(int i=0 ; i < N ; i++)  // loop over N photons with different polarization directions 
    {   
        sphoton& p0 = pp[i*3+0] ; 
        sphoton& p1 = pp[i*3+1] ; 
        sphoton& p2 = pp[i*3+2] ; 

        float frac_twopi = float(i)/float(N); // does not reach 1. to avoid including both 0 and 2pi  

        p.zero(); 

        p.mom = mom ;                     // all N with same momentum direction 
        p.set_polarization(frac_twopi) ;
        p.time = frac_twopi ; 

        p0 = p ;
        p0.wavelength = 0.1f ;  // used for scaling the pol vector

        sboundary b(rng, ctx); // changes ctx.p, p, b.p  (all the same reference)
 
        p1 = b.p ;
        p1.wavelength = b.Coeff ; 

        p2 = p1 ; 
        p2.pol = b.alt_pol ;   // p2: same as p1 but with alt_pol 

        std::cout
            << " b.flag " << OpticksPhoton::Flag(b.flag)
            << std::endl
            << " p0 " << p0.descDir()
            << std::endl
            << " p1 " << p1.descDir()
            << std::endl
            << " p2 " << p2.descDir()
            << std::endl
            ;

        std::cout << b ; 

     }

     NP* a = NP::Make<float>(N,pnum,4,4) ;
     a->read2<float>( (float*)pp.data() );
     a->save(FOLD, "pp.npy");
     std::cout << " save to " << FOLD << "/pp.npy" << std::endl;

     return 0 ; 
}
