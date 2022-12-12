/**
qsim_test.cc : CPU tests of qsim.h CUDA code using mocking 
==============================================================

Standalone compile and run with::

   ./qsim_test.sh 

**/

#include "scuda.h"
#include "smath.h"
#include "squad.h"
#include "srec.h"
#include "stag.h"
#include "sphoton.h"
#include "scurand.h"    // this brings in s_mock_curand.h for CPU 
#include "qsim.h"

#include "OpticksPhoton.hh"
#include "sflow.h"

#include "NP.hh"

const char* FOLD = getenv("FOLD") ; 


void test_generate_photon_dummy(const qsim* sim, curandStateXORWOW& rng)
{
    sphoton p ; 
    p.zero(); 
    quad6 gs ; 
    gs.zero(); 
    unsigned photon_id = 0 ; 
    unsigned genstep_id = 0 ; 
    sim->generate_photon_dummy(p, rng, gs, photon_id, genstep_id ); 
}

void test_uniform_sphere(const qsim* sim, curandStateXORWOW& rng)
{
    for(int i=0 ; i < 10 ; i++)
    {
        float3 dir = sim->uniform_sphere(rng); 
        printf("//test_uniform_sphere dir (%10.4f %10.4f %10.4f) \n", dir.x, dir.y, dir.z ); 
    }
}


/**
test_propagate_at_boundary
---------------------------

               n
           i   :   r     
            \  :  /
             \ : /
              \:/
          -----+-----------
                \
                 \
                  \
                   t





Consider mom is some direction, say +Z::

   (0, 0, 1)

There is a circle of vectors that are perpendicular 
to that mom, all in the XY plane::

   ( cos(phi), sin(phi), 0 )    phi 0->2pi 

Clearly the dot product if that and +Z is zero. 
     


**/
void test_propagate_at_boundary(const qsim* sim, curandStateXORWOW& rng)
{
    float3 nrm = make_float3(0.f, 0.f, 1.f ); // surface normal in +z direction 
    float3 mom = normalize(make_float3(1.f, 0.f, -1.f)) ; 

    std::cout 
        << " test_propagate_at_boundary "
        << " nrm " << nrm 
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
    s.material1.x = 1.0f ; 
    s.material2.x = 1.5f ; 

    sphoton& p = ctx.p ; 
    p.zero(); 

    unsigned flag = 0 ;  
    const int N = 16 ; 

    std::vector<sphoton> pp(N*2) ; 

    for(int i=0 ; i < N ; i++)
    {   
        float frac_twopi = float(i)/float(N)  ;   

        p.mom = mom ; 
        p.set_polarization(frac_twopi) ;  

        sphoton p0(p) ;  
        int ctrl = sim->propagate_at_boundary(flag, rng, ctx) ; 

        pp[i*2+0 ] = p0 ; 
        pp[i*2+1 ] = p ; 

        std::cout 
            << " flag " << OpticksPhoton::Flag(flag) 
            << " ctrl " <<  sflow::desc(ctrl) 
            << std::endl
            << " p0 " << p0.descDir()
            << std::endl
            << " p  " << p.descDir() 
            << std::endl
            ; 
     }

     NP* a = NP::Make<float>(N,2,4,4) ; 
     a->read2<float>( (float*)pp.data() ); 
     a->save(FOLD, "pp.npy");  
     std::cout << " save to " << FOLD << "/pp.npy" << std::endl; 

}

int main(int argc, char** argv)
{
    qsim* sim = new qsim() ; 
    curandStateXORWOW rng(1u); 

    rng.set_fake(0.); // 0/1:forces transmit/reflect 

    /*
    test_generate_photon_dummy(sim, rng); 
    test_uniform_sphere(sim, rng);  
    */
    test_propagate_at_boundary(sim, rng); 

    return 0 ; 
}

