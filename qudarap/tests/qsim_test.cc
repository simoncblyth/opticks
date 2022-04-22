/**
qsim_test.cc : CPU tests of qsim.h CUDA code using mocking 
==============================================================

Standalone compile and run with::

   ./qsim_test.sh 

**/

#include "scuda.h"
#include "squad.h"
#include "qcurand.h"    // this brings in s_mock_curand.h for CPU 
#include "qsim.h"

void test_generate_photon_dummy(const qsim<float>* sim, curandStateXORWOW& rng)
{
    quad4 p ; 
    p.zero(); 
    quad6 gs ; 
    gs.zero(); 
    unsigned photon_id = 0 ; 
    unsigned genstep_id = 0 ; 
    sim->generate_photon_dummy(p, rng, gs, photon_id, genstep_id ); 
}

void test_uniform_sphere(const qsim<float>* sim, curandStateXORWOW& rng)
{
    for(int i=0 ; i < 10 ; i++)
    {
        float3 dir = sim->uniform_sphere(rng); 
        printf("//test_uniform_sphere dir (%10.4f %10.4f %10.4f) \n", dir.x, dir.y, dir.z ); 
    }
}

int main(int argc, char** argv)
{
    qsim<float>* sim = new qsim<float>() ; 
    curandStateXORWOW rng(1u); 

    test_generate_photon_dummy(sim, rng); 
    test_uniform_sphere(sim, rng);  

    return 0 ; 
}

