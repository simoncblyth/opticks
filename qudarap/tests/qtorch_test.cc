/**
qtorch_test.cc : CPU tests of qtorch.h CUDA code using mocking 
================================================================

Standalone compile and run with::

   ./qtorch_test.sh 

**/

#include "scuda.h"
#include "squad.h"
#include "qcurand.h"    // this brings in s_mock_curand.h for CPU 
#include "qsim.h"
#include "qtorch.h"

void test_generate(const qsim<float>* sim, curandStateXORWOW& rng)
{
    qphoton qp ; 
    qp.q.zero(); 
    photon& p = qp.p ; 

    qtorch qt ; 
    qt.q.zero();
    qt.q.q0.u = make_uint4( 1u, 2u, 3u, 100u ) ; 
    qt.t.mode = 255 ;    //torchmode::Type("...");  
    qt.t.type = torchtype::Type("disc");  


    float3 mom = make_float3( 1.f, 1.f, 1.f ); 

    torch& gs = qt.t ; 

    gs.wavelength = 501.f ; 
    gs.mom = normalize(mom); 
    gs.radius = 100.f ; 
    gs.zenith = make_float2( 0.f, 1.f ); 
    gs.azimuth = make_float2( 0.f, 1.f ); 


    // in reality quad6 gensteps come in from buffer, so above setup happens on CPU 

    unsigned photon_id = 0 ; 
    unsigned genstep_id = 0 ; 

    qtorch::generate(p, rng, gs, photon_id, genstep_id ); 

    std::cout << p.desc() << std::endl;  

}


int main(int argc, char** argv)
{
    qsim<float>* sim = new qsim<float>() ; 
    curandStateXORWOW rng(1u); 

    test_generate(sim, rng); 

    return 0 ; 
}

