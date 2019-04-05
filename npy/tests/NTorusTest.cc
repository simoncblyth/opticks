#include <cstdlib>
#include "NGLMExt.hpp"

#include "NTorus.hpp"
#include "OPTICKS_LOG.hh"



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    float r = 10.f ; 
    float R = 100.f ; 

    ntorus* _torus = make_torus( 0, 0, 10, 100) ; 
    const ntorus& torus = *_torus ; 
 
    float epsilon = 1e-5 ; 

    assert( fabsf(torus(R+r, 0, 0)) < epsilon );
    assert( fabsf(torus(R-r, 0, 0)) < epsilon );
    assert( fabsf(torus(-R+r, 0, 0)) < epsilon );
    assert( fabsf(torus(-R-r, 0, 0)) < epsilon );



    


    return 0 ; 
}






