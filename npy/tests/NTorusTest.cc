#include <cstdlib>
#include "NGLMExt.hpp"

#include "NTorus.hpp"
#include "PLOG.hh"



int main(int argc, char** argv)
{
    PLOG_(argc, argv);


    float r = 10.f ; 
    float R = 100.f ; 

    ntorus torus = make_torus( 0, 0, 10, 100) ; 

    float epsilon = 1e-5 ; 

    assert( fabsf(torus(R+r, 0, 0)) < epsilon );
    assert( fabsf(torus(R-r, 0, 0)) < epsilon );
    assert( fabsf(torus(-R+r, 0, 0)) < epsilon );
    assert( fabsf(torus(-R-r, 0, 0)) < epsilon );



    


    return 0 ; 
}






