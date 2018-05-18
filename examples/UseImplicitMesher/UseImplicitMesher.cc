//  externals/implicitmesher/implicitmesher/tests/ImplicitMesherFTest.cc

#include <iostream>

#include "ImplicitMesherF.h"
#include "SphereFunc.h"

int main()
{
    sphere_functor sf(0,0,0,10, false ) ; 
    //std::cout << sf.desc() << std::endl ; 

    int verbosity = 3 ; 
    ImplicitMesherF mesher(sf, verbosity, 0.f, false); 

    glm::vec3 min(-20,-20,-20);
    glm::vec3 max( 20, 20, 20);

    mesher.setParam(100, min, max);

    mesher.addSeed( 0,0,0, 1,0,0 );


    mesher.polygonize();
    if(verbosity > 0) mesher.dump();


    return 0 ; 
}

/**

Contrast the approaches:

ImplicitMesherF
    std::function ctor argument, use requires single header only

ImplicitMesher
   templated functor class, requires a boatload of public headers 



**/
 
