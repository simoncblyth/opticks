#include <iostream>
#include <limits>
#include <algorithm>

#include "BRAP_LOG.hh"
#include "NGLM.hpp"
#include "GLMFormat.hpp"

#include "NMarchingCubesNPY.hpp"
#include "NTrianglesNPY.hpp"
#include "PLOG.hh"

#include "NSphereSDF.hpp"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    NSphereSDF s(0.,0.,0.,100.) ;

    const glm::uvec3 param(10,10,10);
    const glm::vec3 low( -100.,-100.,-100.); 
    const glm::vec3 high( 100., 100., 100.); 

    NMarchingCubesNPY<NSphereSDF> mcs;

    NTrianglesNPY* tris = mcs.march(s, param, low, high);

    assert(tris);


    return 0 ; 
}
