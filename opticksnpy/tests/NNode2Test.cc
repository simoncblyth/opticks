#include <cstdlib>
#include <cfloat>

#include "NGLMExt.hpp"

#include "GLMFormat.hpp"
#include "NBox.hpp"
#include "NSphere.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"



void test_generateParPoints(const nnode* n, unsigned num_gen, unsigned sheetmask)
{
    std::vector<glm::vec3> points ; 
    std::vector<glm::vec3> normals ; 

    unsigned seed = 42 ; 

    glm::vec4 uvdom(0.45,0.55,0,1);

    n->generateParPoints( seed, uvdom, points, normals, num_gen, sheetmask ); 

    LOG(info) << "test_generateParPoints"
              << " num_gen " << num_gen 
              ;
    assert( points.size() == num_gen );
    assert( normals.size() == num_gen );

    for(unsigned i=0 ; i < num_gen ; i++)
    {
        const glm::vec3& pos = points[i] ; 
        const glm::vec3& nrm = normals[i] ; 
        std::cout 
            << " i " << std::setw(4) << i 
            << " pos " << gpresent(pos)
            << " plen " << std::setw(10) << glm::length(pos)
            << " nrm " << gpresent(nrm)
            << " nlen " << std::setw(10) << glm::length(nrm)
            << std::endl
            ;
    }
}

void test_generateParPoints_box()
{
    nbox n = make_box3(2.*1.f,2.*2.f,2.*3.f); 
    n.verbosity = 3 ;  
    n.pdump("make_box3(2.,4.,6.)");
    test_generateParPoints(&n, 60u, 0x3f );  //  b11 1111    
}

void test_generateParPoints_sphere()
{
    nsphere n = make_sphere(0.,0.,0.,10.); 
    n.verbosity = 3 ;  
    n.pdump("make_sphere(0,0,0,10)");
    test_generateParPoints(&n, 20u, 0 );   
}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    test_generateParPoints_box();
    //test_generateParPoints_sphere();


    return 0 ; 
}


