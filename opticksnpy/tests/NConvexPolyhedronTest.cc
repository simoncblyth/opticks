#include "NCSG.hpp"
#include "NConvexPolyhedron.hpp"
#include "NBBox.hpp"
#include "NGLMExt.hpp"

#include <iostream>
#include <iomanip>

#include "PLOG.hh"
#include "NPY_LOG.hh"



nconvexpolyhedron test_make()
{
    nquad param ; 
    nquad param1 ; 
    nquad param2 ; 
    nquad param3 ; 

    param.u = {0,0,0,0} ;
    param1.u = {0,0,0,0} ;
    param2.u = {0,0,0,0} ;
    param3.u = {0,0,0,0} ;

    nconvexpolyhedron cpol = make_convexpolyhedron(param, param1, param2, param3 );
    return cpol ; 
}


void test_sdf(const nconvexpolyhedron* cpol)
{

    std::cout << "cpol(50,50,50) = " << (*cpol)(50,50,50) << std::endl ;
    std::cout << "cpol(-50,-50,-50) = " << (*cpol)(-50,-50,-50) << std::endl ;



    for(float v=-400.f ; v <= 400.f ; v+= 100.f )
    {
        float sd = (*cpol)(v,0,0) ;

        std::cout 
            << " x  " << std::setw(10) << v
            << " sd:  " << std::setw(10) << sd
/*
            << " y  " << std::setw(10) << (*cpol)(0,v,0)
            << " z  " << std::setw(10) << (*cpol)(0,0,v)
            << " xy " << std::setw(10) << (*cpol)(v,v,0)
            << " xz " << std::setw(10) << (*cpol)(v,0,v)
            << " yz " << std::setw(10) << (*cpol)(0,v,v)
            << "xyz " << std::setw(10) << (*cpol)(v,v,v)
*/
            << std::endl ; 
    } 
}



void test_intersect(const nconvexpolyhedron* cpol)
{
    glm::vec3 ray_origin(0,0,0);
    float t_min = 0.f ; 

    for(unsigned i=0 ; i < 3 ; i++)  
    {
        for(unsigned j=0 ; j < 2 ; j++)
        {
            glm::vec3 ray_direction(0,0,0);
            ray_direction.x = i == 0  ? (j == 0 ? 1 : -1 ) : 0 ; 
            ray_direction.y = i == 1  ? (j == 0 ? 1 : -1 ) : 0 ; 
            ray_direction.z = i == 2  ? (j == 0 ? 1 : -1 ) : 0 ; 
            std::cout << " dir " << ray_direction << std::endl ; 

            glm::vec4 isect(0.f);
            bool valid_intersect = cpol->intersect(  t_min , ray_origin, ray_direction , isect );
            assert(valid_intersect);

            std::cout << " isect : " << isect << std::endl ; 
        }
    }
}


void test_bbox(const nconvexpolyhedron* cpol)
{
    nbbox bb = cpol->bbox() ; 
    std::cout << bb.desc() << std::endl ; 
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 


    nnode* root = nnode::load(argc > 1 ?  argv[1] : "$TMP/tboolean-trapezoid--/1" , 1 );
    assert(root->type == CSG_CONVEXPOLYHEDRON );
    nconvexpolyhedron* cpol = (nconvexpolyhedron*)root  ;
    cpol->pdump(argv[0], 1);

    //test_sdf(cpol);
    //test_intersect(cpol);
    test_bbox(cpol);

    return 0 ; 
}

