#include "NCSG.hpp"
#include "NConvexPolyhedron.hpp"
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
    for(float v=-400.f ; v <= 400.f ; v+= 100.f )
    {
        std::cout 
            << " x  " << std::setw(10) << v
            << " sd:  " << std::setw(10) << (*cpol)(v,0,0)
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
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 


    nnode* root = nnode::load(argc > 1 ?  argv[1] : "$TMP/tboolean-trapezoid--/1" , 1 );
    assert(root->type == CSG_CONVEXPOLYHEDRON );
    nconvexpolyhedron* cpol = (nconvexpolyhedron*)root  ;
    cpol->pdump(argv[0], 1);

    test_sdf(cpol);
    //test_intersect(cpol);

    return 0 ; 
}

