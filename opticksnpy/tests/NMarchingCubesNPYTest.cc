#include <iostream>
#include <limits>
#include <algorithm>

#include "BRAP_LOG.hh"
#include "NGLM.hpp"
#include "NPY.hpp"
#include "GLMFormat.hpp"

#include "NMarchingCubesNPY.hpp"
#include "NTrianglesNPY.hpp"
#include "PLOG.hh"

#include "NSphere.hpp"



void test_csgsdf()
{
    nsphere a = make_nsphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_nsphere(0.f,0.f, 50.f,100.f);

    nunion u ; 
    u.left = &a  ;
    u.right = &b ;

    nintersection i ; 
    i.left = &a  ;
    i.right = &b ;

    ndifference d1 ; 
    d1.left = &a  ;
    d1.right = &b ;

    ndifference d2 ; 
    d2.left = &b  ;
    d2.right = &a ;

    nunion u2 ; 
    u2.left = &d1 ;
    u2.right = &d2 ;


    float x = 0.f ; 
    float y = 0.f ; 
    float z = 0.f ; 

    for(int iz=-200 ; iz <= 200 ; iz+= 10, z=iz ) 
        std::cout 
             << " z  " << std::setw(10) << z 
             << " a  " << std::setw(10) << std::fixed << std::setprecision(2) << a(x,y,z) 
             << " b  " << std::setw(10) << std::fixed << std::setprecision(2) << b(x,y,z) 
             << " u  " << std::setw(10) << std::fixed << std::setprecision(2) << u(x,y,z) 
             << " i  " << std::setw(10) << std::fixed << std::setprecision(2) << i(x,y,z) 
             << " d1 " << std::setw(10) << std::fixed << std::setprecision(2) << d1(x,y,z) 
             << " d2 " << std::setw(10) << std::fixed << std::setprecision(2) << d2(x,y,z) 
             << " u2 " << std::setw(10) << std::fixed << std::setprecision(2) << u2(x,y,z) 
             << std::endl 
             ; 

}



void test_sphere()
{
    nsphere a = make_nsphere(0.f,0.f,0.f,100.f);

    const glm::uvec3 param(10,10,10);
    const glm::vec3 low( -100.,-100.,-100.); 
    const glm::vec3 high( 100., 100., 100.); 

    //NMarchingCubesNPY<nsdf> mc;
    //NMarchingCubesNPY<nsphere> mc;
    NMarchingCubesNPY mc;

    

    NTrianglesNPY* tris = mc.march( nsphere::operator() , param, low, high);
    assert(tris);

    tris->getBuffer()->dump("test_sphere");
}


void test_union()
{
    nsphere a = make_nsphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_nsphere(0.f,0.f, 50.f,100.f);

    nunion u ; 
    u.left = &a  ;
    u.right = &b ;

    const glm::uvec3 param(10,10,10);
    const glm::vec3 low( -150.,-150.,-150.); 
    const glm::vec3 high( 150., 150., 150.); 

    //NMarchingCubesNPY<nsdf> mc;
    NMarchingCubesNPY<nunion> mc;
    NTrianglesNPY* tris = mc.march(u, param, low, high);
    assert(tris);

    tris->getBuffer()->dump("test_union");
}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_csgsdf();
    test_sphere();
    test_union();


    return 0 ; 
}
