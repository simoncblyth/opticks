#include <iostream>
#include <limits>
#include <algorithm>

#include "NGLM.hpp"
#include "NPY.hpp"
#include "GLMFormat.hpp"

#include "NMarchingCubesNPY.hpp"
#include "NTrianglesNPY.hpp"
#include "NSphere.hpp"
#include "NBox.hpp"
#include "NNodeSample.hpp"

#include "OPTICKS_LOG.hh"


void test_csgsdf()
{
    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    NNodeSample::Tests(nodes);

    for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
    {
        nnode* n = *it ; 
        OpticksCSG_t type = n->type ; 
        const char* name = n->csgname();
        std::cout 
                  << " type: " << std::setw(3) << type 
                  << " name: " << ( name ? name : "-" ) 
                  << " sdf(0,0,0): " << std::setw(10) << std::fixed << std::setprecision(2) << (*n)(0,0,0)
                  << std::endl 
                  ; 

    }

    float x = 0.f ; 
    float y = 0.f ; 
    float z = 0.f ; 

    for(int iz=-200 ; iz <= 200 ; iz+= 10, z=iz ) 
    {
        std::cout << " z  " << std::setw(10) << z  ;
        for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
        {
             nnode* n = *it ; 
             const char* name = n->csgname();
             printf(" %.3s %10.2f", ( name ? name : "-" ), (*n)(x,y,z) ); 
        }
        std::cout << std::endl ; 
    }
}


void test_sphere(NMarchingCubesNPY& mcu, bool dump)
{
    nsphere a = make_sphere(0.f,0.f,0.f,100.f);

    NTrianglesNPY* tris0 = mcu(&a);
    assert(tris0);
    if(dump) tris0->getTris()->dump("test_sphere");

    NTrianglesNPY* tris1 = mcu((nnode*)&a);
    assert(tris1);
    if(dump) tris1->getTris()->dump("test_sphere (nnode*)");
}


void test_box(NMarchingCubesNPY& mcu, bool dump)
{
    nbox a = make_box(0.f,0.f,0.f,1000.f);

    NTrianglesNPY* tris0 = mcu(&a);
    assert(tris0);
    if(dump) tris0->getTris()->dump("test_box");

    NTrianglesNPY* tris1 = mcu((nnode*)&a);
    assert(tris1);
    if(dump) tris1->getTris()->dump("test_box (nnode*)");
}


void test_union(NMarchingCubesNPY& mcu, bool dump)
{
    nsphere a = make_sphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_sphere(0.f,0.f, 50.f,100.f);
    nunion u = nunion::make_union( &a, &b ); 

    NTrianglesNPY* tris = mcu(&u);
    assert(tris);

    if(dump) tris->getTris()->dump("test_union");
}

void test_intersection(NMarchingCubesNPY& mcu, bool dump)
{
    nsphere a = make_sphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_sphere(0.f,0.f, 50.f,100.f);

    nintersection i = nintersection::make_intersection( &a , &b ); 

    NTrianglesNPY* tris = mcu(&i);
    assert(tris);

    if(dump) tris->getTris()->dump("test_intersection");
}


void test_difference(NMarchingCubesNPY& mcu, bool dump)
{
    nsphere a = make_sphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_sphere(0.f,0.f, 50.f,100.f);

    ndifference d1 = ndifference::make_difference( &a, &b ); 
    ndifference d2 = ndifference::make_difference( &b, &a ); 

    NTrianglesNPY* tris1 = mcu(&d1);
    assert(tris1);
    if(dump) tris1->getTris()->dump("test_difference d1");

    NTrianglesNPY* tris2 = mcu(&d2);
    assert(tris2);
    if(dump) tris2->getTris()->dump("test_difference d2");
}


void test_generic(NMarchingCubesNPY& mcu)
{
    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    NNodeSample::Tests(nodes);

    for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
    {
        nnode* n = *it ; 
        OpticksCSG_t type = n->type ; 
        const char* name = n->csgname();

        assert( type > 0 && name != NULL );

        NTrianglesNPY* tris = mcu(n);   // <--- run marching cubes 

        unsigned ntris = tris ? tris->getNumTriangles() : 0 ; 
        unsigned mxd = n->maxdepth();

        NPY<float>* buf = tris->getTris();

        nbbox* bb = tris->findBBox(); 

        std::cout 
                  << " type: " << std::setw(3) << type 
                  << " name: " << std::setw(15) << ( name ? name : "-" ) 
                  << " sdf(0,0,0): " << std::setw(10) << std::fixed << std::setprecision(2) << (*n)(0,0,0)
                  << " ntris " << ntris 
                  << " maxdepth " << mxd 
                  << " sh " << buf->getShapeString()
                  << " " << ( bb ? bb->desc() : "bb:NULL" )
                  << std::endl 
                  ; 
    }
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    NMarchingCubesNPY mcu(15);

    /*
    bool dump = false ; 
    test_csgsdf();
    test_union(mcu, dump);
    test_intersection(mcu, dump);
    test_difference(mcu, dump);
    test_sphere(mcu, dump);
    test_box(mcu,dump);
    */

    test_generic(mcu);

    NMarchingCubesNPY mcu_10(10);
    test_generic(mcu_10);

    return 0 ; 
}
