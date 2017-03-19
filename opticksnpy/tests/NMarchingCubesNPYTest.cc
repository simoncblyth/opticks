#include <iostream>
#include <limits>
#include <algorithm>

#include "NGLM.hpp"
#include "NPY.hpp"
#include "GLMFormat.hpp"

#include "NMarchingCubesNPY.hpp"
#include "NTrianglesNPY.hpp"
#include "NSphere.hpp"

#include "PLOG.hh"
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"


void test_csgsdf()
{
    nsphere a = make_nsphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_nsphere(0.f,0.f, 50.f,100.f);

    nunion u = make_nunion( &a, &b );
    nintersection i = make_nintersection( &a, &b ); 
    ndifference d1 = make_ndifference( &a, &b ); 
    ndifference d2 = make_ndifference( &b, &a ); 
    nunion u2 = make_nunion( &d1, &d2 );

    typedef std::vector<nnode*> VN ;

    VN nodes ; 
    nodes.push_back( (nnode*)&a );
    nodes.push_back( (nnode*)&b );
    nodes.push_back( (nnode*)&u );
    nodes.push_back( (nnode*)&i );
    nodes.push_back( (nnode*)&d1 );
    nodes.push_back( (nnode*)&d2 );
    nodes.push_back( (nnode*)&u2 );

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
        std::cout << " z  " << std::setw(10) << z 
             << " a  " << std::setw(10) << std::fixed << std::setprecision(2) << a(x,y,z) 
             << " b  " << std::setw(10) << std::fixed << std::setprecision(2) << b(x,y,z) 
             << " u  " << std::setw(10) << std::fixed << std::setprecision(2) << u(x,y,z) 
             << " i  " << std::setw(10) << std::fixed << std::setprecision(2) << i(x,y,z) 
             << " d1 " << std::setw(10) << std::fixed << std::setprecision(2) << d1(x,y,z) 
             << " d2 " << std::setw(10) << std::fixed << std::setprecision(2) << d2(x,y,z) 
             << " u2 " << std::setw(10) << std::fixed << std::setprecision(2) << u2(x,y,z) 
             << std::endl 
             ; 

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
    nsphere a = make_nsphere(0.f,0.f,0.f,100.f);

    NTrianglesNPY* tris0 = mcu(&a);
    assert(tris0);
    if(dump) tris0->getBuffer()->dump("test_sphere");

    NTrianglesNPY* tris1 = mcu((nnode*)&a);
    assert(tris1);
    if(dump) tris1->getBuffer()->dump("test_sphere (nnode*)");
}

void test_union(NMarchingCubesNPY& mcu, bool dump)
{
    nsphere a = make_nsphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_nsphere(0.f,0.f, 50.f,100.f);
    nunion u = make_nunion( &a, &b ); 

    NTrianglesNPY* tris = mcu(&u);
    assert(tris);

    if(dump) tris->getBuffer()->dump("test_union");
}

void test_intersection(NMarchingCubesNPY& mcu, bool dump)
{
    nsphere a = make_nsphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_nsphere(0.f,0.f, 50.f,100.f);

    nintersection i = make_nintersection( &a , &b ); 

    NTrianglesNPY* tris = mcu(&i);
    assert(tris);

    if(dump) tris->getBuffer()->dump("test_intersection");
}


void test_difference(NMarchingCubesNPY& mcu, bool dump)
{
    nsphere a = make_nsphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_nsphere(0.f,0.f, 50.f,100.f);

    ndifference d1 = make_ndifference( &a, &b ); 
    ndifference d2 = make_ndifference( &b, &a ); 

    NTrianglesNPY* tris1 = mcu(&d1);
    assert(tris1);
    if(dump) tris1->getBuffer()->dump("test_difference d1");

    NTrianglesNPY* tris2 = mcu(&d2);
    assert(tris2);
    if(dump) tris2->getBuffer()->dump("test_difference d2");
}





void test_generic(NMarchingCubesNPY& mcu)
{
    nsphere a = make_nsphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_nsphere(0.f,0.f, 50.f,100.f);
    nunion u = make_nunion( &a, &b );
    nintersection i = make_nintersection( &a, &b ); 
    ndifference d1 = make_ndifference( &a, &b ); 
    ndifference d2 = make_ndifference( &b, &a ); 
    nunion u2 = make_nunion( &d1, &d2 );

    typedef std::vector<nnode*> VN ;

    VN nodes ; 
    nodes.push_back( (nnode*)&a );
    nodes.push_back( (nnode*)&b );
    nodes.push_back( (nnode*)&u );
    nodes.push_back( (nnode*)&i );
    nodes.push_back( (nnode*)&d1 );
    nodes.push_back( (nnode*)&d2 );
    nodes.push_back( (nnode*)&u2 );

    for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
    {
        nnode* n = *it ; 
        OpticksCSG_t type = n->type ; 
        const char* name = n->csgname();

        assert( type > 0 && name != NULL );

        NTrianglesNPY* tris = mcu(n);
        unsigned ntris = tris ? tris->getNumTriangles() : 0 ; 

        unsigned mxd = n->maxdepth();


        std::cout 
                  << " type: " << std::setw(3) << type 
                  << " name: " << std::setw(15) << ( name ? name : "-" ) 
                  << " sdf(0,0,0): " << std::setw(10) << std::fixed << std::setprecision(2) << (*n)(0,0,0)
                  << " ntris " << ntris 
                  << " maxdepth " << mxd 
                  << std::endl 
                  ; 
    }
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    test_csgsdf();

    nuvec3 param = {10,10,10};
    NMarchingCubesNPY mcu(param);

    bool dump = false ; 

    test_union(mcu, dump);
    test_intersection(mcu, dump);
    test_difference(mcu, dump);
    test_sphere(mcu, dump);
    test_generic(mcu);

    return 0 ; 
}
