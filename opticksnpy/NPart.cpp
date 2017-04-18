#include "NPart.hpp"
#include "NBBox.hpp"
#include "NPlane.hpp"

#include <cstdio>
#include <cassert>

unsigned npart::VERSION = 1 ;   // could become an enumerated LAYOUT bitfield if need more granularity 

void npart::setTypeCode(OpticksCSG_t typecode)
{
    assert( TYPECODE_J == 2 && TYPECODE_K == 3 );
    q2.u.w = typecode ; 
}

void npart::setGTransform(unsigned gtransform_idx)
{
    assert(VERSION == 1u);
    assert( GTRANSFORM_J == 3 && GTRANSFORM_K == 0 );
    q3.u.x = gtransform_idx ; 
}



// thought not used, but they are for in memory npart 
// the implicit left/rigth from the index is for the serialization

void npart::setLeft(unsigned left)
{
    //assert( LEFT_J == 0 && LEFT_K == 3 );  // note same location as primitive param.w
    qx.u.w = left ; 
}
void npart::setRight(unsigned right)
{
    //assert( RIGHT_J == 1 && RIGHT_K == 3 );
    qx.u.w = right ; 
}
unsigned npart::getLeft()
{
    return qx.u.w ; 
}
unsigned npart::getRight()
{
    return qx.u.w ; 
}







OpticksCSG_t npart::getTypeCode()
{
    return (OpticksCSG_t)q2.u.w ; 
}

bool npart::isPrimitive()
{
    return CSGIsPrimitive(getTypeCode());
}

void npart::traverse( npart* tree, unsigned numNodes, unsigned i )
{
    assert( i >= 0 && i < numNodes );

    npart* node = tree + i ; 
    bool prim = node->isPrimitive();
    printf("npart::traverse numNodes %u i %u prim %d \n", numNodes, i, prim) ;

    if(prim)
    {
        node->dump("traverse primitive");
    }
    else
    {
        unsigned l = node->getLeft();
        unsigned r = node->getRight();
        printf("npart::traverse (non-prim) numNodes %u i %u l %u r %u \n", numNodes, i, l, r) ;

        assert(l > 0 and r > 0);
         
        traverse( tree, numNodes, l ); 
        traverse( tree, numNodes, r ); 
    }
}



// primitive parts

void npart::dump(const char* msg)
{
    printf("%s\n", msg);
    q0.dump("q0");
    q1.dump("q1");
    q2.dump("q2");
    q3.dump("q3");

    qx.dump("qx");
}

void npart::zero()
{
    q0.u = {0,0,0,0} ;
    q1.u = {0,0,0,0} ;
    q2.u = {0,0,0,0} ;
    q3.u = {0,0,0,0} ;

    qx.u = {0,0,0,0} ;
}
void npart::setParam(const nquad& q0_)
{
    assert( PARAM_J == 0 && PARAM_K == 0 );
    q0 = q0_;
}
void npart::setParam1(const nquad& q1_)
{
    assert( PARAM1_J == 1 && PARAM_K == 0 );
    q1 = q1_;
}
void npart::setParam2(const nquad& q2_)
{
    q2 = q2_;
}
void npart::setParam3(const nquad& q3_)
{
    q3 = q3_;
}




void npart::setParam(float x, float y, float z, float w)
{
    nquad param ;
    param.f = {x,y,z,w} ;
    setParam( param );
}
void npart::setParam1(float x, float y, float z, float w)
{
    nquad param1 ;
    param1.f = {x,y,z,w} ;
    setParam1( param1 );
}






void npart::setBBox(const nbbox& bb)
{
    assert(VERSION == 0u);  

    assert( BBMIN_J == 2 && BBMIN_K == 0 );
    q2.f.x = bb.min.x ; 
    q2.f.y = bb.min.y ; 
    q2.f.z = bb.min.z ;
 
    assert( BBMAX_J == 3 && BBMAX_K == 0 );
    q3.f.x = bb.max.x ; 
    q3.f.y = bb.max.y ; 
    q3.f.z = bb.max.z ;
}

