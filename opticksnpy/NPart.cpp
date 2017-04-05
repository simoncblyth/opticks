#include "NPart.hpp"
#include "NBBox.hpp"
#include "NPlane.hpp"

#include <cstdio>
#include <cassert>

void npart::setTypeCode(OpticksCSG_t typecode)
{
    assert( TYPECODE_J == 2 && TYPECODE_K == 3 );
    q2.u.w = typecode ; 
}


// thought not used, but they are for in memory npart 
// the implicit left/rigth from the index is for the serialization

void npart::setLeft(unsigned left)
{
    assert( LEFT_J == 0 && LEFT_K == 3 );
    q0.u.w = left ; 
}
void npart::setRight(unsigned right)
{
    assert( RIGHT_J == 1 && RIGHT_K == 3 );
    q1.u.w = right ; 
}
unsigned npart::getLeft()
{
    return q0.u.w ; 
}
unsigned npart::getRight()
{
    return q1.u.w ; 
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
}

void npart::zero()
{
    q0.u = {0,0,0,0} ;
    q1.u = {0,0,0,0} ;
    q2.u = {0,0,0,0} ;
    q3.u = {0,0,0,0} ;
}
void npart::setParam(const nvec4& param)
{
    assert( PARAM_J == 0 && PARAM_K == 0 );
    q0.f = param;
}
void npart::setParam(float x, float y, float z, float w)
{
    assert( PARAM_J == 0 && PARAM_K == 0 );
    q0.f.x = x;
    q0.f.y = y;
    q0.f.z = z;
    q0.f.w = w;
}



void npart::setBBox(const nbbox& bb)
{
    assert( BBMIN_J == 2 && BBMIN_K == 0 );
    q2.f.x = bb.min.x ; 
    q2.f.y = bb.min.y ; 
    q2.f.z = bb.min.z ;
 
    assert( BBMAX_J == 3 && BBMAX_K == 0 );
    q3.f.x = bb.max.x ; 
    q3.f.y = bb.max.y ; 
    q3.f.z = bb.max.z ;
}

