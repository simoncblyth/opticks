#include "NPart.hpp"
#include "NPlane.hpp"

#include <cstdio>
#include <cassert>

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

void npart::setTypeCode(unsigned int typecode)
{
    //assert( TYPECODE_J == 2 && TYPECODE_K == W );
    q2.u.w = typecode ; 
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

