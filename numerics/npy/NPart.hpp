#pragma once

#include "NQuad.hpp"
#include <cassert>

enum {
      ZERO, 
      SPHERE, 
      TUBS, 
      BOX 
      };  

enum { PARAM_J  = 0, PARAM_K  = 0 };

enum { INDEX_J  = 1, INDEX_K  = 1 };
enum { BOUNDARY_J = 1, BOUNDARY_K = 2 };
enum { FLAGS_J  = 1, FLAGS_K  = 3 };

enum { BBMIN_J = 2,     BBMIN_K = 0 };
enum { TYPECODE_J  = 2, TYPECODE_K  = 3 };

enum { BBMAX_J = 3,     BBMAX_K = 0 };
enum { NODEINDEX_J = 3, NODEINDEX_K = 3 };



struct npart 
{
    nquad q0 ; 
    nquad q1 ; 
    nquad q2 ; 
    nquad q3 ; 

    void zero();
    void dump(const char* msg);
    void setTypeCode(unsigned int typecode);
};


inline void npart::zero()
{
    q0.u = {0,0,0,0} ;
    q1.u = {0,0,0,0} ;
    q2.u = {0,0,0,0} ;
    q3.u = {0,0,0,0} ;
}


inline void npart::setTypeCode(unsigned int typecode)
{
    assert( TYPECODE_J == 2 && TYPECODE_K == W );
    q2.u.w = typecode ; 
}
