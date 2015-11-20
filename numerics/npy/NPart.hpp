#pragma once

#include "NQuad.hpp"

enum {
      ZERO, 
      SPHERE, 
      TUBS, 
      BOX 
      };  


// these internals exposed, 
// as still being used at higher level in ggeo-/GParts

enum { PARAM_J  = 0, PARAM_K  = 0 };

enum { INDEX_J  = 1, INDEX_K  = 1 };
enum { BOUNDARY_J = 1, BOUNDARY_K = 2 };
enum { FLAGS_J  = 1, FLAGS_K  = 3 };

enum { BBMIN_J = 2,     BBMIN_K = 0 };
enum { TYPECODE_J  = 2, TYPECODE_K  = 3 };

enum { BBMAX_J = 3,     BBMAX_K = 0 };
enum { NODEINDEX_J = 3, NODEINDEX_K = 3 };


struct nbbox ; 

struct npart 
{
    nquad q0 ; 
    nquad q1 ; 
    nquad q2 ; 
    nquad q3 ; 

    void zero();
    void dump(const char* msg);
    void setTypeCode(unsigned int typecode);
    void setBBox(const nbbox& bb);
    void setParam(const nvec4& param);
};







