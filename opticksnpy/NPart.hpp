#pragma once

#include "NPart.h"
#include "NQuad.hpp"

// these internals exposed, 
// as still being used at higher level in ggeo-/GParts

enum { PARAM_J  = 0, PARAM_K  = 0 };       // q0.f.xyzw

enum { INDEX_J    = 1, INDEX_K    = 1 };   // q1.u.y
enum { BOUNDARY_J = 1, BOUNDARY_K = 2 };   // q1.u.z
enum { FLAGS_J    = 1, FLAGS_K    = 3 };   // q1.u.w

enum { BBMIN_J     = 2, BBMIN_K     = 0 };  // q2.f.xyz
enum { TYPECODE_J  = 2, TYPECODE_K  = 3 };  // q2.u.w

enum { BBMAX_J     = 3,     BBMAX_K = 0 };  // q3.f.xyz 
enum { NODEINDEX_J = 3, NODEINDEX_K = 3 };  // q3.u.w 


struct nbbox ; 

struct NPY_API npart 
{
    nquad q0 ;  // x,y,z,w (float): param 
    nquad q1 ;  // x,y,z,w (uint) -/index/boundary/flags
    nquad q2 ;  // x,y,z (float):bbmin   w(uint):typecode  
    nquad q3 ;  // x,y,z (float):bbmax

    void zero();
    void dump(const char* msg);
    void setTypeCode(NPart_t typecode);
    void setBBox(const nbbox& bb);
    void setParam(const nvec4& param);
};




