#pragma once

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


#include "NQuad.hpp"

struct npart 
{
    void dump(const char* msg);

    npart();
    npart(const npart& other);
    ~npart();

    void setTypeCode(unsigned int typecode);

    nquad q0 ; 
    nquad q1 ; 
    nquad q2 ; 
    nquad q3 ; 

};


inline npart::npart() : q0(), q1(), q2(), q3() {}
inline npart::~npart(){}
inline npart::npart(const npart& other) : q0(other.q0), q1(other.q1), q2(other.q2), q3(other.q3) {}


inline void npart::setTypeCode(unsigned int typecode)
{
    q2.u[TYPECODE_K] = typecode ; 
}
