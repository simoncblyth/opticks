#pragma once

#include "NPart.h"
#include "NQuad.hpp"


struct nbbox ; 

struct NPY_API npart 
{
    nquad q0 ;  // x,y,z,w (float): param 
    nquad q1 ;  // x,y,z,w (uint) -/index/boundary/flags
    nquad q2 ;  // x,y,z (float):bbmin   w(uint):typecode  
    nquad q3 ;  // x,y,z (float):bbmax

    void zero();
    void dump(const char* msg);
    //void setTypeCode(NPart_t typecode);
    void setTypeCode(unsigned typecode);
    void setBBox(const nbbox& bb);
    void setParam(const nvec4& param);
};




