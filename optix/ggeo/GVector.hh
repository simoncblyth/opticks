#ifndef GVECTOR_H
#define GVECTOR_H

#include "GMatrix.hh"

struct gfloat3 
{
    gfloat3() : x(0.f), y(0.f), z(0.f) {} ;
    gfloat3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {} ;

    gfloat3& operator *= (const GMatrixF& m)
    {
       float _x, _y, _z ;   
       _x = m.a1 * x + m.a2 * y + m.a3 * z + m.a4;
       _y = m.b1 * x + m.b2 * y + m.b3 * z + m.b4;
       _z = m.c1 * x + m.c2 * y + m.c3 * z + m.c4;

       x = _x ; 
       y = _y ; 
       z = _z ; 

       return *this ;
    }

    void Summary(const char* msg)
    {
        printf("%s %10.3f %10.3f %10.3f\n", msg, x, y, z);
    }


    float x,y,z ;
};

struct guint3 
{
    unsigned int x,y,z ;
};


#endif
