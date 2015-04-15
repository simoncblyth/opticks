#pragma once

#include "NPY.hpp"
#include "float.h"

class VecNPY {
    public:
        VecNPY(NPY* npy, unsigned int j, unsigned int k) :
            m_data(npy->getBytes()),
         //   m_size(3),
            m_stride(npy->getNumBytes(1)),
            m_offset(npy->getByteIndex(0,j,k)),
            m_count(npy->getShape(0))
        {
        }

    public:
        void dump(const char* msg);

    private:
        void*        m_data   ;
        //unsigned int m_size   ;    // typically 1,2,3,4 
        unsigned int m_stride ;  
        unsigned int m_offset ;  
        unsigned int m_count ;  

};


void VecNPY::dump(const char* msg)
{
    float xx[4] = {-FLT_MAX, FLT_MAX, 0.f, 0.f};
    float yy[4] = {-FLT_MAX, FLT_MAX, 0.f, 0.f};
    float zz[4] = {-FLT_MAX, FLT_MAX, 0.f, 0.f};

    const char* fmt = "VecNPY::dump %5s %6u/%6u :  %15f %15f %15f \n";

    for(unsigned int i=0 ; i < m_count ; ++i )
    {   
        char* ptr = (char*)m_data + m_offset + i*m_stride  ;   
        float* f = (float*)ptr ; 
        float x(*(f+0));
        float y(*(f+1));
        float z(*(f+2));

        if( x>xx[0] ) xx[0] = x ;  
        if( x<xx[1] ) xx[1] = x ;  

        if( y>yy[0] ) yy[0] = y ;  
        if( y<yy[1] ) yy[1] = y ;  

        if( z>zz[0] ) zz[0] = z ;  
        if( z<zz[1] ) zz[1] = z ;  

        if(i < 5 || i > m_count - 5) printf(fmt, "", i,m_count, x, y, z);
    }

    xx[2] = xx[1] - xx[0] ;
    yy[2] = yy[1] - yy[0] ;
    zz[2] = zz[1] - zz[0] ;

    xx[3] = (xx[1] + xx[0])/2.f ;
    yy[3] = (yy[1] + yy[0])/2.f ;
    zz[3] = (zz[1] + zz[0])/2.f ;

    printf(fmt, "min", 0,0,xx[0],yy[0],zz[0]);
    printf(fmt, "max", 0,0,xx[1],yy[1],zz[1]);
    printf(fmt, "dif", 0,0,xx[2],yy[2],zz[2]);
    printf(fmt, "cen", 0,0,xx[3],yy[3],zz[3]);
}



