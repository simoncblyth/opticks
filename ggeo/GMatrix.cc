
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstdio>

// sysrap-
#include "SDigest.hh"

// ggeo-
#include "GMatrix.hh"






template<typename T>
T GMatrix<T>::largestDiff(const GMatrix& m)
{
    T d(0);
    T delta ; 

    delta = fabs(m.a1 - a1) ; if(delta > d) d = delta ; 
    delta = fabs(m.a2 - a2) ; if(delta > d) d = delta ; 
    delta = fabs(m.a3 - a3) ; if(delta > d) d = delta ; 
    delta = fabs(m.a4 - a4) ; if(delta > d) d = delta ; 

    delta = fabs(m.b1 - b1) ; if(delta > d) d = delta ; 
    delta = fabs(m.b2 - b2) ; if(delta > d) d = delta ; 
    delta = fabs(m.b3 - b3) ; if(delta > d) d = delta ; 
    delta = fabs(m.b4 - b4) ; if(delta > d) d = delta ; 

    delta = fabs(m.c1 - c1) ; if(delta > d) d = delta ; 
    delta = fabs(m.c2 - c2) ; if(delta > d) d = delta ; 
    delta = fabs(m.c3 - c3) ; if(delta > d) d = delta ; 
    delta = fabs(m.c4 - c4) ; if(delta > d) d = delta ; 

    delta = fabs(m.d1 - d1) ; if(delta > d) d = delta ; 
    delta = fabs(m.d2 - d2) ; if(delta > d) d = delta ; 
    delta = fabs(m.d3 - d3) ; if(delta > d) d = delta ; 
    delta = fabs(m.d4 - d4) ; if(delta > d) d = delta ; 

    return d ; 
}


template<typename T>
GMatrix<T>::GMatrix() :
      GBuffer( sizeof(T)*16, NULL, sizeof(T), 1 , "GMatrix0"),
      a1(1.f), a2(0.f), a3(0.f), a4(0.f), 
      b1(0.f), b2(1.f), b3(0.f), b4(0.f), 
      c1(0.f), c2(0.f), c3(1.f), c4(0.f), 
      d1(0.f), d2(0.f), d3(0.f), d4(1.f) 
{   
}


template<typename T>
GMatrix<T>::GMatrix(T _s) :
      GBuffer( sizeof(T)*16, NULL, sizeof(T), 1 , "GMatrix1"),
      a1( _s), a2(0.f), a3(0.f), a4(0.f), 
      b1(0.f), b2( _s), b3(0.f), b4(0.f), 
      c1(0.f), c2(0.f), c3( _s), c4(0.f), 
      d1(0.f), d2(0.f), d3(0.f), d4(1.f) 
{   
}

template<typename T>
GMatrix<T>::GMatrix(T _x, T _y, T _z, T _s) :
      GBuffer( sizeof(T)*16, NULL, sizeof(T), 1 , "GMatrix2"),
      a1( _s), a2(0.f), a3(0.f), a4(_x), 
      b1(0.f), b2( _s), b3(0.f), b4(_y), 
      c1(0.f), c2(0.f), c3( _s), c4(_z), 
      d1(0.f), d2(0.f), d3(0.f), d4(1.f) 
{   
}

template<typename T>
GMatrix<T>::GMatrix(const GMatrix& m) : 
      GBuffer( sizeof(T)*16, NULL, sizeof(T), 1 , "GMatrix3"),
      a1(m.a1), a2(m.a2), a3(m.a3), a4(m.a4), 
      b1(m.b1), b2(m.b2), b3(m.b3), b4(m.b4), 
      c1(m.c1), c2(m.c2), c3(m.c3), c4(m.c4), 
      d1(m.d1), d2(m.d2), d3(m.d3), d4(m.d4) 
{ 
}


template<typename T>
GMatrix<T>::GMatrix(T* buf) :
      GBuffer( sizeof(T)*16, NULL, sizeof(T), 1 , "GMatrix4"),

      a1(buf[0]),
      a2(buf[4]),
      a3(buf[8]),
      a4(buf[12]),

      b1(buf[1]),
      b2(buf[5]),
      b3(buf[9]),
      b4(buf[13]),

      c1(buf[2]),
      c2(buf[6]),
      c3(buf[10]),
      c4(buf[14]),

      d1(buf[3]),
      d2(buf[7]),
      d3(buf[11]),
      d4(buf[15])
{
}




template<typename T>
GMatrix<T>::GMatrix(
          T _a1, T _a2, T _a3, T _a4,
          T _b1, T _b2, T _b3, T _b4,
          T _c1, T _c2, T _c3, T _c4,
          T _d1, T _d2, T _d3, T _d4
                ) :
      GBuffer( sizeof(T)*16, NULL, sizeof(T), 1 , "GMatrix5"),
      a1(_a1), a2(_a2), a3(_a3), a4(_a4), 
      b1(_b1), b2(_b2), b3(_b3), b4(_b4), 
      c1(_c1), c2(_c2), c3(_c3), c4(_c4), 
      d1(_d1), d2(_d2), d3(_d3), d4(_d4) 
{   
}


template <typename T>
void GMatrix<T>::copyTo(T* buf)
{
    buf[0] = a1 ;
    buf[1] = b1 ;
    buf[2] = c1 ;
    buf[3] = d1 ;

    buf[4] = a2 ;
    buf[5] = b2 ;
    buf[6] = c2 ;
    buf[7] = d2 ;

    buf[8] = a3 ;
    buf[9] = b3 ;
    buf[10] = c3 ;
    buf[11] = d3 ;  

    buf[12] = a4 ;  // 13th t_x
    buf[13] = b4 ;  // 14th t_y
    buf[14] = c4 ;  // 15th t_z 
    buf[15] = d4 ;  // 16th 1.0
}


template <typename T>
bool GMatrix<T>::isIdentity()
{
    T one(1.) ; 
    T zero(0.) ; 
    return
         a1 == one  && a2 == zero  && a3 == zero && a4 == zero  &&
         b1 == zero && b2 == one   && b3 == zero && b4 == zero  &&
         c1 == zero && c2 == zero  && c3 == one  && c4 == zero  &&
         d1 == zero && d2 == zero  && d3 == zero && d4 == one   ;

}

template <typename T>
void* GMatrix<T>::getPointer()
{

/*

* https://www.opengl.org/archives/resources/faq/technical/transformations.htm

9.005 Are OpenGL matrices column-major or row-major?

For programming purposes, OpenGL matrices are 16-value arrays with base vectors
laid out contiguously in memory. The translation components occupy the 13th,
14th, and 15th elements of the 16-element matrix, where indices are numbered
from 1 to 16 as described in section 2.11.2 of the OpenGL 2.1 Specification.

Column-major versus row-major is purely a notational convention. Note that
post-multiplying with column-major matrices produces the same result as
pre-multiplying with row-major matrices. The OpenGL Specification and the
OpenGL Reference Manual both use column-major notation. You can use any
notation, as long as it's clearly stated.

Sadly, the use of column-major format in the spec and blue book has resulted in
endless confusion in the OpenGL programming community. Column-major notation
suggests that matrices are not laid out in memory as a programmer would expect.

*/
    if( !m_pointer )
    {
        T* buf = new T[16];
        copyTo(buf);
        m_pointer = buf ;
    }
    return m_pointer ; 
}



template <typename T>
inline GMatrix<T>& GMatrix<T>::operator *= (const GMatrix<T>& m)
{
    *this = GMatrix<T>(
        m.a1 * a1 + m.b1 * a2 + m.c1 * a3 + m.d1 * a4,
        m.a2 * a1 + m.b2 * a2 + m.c2 * a3 + m.d2 * a4,
        m.a3 * a1 + m.b3 * a2 + m.c3 * a3 + m.d3 * a4,
        m.a4 * a1 + m.b4 * a2 + m.c4 * a3 + m.d4 * a4,
        m.a1 * b1 + m.b1 * b2 + m.c1 * b3 + m.d1 * b4,
        m.a2 * b1 + m.b2 * b2 + m.c2 * b3 + m.d2 * b4,
        m.a3 * b1 + m.b3 * b2 + m.c3 * b3 + m.d3 * b4,
        m.a4 * b1 + m.b4 * b2 + m.c4 * b3 + m.d4 * b4,
        m.a1 * c1 + m.b1 * c2 + m.c1 * c3 + m.d1 * c4,
        m.a2 * c1 + m.b2 * c2 + m.c2 * c3 + m.d2 * c4,
        m.a3 * c1 + m.b3 * c2 + m.c3 * c3 + m.d3 * c4,
        m.a4 * c1 + m.b4 * c2 + m.c4 * c3 + m.d4 * c4,
        m.a1 * d1 + m.b1 * d2 + m.c1 * d3 + m.d1 * d4,
        m.a2 * d1 + m.b2 * d2 + m.c2 * d3 + m.d2 * d4,
        m.a3 * d1 + m.b3 * d2 + m.c3 * d3 + m.d3 * d4,
        m.a4 * d1 + m.b4 * d2 + m.c4 * d3 + m.d4 * d4);
    return *this;
}


template <typename T>
inline GMatrix<T> GMatrix<T>::operator* (const GMatrix<T>& m) const
{
    GMatrix<T> temp(*this);
    temp *= m;
    return temp;
}



template<typename T>
GMatrix<T>::~GMatrix()
{
}
 
template<typename T>
void GMatrix<T>::Summary(const char* msg)
{
    printf("%s\n", msg);

    printf(" a %10.3f %10.3f %10.3f %10.3f \n", a1, a2, a3, a4 );
    printf(" b %10.3f %10.3f %10.3f %10.3f \n", b1, b2, b3, b4 );
    printf(" c %10.3f %10.3f %10.3f %10.3f \n", c1, c2, c3, c4 );
    printf(" d %10.3f %10.3f %10.3f %10.3f \n", d1, d2, d3, d4 );


}

template<typename T>
std::string GMatrix<T>::brief(unsigned int w)
{
    std::stringstream ss ;

    ss << std::dec << std::fixed << std::setprecision(3) ; 

    ss << std::setw(w) << a1 << " " ;  
    ss << std::setw(w) << a2 << " " ;  
    ss << std::setw(w) << a3 << " " ;  
    ss << std::setw(w) << a4 << "," ;  

    ss << std::setw(w) << b1 << " " ;  
    ss << std::setw(w) << b2 << " " ;  
    ss << std::setw(w) << b3 << " " ;  
    ss << std::setw(w) << b4 << "," ;  

    ss << std::setw(w) << c1 << " " ;  
    ss << std::setw(w) << c2 << " " ;  
    ss << std::setw(w) << c3 << " " ;  
    ss << std::setw(w) << c4 << "," ;  

    ss << std::setw(w) << d1 << " " ;  
    ss << std::setw(w) << d2 << " " ;  
    ss << std::setw(w) << d3 << " " ;  
    ss << std::setw(w) << d4 << "," ;  


    return ss.str();
}

template<typename T>
std::string GMatrix<T>::digest()
{
    SDigest dig ;
    dig.update( (char*)getPointer(), sizeof(T)*16 );  
    return dig.finalize();
}





template class GMatrix<float>;
template class GMatrix<double>;


