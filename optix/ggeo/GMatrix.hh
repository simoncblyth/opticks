#ifndef GMATRIX_H
#define GMATRIX_H

#include <stdio.h>


template<typename T>
class GMatrix
{
   public:
       GMatrix();
       GMatrix(const GMatrix& m);
       GMatrix(
          T _a1, T _a2, T _a3, T _a4,
          T _b1, T _b2, T _b3, T _b4,
          T _c1, T _c2, T _c3, T _c4,
          T _d1, T _d2, T _d3, T _d4); 
 

       virtual ~GMatrix();

       void Dump(const char* msg="GMatrix::Dump");

       GMatrix& operator *= (const GMatrix& m); 
       GMatrix  operator *  (const GMatrix& m) const;

   public:
       T a1, a2, a3, a4 ; 
       T b1, b2, b3, b4 ; 
       T c1, c2, c3, c4 ; 
       T d1, d2, d3, d4 ; 


};


template<typename T>
GMatrix<T>::GMatrix() :
      a1(1.f), a2(0.f), a3(0.f), a4(0.f), 
      b1(0.f), b2(1.f), b3(0.f), b4(0.f), 
      c1(0.f), c2(0.f), c3(1.f), c4(0.f), 
      d1(0.f), d2(0.f), d3(0.f), d4(1.f) 
{   
}

template<typename T>
GMatrix<T>::GMatrix(const GMatrix& m) : 
      a1(m.a1), a2(m.a2), a3(m.a3), a4(m.a4), 
      b1(m.b1), b2(m.b2), b3(m.b3), b4(m.b4), 
      c1(m.c1), c2(m.c2), c3(m.c3), c4(m.c4), 
      d1(m.d1), d2(m.d2), d3(m.d3), d4(m.d4) 
{ 
}

template<typename T>
GMatrix<T>::GMatrix(
          T _a1, T _a2, T _a3, T _a4,
          T _b1, T _b2, T _b3, T _b4,
          T _c1, T _c2, T _c3, T _c4,
          T _d1, T _d2, T _d3, T _d4
                ) :
      a1(_a1), a2(_a2), a3(_a3), a4(_a4), 
      b1(_b1), b2(_b2), b3(_b3), b4(_b4), 
      c1(_c1), c2(_c2), c3(_c3), c4(_c4), 
      d1(_d1), d2(_d2), d3(_d3), d4(_d4) 
{   
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
void GMatrix<T>::Dump(const char* msg)
{
    printf("%s\n", msg);

    printf(" a %10.3f %10.3f %10.3f %10.3f \n", a1, a2, a3, a4 );
    printf(" b %10.3f %10.3f %10.3f %10.3f \n", b1, b2, b3, b4 );
    printf(" c %10.3f %10.3f %10.3f %10.3f \n", c1, c2, c3, c4 );
    printf(" d %10.3f %10.3f %10.3f %10.3f \n", d1, d2, d3, d4 );


}
 


typedef GMatrix<float>  GMatrixF ;

#endif

