#pragma once

#include <string>
#include "GBuffer.hh"

// TODO: replace with glm ?

template<typename T>
class GMatrix : public GBuffer 
{
   public:
       GMatrix(T _s);                   // homogenous scaling  matrix
       GMatrix(T _x, T _y, T _z, T _s=1.0f); // homogenous translate then scale matrix (ie translation not scaled)
       GMatrix();
       GMatrix(const GMatrix& m);
       GMatrix(
          T _a1, T _a2, T _a3, T _a4,
          T _b1, T _b2, T _b3, T _b4,
          T _c1, T _c2, T _c3, T _c4,
          T _d1, T _d2, T _d3, T _d4); 
 

       virtual ~GMatrix();

       void Summary(const char* msg="GMatrix::Summary");

       GMatrix& operator *= (const GMatrix& m); 
       GMatrix  operator *  (const GMatrix& m) const;

       T largestDiff(const GMatrix& m);

       void copyTo(T* buf);
       void* getPointer();  // override GBuffer

       std::string digest();
       std::string brief(unsigned int w=11);

   public:
       T a1, a2, a3, a4 ; 
       T b1, b2, b3, b4 ; 
       T c1, c2, c3, c4 ; 
       T d1, d2, d3, d4 ; 


};



typedef GMatrix<float>  GMatrixF ;


