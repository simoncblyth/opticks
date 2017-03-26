#pragma once 
#include <cmath>
#include <cstdio>

// initially attempted to use glm vec constituents 
// but that runs into complications with unions and non-default ctor, dtor, copy-ctor
//  
// http://stackoverflow.com/questions/26572240/why-does-union-has-deleted-default-constructor-if-one-of-its-member-doesnt-have
// http://stackoverflow.com/questions/7299171/union-member-has-a-non-trivial-copy-constructor
// http://www.boost.org/doc/libs/1_47_0/doc/html/variant.html
//
//
// any ctors cause headaches for the union of them, 
// regarding deletion of default ctors ... SO LEAVE THESE WITHOUT CTORS
// AS THE POINT OF THEM IS TO PUT THEM INTO THE UNION
//


enum { X, Y, Z, W };


#include "NPY_API_EXPORT.hh"


template<typename T>
struct NPY_API ntvec4 {
   T x ;
   T y ;
   T z ;
   T w ;
};

template<typename T>
struct NPY_API ntvec3 {
   T x ;
   T y ;
   T z ;
  const char* desc() const ;
};

template<typename T>
struct NPY_API ntrange4 {
   ntvec4<T> min ; 
   ntvec4<T> max ; 
};

template<typename T>
struct NPY_API ntrange3 {
   ntvec3<T> min ; 
   ntvec3<T> max ; 
};




struct NPY_API nuvec4 {

  // NO CTOR
  void dump(const char* msg);

  unsigned x ; 
  unsigned y ; 
  unsigned z ; 
  unsigned w ; 
};


struct NPY_API nuvec3 {

  // NO CTOR
  void dump(const char* msg);
  const char* desc() const ;

  unsigned x ; 
  unsigned y ; 
  unsigned z ; 
};





struct NPY_API nivec4 {

  // NO CTOR
  void dump(const char* msg);

  int x ; 
  int y ; 
  int z ; 
  int w ; 
};



struct NPY_API nivec3 {
  
  // non-quad so give it a CTOR
  nivec3( int x, int y, int z ) : x(x), y(y), z(z) {} ; 
  const char* desc() const ;
  void dump(const char* msg);

  int x ; 
  int y ; 
  int z ; 

  nivec3& operator += (const nivec3& other)
  {
      x += other.x ; 
      y += other.y ; 
      z += other.z ; 
      return *this ; 
  }

  nivec3& operator *= (const int s)
  {
      x *= s ; 
      y *= s ; 
      z *= s ; 
      return *this ; 
  }

};


inline NPY_API nivec3 operator+(const nivec3 &a, const nivec3& b)
{
     return nivec3(a.x + b.x, a.y + b.y, a.z + b.z );
}
inline NPY_API nivec3 operator-(const nivec3 &a, const nivec3& b)
{
     return nivec3(a.x - b.x, a.y - b.y, a.z - b.z );
}

inline NPY_API nivec3 operator*(const nivec3 &a, const int s)
{
     return nivec3(a.x*s, a.y*s, a.z*s );
}
inline NPY_API int operator*(const nivec3 &a, const nivec3& b)
{
     return a.x*b.x + a.y*b.y + a.z*b.z ;
}

inline NPY_API nivec3 operator*(const int s , const nivec3 &a)
{
     return nivec3(s*a.x, s*a.y, s*a.z );
}







struct NPY_API nvec4 {

  // NO CTOR
  void dump(const char* msg);
  const char* desc();

  float x ; 
  float y ; 
  float z ; 
  float w ; 



};


union NPY_API nquad 
{

   nuvec4 u ; 
   nivec4 i ; 
   nvec4  f ; 

   void dump(const char* msg);
};


struct NPY_API nvec3 {

  //nvec3(float x, float y, float z) : x(x), y(y), z(z) {} ; 
  //~nvec3(){} ;

  void dump(const char* msg);
  const char* desc();

  float x ; 
  float y ; 
  float z ; 


};







inline NPY_API nuvec4 make_nuvec4(unsigned x, unsigned y, unsigned z, unsigned w ) 
{
   nuvec4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}
inline NPY_API nuvec3 make_nuvec3(unsigned x, unsigned y, unsigned z) 
{
   nuvec3 t; t.x = x; t.y = y; t.z = z; return t;
}
inline NPY_API nivec4 make_nivec4(int x, int y, int z, int w )
{
   nivec4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}
inline NPY_API nivec3 make_nivec3(int x, int y, int z )
{
   nivec3 t(x,y,z); return t;
}





inline NPY_API unsigned nminu(const unsigned a, const unsigned b)
{
    return a < b ? a : b ; 
}
inline NPY_API unsigned nmaxu(const unsigned a, const unsigned b)
{
    return a > b ? a : b ; 
}
inline NPY_API int nmini(const int a, const int b)
{
    return a < b ? a : b ; 
}
inline NPY_API int nmaxi(const int a, const int b)
{
    return a > b ? a : b ; 
}
inline NPY_API float nminf(const float a, const float b)
{
    return a < b ? a : b ; 
}
inline NPY_API float nmaxf(const float a, const float b)
{
    return a > b ? a : b ; 
}

inline NPY_API float nmaxf(const nvec3& a)
{
   return nmaxf(nmaxf(a.x, a.y), a.z);
}




inline NPY_API nvec3 make_nvec3(float x, float y, float z )
{
   nvec3 t ; t.x = x ; t.y = y ; t.z = z  ; return t;
}
inline NPY_API nvec3 nminf( const nvec3& a, const nvec3& b )
{
    return make_nvec3( nminf(a.x, b.x), nminf(a.y, b.y), nminf(a.z, b.z) );
}
inline NPY_API nvec3 nmaxf( const nvec3& a, const nvec3& b )
{
    return make_nvec3( nmaxf(a.x, b.x), nmaxf(a.y, b.y), nmaxf(a.z, b.z) );
}

inline NPY_API nvec3 nabsf(const nvec3& a)
{
   return make_nvec3( fabs(a.x), fabs(a.y), fabs(a.z));
}










inline NPY_API nvec4 make_nvec4(float x, float y, float z, float w )
{
   nvec4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}
inline NPY_API nvec4 nminf( const nvec4& a, const nvec4& b )
{
    return make_nvec4( nminf(a.x, b.x), nminf(a.y, b.y), nminf(a.z, b.z), nminf(a.w, b.w) );
}
inline NPY_API nvec4 nmaxf( const nvec4& a, const nvec4& b )
{
    return make_nvec4( nmaxf(a.x, b.x), nmaxf(a.y, b.y), nmaxf(a.z, b.z), nmaxf(a.w, b.w) );
}



inline NPY_API nvec3 operator+(const nvec3 &a, const nvec3& b)
{
     return make_nvec3(a.x + b.x, a.y + b.y, a.z + b.z );
}
inline NPY_API nvec3 operator-(const nvec3 &a, const nvec3& b)
{
     return make_nvec3(a.x - b.x, a.y - b.y, a.z - b.z );
}


inline NPY_API nvec3 operator*(const nvec3 &a, const float s)
{
     return make_nvec3(a.x*s, a.y*s, a.z*s );
}
inline NPY_API nvec3 operator*(const float s, const nvec3 &a)
{
     return make_nvec3(a.x*s, a.y*s, a.z*s );
}

inline NPY_API float operator*(const nvec3 &a, const nvec3& b)
{
     return a.x*b.x + a.y*b.y + a.z*b.z ;
}





