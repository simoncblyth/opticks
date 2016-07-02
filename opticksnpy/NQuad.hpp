#pragma once 
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


struct NPY_API nuvec4 {

  // NO CTOR
  void dump(const char* msg);

  unsigned int x ; 
  unsigned int y ; 
  unsigned int z ; 
  unsigned int w ; 
};


struct NPY_API nivec4 {

  // NO CTOR
  void dump(const char* msg);

  int x ; 
  int y ; 
  int z ; 
  int w ; 
};

struct NPY_API nvec4 {

  // NO CTOR
  void dump(const char* msg);

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

  // NO CTOR
  void dump(const char* msg);

  float x ; 
  float y ; 
  float z ; 
};




inline NPY_API nuvec4 make_nuvec4(unsigned int x, unsigned int y, unsigned int z, unsigned int w ) 
{
   nuvec4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}


inline NPY_API nivec4 make_nivec4(int x, int y, int z, int w )
{
   nivec4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

inline NPY_API nvec4 make_nvec4(float x, float y, float z, float w )
{
   nvec4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}


inline NPY_API nvec3 make_nvec3(float x, float y, float z )
{
   nvec3 t; t.x = x; t.y = y; t.z = z; return t;
}



