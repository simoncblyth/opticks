#pragma once 
#include <cstdio>

// initially attempted to use glm vec constituents 
// but that runs into complications with unions and non-default ctor, dtor, copy-ctor
//  
// http://stackoverflow.com/questions/26572240/why-does-union-has-deleted-default-constructor-if-one-of-its-member-doesnt-have
// http://stackoverflow.com/questions/7299171/union-member-has-a-non-trivial-copy-constructor
// http://www.boost.org/doc/libs/1_47_0/doc/html/variant.html
//

enum { X, Y, Z, W };

struct nuvec4 {
  unsigned int x ; 
  unsigned int y ; 
  unsigned int z ; 
  unsigned int w ; 
  void dump(const char* msg);
};

struct nivec4 {
  int x ; 
  int y ; 
  int z ; 
  int w ; 
  void dump(const char* msg);
};

struct nvec4 {
  float x ; 
  float y ; 
  float z ; 
  float w ; 
  void dump(const char* msg);
};

union nquad 
{
   nuvec4 u ; 
   nivec4 i ; 
   nvec4  f ; 
   void dump(const char* msg);
};


inline void nuvec4::dump(const char* msg)
{
    printf("%s : %10u %10u %10u %10u \n",msg, x,y,z,w ); 
}
inline void nivec4::dump(const char* msg)
{
    printf("%s : %10d %10d %10d %10d \n",msg, x,y,z,w ); 
}
inline void nvec4::dump(const char* msg)
{
    printf("%s : %10.4f %10.4f %10.4f %10.4f \n",msg, x,y,z,w ); 
}
inline void nquad::dump(const char* msg)
{
    printf("%s\n", msg);
    f.dump("f");
    u.dump("u");
    i.dump("i");
}



