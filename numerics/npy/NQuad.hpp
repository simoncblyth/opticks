#pragma once 

#include <glm/glm.hpp>

// this might be a problem with older compilers
// http://stackoverflow.com/questions/26572240/why-does-union-has-deleted-default-constructor-if-one-of-its-member-doesnt-have

union nquad 
{
   nquad();
   nquad(const nquad& other);
   nquad(const glm::vec4& f_);
   nquad(const glm::uvec4& u_);
   nquad(const glm::ivec4& i_);
   ~nquad();

   glm::uvec4 u ; 
   glm::ivec4 i ; 
   glm::vec4  f ; 
};


inline nquad::nquad() : f(0,0,0,0) {}

inline nquad::nquad(const nquad& other) : f(other.f) {}
inline nquad::nquad(const glm::vec4&  f_) : f(f_) {}
inline nquad::nquad(const glm::ivec4& i_) : i(i_) {}
inline nquad::nquad(const glm::uvec4& u_) : u(u_) {}

inline nquad::~nquad() {}

