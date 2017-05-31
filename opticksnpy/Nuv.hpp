#pragma once
#include <string>
#include "NPY_API_EXPORT.hh"

struct NPY_API nuv 
{
    unsigned s() const {   return s_ & 0xffff ; } 
    unsigned u() const {   return u_ & 0xffff ; } 
    unsigned v() const {   return v_ & 0xffff ; } 

    unsigned nu() const {  return u_ >> 16 ; } 
    unsigned nv() const {  return v_ >> 16 ; } ;

    std::string desc() const ;
    std::string detail() const ;

    unsigned s_ ; 
    unsigned u_ ; 
    unsigned v_ ;
};


inline nuv make_uv(unsigned s, unsigned u, unsigned v, unsigned nu, unsigned nv)
{
   nuv uv ; 

   uv.s_ = s ; 
   uv.u_ = (nu << 16) | u  ;
   uv.v_ = (nv << 16) | v  ;

   return uv ; 
}

