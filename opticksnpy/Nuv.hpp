#pragma once
#include <string>
#include "NPY_API_EXPORT.hh"

struct NPY_API nuv 
{
    unsigned s() const {   return s_ & 0xffff ; } 
    unsigned u() const {   return u_ & 0xffff ; } 
    unsigned v() const {   return v_ & 0xffff ; } 
    unsigned p() const {   return p_ ; }   // primitive index

    unsigned nu() const {  return u_ >> 16 ; } 
    unsigned nv() const {  return v_ >> 16 ; } ;

    float   fu() const  { return float(u())/float(nu()) ; } 
    float   fv() const  { return float(v())/float(nv()) ; } 

    float  fu2pi(bool seamed=true) const ; 
    float  fvpi() const ; 

    bool is_interior(unsigned margin=0) const ;


    std::string desc() const ;
    std::string detail() const ;

    unsigned s_ ; 
    unsigned u_ ; 
    unsigned v_ ;
    unsigned p_ ;
};



inline nuv make_uv(unsigned s, unsigned u, unsigned v, unsigned nu, unsigned nv, unsigned p)
{
   nuv uv ; 

   uv.s_ = s ; 
   uv.u_ = (nu << 16) | u  ;
   uv.v_ = (nv << 16) | v  ;
   uv.p_ = p ; 


   return uv ; 
}

