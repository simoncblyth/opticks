#pragma once

#include "NPY_API_EXPORT.hh"
#include <vector>
#include <string>

struct nd ; 
struct nmat4triple ;


struct NPY_API nd
{
   unsigned idx ;
   int repeatIdx ;
   unsigned mesh ; 
   unsigned depth ; 
   std::string boundary ; 

   nd*          parent ; 
   nmat4triple* transform ; 
   nmat4triple* gtransform ; 
   std::vector<nd*> children ; 

   std::string desc();
   static nmat4triple* make_global_transform(nd* n) ; 
};


