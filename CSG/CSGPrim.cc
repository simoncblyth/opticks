
#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include "sutil_vec_math.h"
#include "qat4.h"
#include "CSGPrim.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>
#include <cstring>


std::string CSGPrim::desc() const 
{  
    std::stringstream ss ; 
    ss 
      << "CSGPrim"
      << " numNode/node/tran/plan" 
      << std::setw(4) << numNode() << " "  
      << std::setw(4) << nodeOffset() << " "
      << std::setw(4) << tranOffset() << " "
      << std::setw(4) << planOffset() << " " 
      << "sbtOffset/meshIdx/repeatIdx/primIdx " 
      << std::setw(4) << sbtIndexOffset() << " "
      << std::setw(4) << meshIdx() << " "
      << std::setw(4) << repeatIdx() << " " 
      << std::setw(4) << primIdx()  
      << " mn " << mn() 
      << " mx " << mx() 
      ;
    std::string s = ss.str(); 
    return s ; 
}



CSGPrimSpec CSGPrim::MakeSpec( const CSGPrim* prim0,  unsigned primIdx, unsigned numPrim ) // static 
{
    const CSGPrim* prim = prim0 + primIdx ; 

    CSGPrimSpec ps ; 
    ps.aabb = prim->AABB() ; 
    ps.sbtIndexOffset = prim->sbtIndexOffsetPtr() ;  
    ps.num_prim = numPrim ; 
    ps.stride_in_bytes = sizeof(CSGPrim); 
    return ps ; 
}

#endif


