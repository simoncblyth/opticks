#pragma once

#include "YOG_API_EXPORT.hh"
#include <vector>

template <typename T> class NPY ;

namespace YOG  {
struct YOG_API Geometry
{
    Geometry( int count_ );

    int             count ; 
    NPY<float>*     vtx ; 
    NPY<unsigned>*  idx ; 

    std::vector<float> vtx_minf ; 
    std::vector<float> vtx_maxf ; 

    std::vector<unsigned> idx_min ; 
    std::vector<unsigned> idx_max ; 

    std::vector<float> idx_minf ; 
    std::vector<float> idx_maxf ; 


    void make_triangle(); 
};

}  // namespace

