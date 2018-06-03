#pragma once

#include "YOG_API_EXPORT.hh"

template <typename T> class NPY ;
struct NPYBufferSpec ; 

struct YOG_API YOGGeometry
{
    NPY<float>*     vtx ; 
    NPYBufferSpec*  vtx_spec ; 

    NPY<unsigned>*  idx ; 
    NPYBufferSpec*  idx_spec ; 

    void make_triangle(); 
};



