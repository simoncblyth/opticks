#pragma once

#include <vector>
#include "NPY_API_EXPORT.hh"
struct NP ; 
struct nbbox ; 

struct NPY_API NContour
{
    static void XZ_bbox_grid( std::vector<float>& xx, std::vector<float>& yy, const nbbox& bb, float sx=0.01, float sy=0.01, int mx=10, int my=20 ); 

    unsigned ni ; 
    unsigned nj ; 

    NP* X ; 
    NP* Y ; 
    NP* Z ; 

    float* zdat ; 

    NContour(const std::vector<float>& xx, const std::vector<float>& yy ); 
    void setZ(unsigned i, unsigned j, float z); 
    void save(const char* base, const char* rela, const char* relb ) const ; 

};

