#pragma once

struct CSGQuery ; 
struct SCanvas ; 

#include "CSG_API_EXPORT.hh"

struct CSG_API CSGDraw
{
    CSGDraw(const CSGQuery* q_) ; 
    void draw(const char* msg);
    void draw_r(int nodeIdxRel, int depth, int& inorder ) ;

    const CSGQuery* q ; 
    unsigned   width ; 
    unsigned   height ; 
    SCanvas*   canvas ; 

};
 
