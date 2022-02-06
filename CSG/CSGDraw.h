#pragma once

struct CSGQuery ; 
struct SCanvas ; 
#include "CSG_API_EXPORT.hh"

struct CSG_API CSGDraw
{
    CSGDraw(const CSGQuery* q_) ; 
    void draw(const char* msg);
    void draw_tree_r(int nodeIdxRel, int depth, int& inorder, char axis ) ;
    void draw_list(); 
    void draw_leaf();
    void draw_list_item( const CSGNode* nd, unsigned idx ); 

    const CSGQuery* q ; 

    int          type ; 
    unsigned     width ; 
    unsigned     height ; 
    SCanvas*     canvas ; 
    bool         dump ; 

};
 
