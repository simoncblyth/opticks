#pragma once

#include "plog/Severity.h"
#include "CSG_API_EXPORT.hh"

struct CSGQuery ; 
struct SCanvas ; 

struct CSG_API CSGDraw
{
    static const plog::Severity LEVEL ; 

    CSGDraw(const CSGQuery* q_) ; 
    void draw(const char* msg);
    void draw_tree_r(int nodeIdxRel, int depth, int& inorder, char axis ) ;
    void draw_list(); 
    void draw_leaf();
    void draw_list_item( const CSGNode* nd, unsigned idx ); 

    const CSGQuery* q ; 

    int          type ; 
    int          width ; 
    int          height ; 
    SCanvas*     canvas ; 
    bool         dump ; 

};
 
