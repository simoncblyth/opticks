#pragma once

#include "plog/Severity.h"
#include "CSG_API_EXPORT.hh"

struct CSGQuery ; 
struct SCanvas ; 

struct CSG_API CSGDraw
{
    static const plog::Severity LEVEL ; 

    CSGDraw(const CSGQuery* q_, char axis_ ) ; 
    void draw(const char* msg);
    void draw_tree_r(int nodeIdxRel, int depth, int& inorder ) ;
    void draw_list(); 
    void draw_leaf();
    void draw_list_item( const CSGNode* nd, unsigned idx ); 

    const CSGQuery* q ; 
    const char   axis ; 

    int          type ; 
    int          width ; 
    int          height ; 
    SCanvas*     canvas ; 
    bool         dump ; 

};
 
