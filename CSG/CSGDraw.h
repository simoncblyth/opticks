#pragma once
/**
CSGDraw.h
============


**/

#include "plog/Severity.h"
#include "CSG_API_EXPORT.hh"
#include <string>

struct CSGQuery ; 
struct scanvas ; 

struct CSG_API CSGDraw
{
    static const plog::Severity LEVEL ; 

    CSGDraw(const CSGQuery* q_, char axis_ ) ; 

    char* get() const ; 
    void render(); 
    void draw_tree_r(int nodeIdxRel, int depth, int& inorder ) ;
    void draw_list(); 
    void draw_leaf();
    void draw_list_item( const CSGNode* nd, unsigned idx ); 

    std::string hdr() const ; 
    std::string desc()  ; 


    const CSGQuery* q ; 
    const char   axis ; 

    int          type ; 
    int          width ; 
    int          height ; 
    scanvas*     canvas ; 
    bool         dump ; 

};
 
