#pragma once
/**
CSGGeometry
=============

CSGGeometry is a higher level wrapper for CSGFoundry which 
avoids repetition of geometry setup, loading and querying mechanics. 

**/

#include <vector>
struct CSGFoundry ; 
struct CSGQuery ; 
struct CSGDraw ; 
struct float4 ; 

#include "plog/Severity.h"
#include "CSG_API_EXPORT.hh"

struct CSG_API CSGGeometry 
{
    static const plog::Severity LEVEL ;   
    static void Draw( const CSGFoundry* fd, const char* msg="CSGGeometry::Draw"  ); 
    static const char* OutDir(const char* cfbase, const char* geom, const char* sopr);  

    const char* default_geom ; 
    const char* default_sopr ; 
    const char* geom ; 
    const char* sopr ; 
    const char* cfbase ; 
    const char* outdir ; 
    const char* name ; 

    const CSGFoundry* fd ; 
    const CSGQuery*   q ;  
    float4*          ce ; 

    CSGDraw*    d ;    // cannot be const because of canvas

    std::vector<int>* sxyzw ;
    std::vector<int>* sxyz ; 
    bool no_selection ; 
    int sx ;  
    int sy ;  
    int sz ;  
    int sw ;  

    void operator()(); 
    CSGGeometry(const char* default_cfbase = nullptr, const CSGFoundry* fd_ = nullptr); 

    void init(); 
    void init_fd(); 
    void init_selection();  


    void saveSignedDistanceField() const ; 


    void centerExtentGenstepIntersect() ; // invokes one of the below
    void saveCenterExtentGenstepIntersect(float t_min) const ; 
    void intersectSelected(const char* path); 


    void dump(const char* msg="CSGGeometry::dump") const ; 
    void draw(const char* msg="CSGGeometry::draw") ; 

};



