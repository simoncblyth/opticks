#pragma once
/**
CSGGeometry
=============

CSGGeometry is a higher level wrapper for CSGFoundry which 
avoids repetition of geometry setup, loading and querying mechanics. 

**/

struct CSGFoundry ; 
struct CSGQuery ; 

#include "CSG_API_EXPORT.hh"

struct CSG_API CSGGeometry 
{
    const char* default_geom ; 
    const char* geom ; 
    const char* cfbase ; 
    const char* name ; 

    const CSGFoundry* fd ; 
    const CSGQuery*   q ; 

    CSGGeometry(); 

    void init(); 
    void init_geom(); 
    void init_cfbase();  


    void saveSignedDistanceField() const ; 
    void saveCenterExtentGenstepIntersect() const ; 

};



