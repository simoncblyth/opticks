#pragma once

struct BConfig ; 

typedef enum 
{
   CSG_BBOX_ANALYTIC = 1,
   CSG_BBOX_POLY     = 2, 
   CSG_BBOX_PARSURF  = 3,
   CSG_BBOX_G4POLY   = 4
} NSceneConfigBBoxType ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API NSceneConfig 
{
    static const char* CSG_BBOX_ANALYTIC_ ;
    static const char* CSG_BBOX_POLY_ ;
    static const char* CSG_BBOX_PARSURF_ ;
    static const char* CSG_BBOX_G4POLY_ ;

    static const char* BBoxType( NSceneConfigBBoxType bbty );

    NSceneConfig(const char* cfg);
    struct BConfig* bconfig ;  
    void dump(const char* msg="NSceneConfig::dump") const ; 

    int check_surf_containment ; 
    int check_aabb_containment ; 
    int disable_instancing     ;   // useful whilst debugging geometry subsets 

    int  csg_bbox_analytic ; 
    int  csg_bbox_poly ; 
    int  csg_bbox_parsurf ; 
    int  csg_bbox_g4poly ;   // only available from GScene level 

    int parsurf_target ; 
    int parsurf_level ; 
    int parsurf_margin ; 
 
    NSceneConfigBBoxType default_csg_bbty ; 
    NSceneConfigBBoxType bbox_type() const ; 
    const char* bbox_type_string() const ; 

};

