/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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

    void env_override() ; 
    void dump(const char* msg="NSceneConfig::dump") const ; 

    int check_surf_containment ; 
    int check_aabb_containment ; 
    int disable_instancing     ;   // useful whilst debugging geometry subsets 

    int  csg_bbox_analytic ; 
    int  csg_bbox_poly ; 
    int  csg_bbox_parsurf ; 
    int  csg_bbox_g4poly ;   // only available from GScene level 

    int parsurf_epsilon ;   // specified by exponent eg -5 for 1e-5 is typical 
    int parsurf_target ; 
    int parsurf_level ; 
    int parsurf_margin ; 
    int verbosity ; 
    int polygonize ; 

    int instance_repeat_min ; 
    int instance_vertex_min ; 

 
    NSceneConfigBBoxType default_csg_bbty ; 
    NSceneConfigBBoxType bbox_type() const ; 
    const char* bbox_type_string() const ; 
    float get_parsurf_epsilon() const ;

};

