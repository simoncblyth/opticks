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

#include <glm/fwd.hpp>
#include "NPY_API_EXPORT.hh"
struct nnode ; 

struct NPY_API NSolid
{
    static const char* x018_label ; 
    static const char* x019_label ; 
    static const char* x020_label ; 
    static const char* x021_label ; 

    static nnode* createEllipsoid( const char* name, float ax, float by, float cz, float zcut1, float zcut2 ) ;
    static nnode* createTubs( const char* name , float rmin, float rmax, float hz ) ;
    static nnode* createTorus( const char* name,  float rmin, float rmax, float rtor ) ;

    static nnode* createSubtractionSolid(   const char* name, nnode* left , nnode* right, void* rot, glm::vec3 tlate ) ;
    static nnode* createUnionSolid(         const char* name, nnode* left , nnode* right, void* rot, glm::vec3 tlate ) ;
    static nnode* createIntersectionSolid(  const char* name, nnode* left , nnode* right, void* rot, glm::vec3 tlate ) ;

    static nnode* create_x018();
    static nnode* create_x018_f();
    static nnode* create_x018_c();

    static nnode* create_x019();
    static nnode* create_x020();
    static nnode* create_x021();

    static nnode* create(int lv);

};


