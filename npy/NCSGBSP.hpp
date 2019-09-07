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


#include "OpticksCSG.h"
#include "NPY_API_EXPORT.hh"

#include "NTriSource.hpp" 

struct csgjs_csgnode ; 
struct csgjs_model ; 

struct NPY_API NCSGBSP : NTriSource 
{
    static csgjs_csgnode* ConvertToBSP( const NTriSource* tris) ; 

    NCSGBSP(const NTriSource* left_, const NTriSource* right_, OpticksCSG_t operation );
    void init();

    csgjs_csgnode* left ; 
    csgjs_csgnode* right ; 
    OpticksCSG_t   operation ; 

    csgjs_csgnode* combined ; 
    csgjs_model*   model ; 


    // NTriSource interface
    unsigned get_num_tri() const ;
    unsigned get_num_vert() const ;
    void get_vert( unsigned i, glm::vec3& v ) const ;
    void get_normal( unsigned i, glm::vec3& n ) const ;
    void get_uv(  unsigned i, glm::vec3& uv ) const ;
    void get_tri( unsigned i, glm::uvec3& t ) const ;
    void get_tri( unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const ;


};


