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

#include <vector>
#include <string>

#include "NTriSource.hpp"

struct NPY_API NTris : NTriSource
{
    std::vector<glm::vec3>  verts ; 
    std::vector<glm::uvec3> tris ; 

    void add( const glm::vec3& a, const glm::vec3& b, const glm::vec3& s);

    unsigned get_num_tri() const ;
    unsigned get_num_vert() const ;
    void get_vert( unsigned i, glm::vec3& v ) const ;
    void get_normal( unsigned i, glm::vec3& n ) const ;
    void get_uv(  unsigned i, glm::vec3& uv ) const  ;
    void get_tri( unsigned j, glm::uvec3& t ) const ;
    void get_tri( unsigned j, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const ;

    void dump(const char* msg="Tris::dump") const ;
    std::string brief() const ;


    static NTris* make_sphere( unsigned n_polar=8, unsigned n_azimuthal=8, float ctmin=-1.f, float ctmax=1.f ) ;

};

