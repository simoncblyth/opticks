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

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

struct NPY_API NTriSource 
{
    virtual unsigned get_num_tri() const = 0 ;
    virtual unsigned get_num_vert() const = 0 ;
    virtual void get_vert( unsigned i, glm::vec3& v ) const = 0 ;
    virtual void get_normal( unsigned i, glm::vec3& n ) const = 0 ;
    virtual void get_uv(  unsigned i, glm::vec3& v ) const = 0 ;
    virtual void get_tri( unsigned i, glm::uvec3& t ) const = 0 ;
    virtual void get_tri( unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const = 0;

};

