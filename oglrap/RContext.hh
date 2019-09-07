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

#include "OGLRAP_API_EXPORT.hh"
#include "plog/Severity.h"
#include <glm/glm.hpp>


struct InstLODCullContext ;

struct OGLRAP_API RContext
{
    static const plog::Severity LEVEL ; 
    static const char* uniformBlockName ;

    InstLODCullContext*   uniform ; 
    GLuint                uniformBO ; 
   
    RContext();

    void init();

    void initUniformBuffer();
    void bindUniformBlock(GLuint program);

    void update( const glm::mat4& world2clip, const glm::mat4& world2eye, const glm::vec4& lodcut );

};
