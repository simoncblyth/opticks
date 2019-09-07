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

#include "USEINSTANCE_API_EXPORT.hh"

struct USEINSTANCE_API Prog
{
    const char* vertSrc ;
    const char* geomSrc ;
    const char* fragSrc ;

    bool vert ; 
    bool geom ; 
    bool frag ;

    unsigned program ; 
    unsigned vertShader ;
    unsigned geomShader ;
    unsigned fragShader ;


    Prog(const char* vertSrc_, const char* geomSrc_,  const char* fragSrc_);

    void compile();
    void create();
    void link();
    void destroy();
    
    int getAttribLocation(const char* att);
};




