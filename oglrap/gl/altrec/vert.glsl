#version 410 core
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


// altrec/vert.glsl

layout(location = 0) in vec4  rpos;
layout(location = 1) in vec4  rpol;  
layout(location = 2) in ivec4 rflg;  
layout(location = 3) in ivec4 rsel;  
layout(location = 4) in uvec4 rflq;  

out vec4 polarization ;
out uvec4 flags ;
out uvec4 flq ;
out ivec4 sel  ;

void main () 
{
    sel = rsel ; 
    polarization = rpol ; 
    flags = rflg ; 
    flq = rflq ;

    gl_Position = rpos ; 
    gl_PointSize = 1.0;

}


