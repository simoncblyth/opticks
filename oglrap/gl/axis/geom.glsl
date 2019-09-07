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


uniform mat4 Projection ;
uniform mat4 ModelViewProjection ;
uniform vec4 Param ; 

in vec3 direction[];
in vec3 colour[];

layout (points) in;
layout (line_strip, max_vertices = 2) out;

out vec3 fcolour ; 


void main () 
{
    gl_Position = ModelViewProjection * gl_in[0].gl_Position ;
    fcolour = colour[0] ;
    EmitVertex();

    gl_Position = ModelViewProjection * ( gl_in[0].gl_Position + Param.x*vec4(direction[0], 0.) ) ;
    fcolour = colour[0] ;
    EmitVertex();

    EndPrimitive();

} 
