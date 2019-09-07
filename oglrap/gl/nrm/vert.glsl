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


#incl dynamic.h

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;
uniform vec4 ClipPlane ;
uniform vec4 LightPosition ; 
uniform vec4 Param ;
uniform ivec4 NrmParam ;

uniform vec4 ColorDomain ;
uniform sampler1D Colors ;


layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_colour;
layout(location = 2) in vec3 vertex_normal;

float gl_ClipDistance[1]; 

out vec4 colour;

void main () 
{
    //
    // NB using flipped normal, for lighting from inside geometry 
    //
    //    normals are expected to be outwards so the natural 
    //    sign of costheta is negative when the light is inside geometry 
    //    thus in order to see something flip the normals 
    //

    float flip = NrmParam.x == 1 ? -1. : 1. ;

    vec3 normal = flip * normalize(vec3( ModelView * vec4 (vertex_normal, 0.0)));

    vec3 vpos_e = vec3( ModelView * vec4 (vertex_position, 1.0));  // vertex position in eye space 

    gl_ClipDistance[0] = dot(vec4(vertex_position, 1.0), ClipPlane);

    vec3 ambient = vec3(0.1, 0.1, 0.1) ;

#incl vcolor.h

    gl_Position = ModelViewProjection * vec4 (vertex_position, 1.0);

}


