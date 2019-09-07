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

//#pragma debug(on)

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_colour;
layout(location = 2) in vec3 vertex_normal;
layout(location = 3) in vec2 vertex_texcoord;

//out vec3 colour;
out vec2 texcoord;

void main () 
{
    vec4 normal = ModelView * vec4 (vertex_normal, 0.0);

    //colour = normalize(vec3(normal))*0.5 + 0.5 ;
    ////colour = normalize(vec3(1.))*0.5 + 0.5 ;
    ////colour = vertex_colour ;
    ////colour = vec3(0,1.0,0) ;

    //gl_Position = ModelViewProjection * vec4 (vertex_position, 1.0);
    gl_Position = vec4 (vertex_position, 1.0);

    texcoord = vertex_texcoord;

}


// color not used in frag so dont output, only the texcoord is used 
//

