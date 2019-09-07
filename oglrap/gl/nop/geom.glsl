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

//  nop/geom.glsl : flying point 

#incl dynamic.h

uniform mat4 ModelViewProjection ;
uniform vec4 Param ; 

in vec4 polarization[];

layout (lines) in;
layout (points, max_vertices = 1) out;

out vec4 fcolor ; 

void main () 
{
    vec4 p0 = gl_in[0].gl_Position  ;
    vec4 p1 = gl_in[1].gl_Position  ;

    fcolor = vec4(1.0,0.0,0.0,1.0);

    gl_PointSize = 2.0;

    //gl_Position = ModelViewProjection * vec4( vec3(p0), 1.0 ) ; 
    //EmitVertex();
    //EndPrimitive();

    float tc = Param.w ;

    uint valid  = (uint(p0.w > 0.)  << 0) + (uint(p1.w > 0.) << 1) + (uint(p1.w > p0.w) << 2) ; 
    uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) ;
    uint vselect = valid & select ; 
   
    // b111 = 0x7 
    // b101 = 0x5
    // b001 = 0x1  

    if(vselect == 0x7) // both valid and straddling tc 
    {
        vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) );
        gl_Position = ModelViewProjection * vec4( pt, 1.0 ) ; 
   
        EmitVertex();
        EndPrimitive();
    }
    else if( valid == 0x7 && select == 0x1 )     // both valid and prior to tc 
    {
        vec3 pt = vec3(p1) ;
        gl_Position = ModelViewProjection * vec4( pt, 1.0 ) ; 
   
        EmitVertex();
        EndPrimitive();
    }

} 


