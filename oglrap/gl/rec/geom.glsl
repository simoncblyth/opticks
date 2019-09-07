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

//  rec/geom.glsl : flying point 

#incl dynamic.h

uniform mat4 ISNormModelViewProjection ;
uniform vec4 TimeDomain ;
uniform vec4 ColorDomain ;
uniform vec4 Param ; 

uniform  vec4 ScanParam ;
uniform ivec4 NrmParam ;

//more efficient to skip in geometry shader rather than in fragment, if possible


uniform ivec4 Pick ;
uniform ivec4 RecSelect ; 
uniform ivec4 ColorParam ;
uniform ivec4 PickPhoton ;

uniform sampler1D Colors ;

in uvec4 flq[];
in ivec4 sel[];
in vec4 polarization[];

layout (lines) in;
layout (points, max_vertices = 1) out;

out vec4 fcolor ; 

void main () 
{
    uint seqhis = sel[0].x ; 
    uint seqmat = sel[0].y ; 
    if( RecSelect.x > 0 && RecSelect.x != seqhis )  return ;
    if( RecSelect.y > 0 && RecSelect.y != seqmat )  return ;

    uint photon_id = gl_PrimitiveIDIn/MAXREC ;   
    if( PickPhoton.x > 0 && PickPhoton.y > 0 && PickPhoton.x != photon_id )  return ;


    vec4 p0 = gl_in[0].gl_Position  ;
    vec4 p1 = gl_in[1].gl_Position  ;
    float tc = Param.w / TimeDomain.y ;

    uint valid  = (uint(p0.w >= 0.)  << 0) + (uint(p1.w >= 0.) << 1) + (uint(p1.w > p0.w) << 2) ; 
    uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(Pick.x == 0 || photon_id % Pick.x == 0) << 2) ;  
    uint vselect = valid & select ; 

#incl fcolor.h

    if(vselect == 0x7) // both valid and straddling tc 
    {
        vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) );
        gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ; 

        if(NrmParam.z == 1)
        {
            float depth = ((gl_Position.z / gl_Position.w) + 1.0) * 0.5;
            if(depth < ScanParam.x || depth > ScanParam.y ) return ; 
        }


        EmitVertex();
        EndPrimitive();
    }
    else if( valid == 0x7 && select == 0x5 )     // both valid and prior to tc 
    {
        vec3 pt = vec3(p1) ;
        gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ; 

        if(NrmParam.z == 1)
        {
            float depth = ((gl_Position.z / gl_Position.w) + 1.0) * 0.5;
            if(depth < ScanParam.x || depth > ScanParam.y ) return ; 
        }


        EmitVertex();
        EndPrimitive();
    }

} 


