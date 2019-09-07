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


uniform mat4 ModelViewProjection ;
uniform ivec4 PickFace ;

layout (triangles) in;
layout (line_strip, max_vertices=6) out;

out vec4 fcolor ;

//
//  vertex ordering matching OptiX: intersect_triangle_branchless
//    /Developer/OptiX/include/optixu/optixu_math_namespace.h  
//

void main()
{
   vec3 p0 = gl_in[0].gl_Position.xyz;
   vec3 p1 = gl_in[1].gl_Position.xyz;
   vec3 p2 = gl_in[2].gl_Position.xyz;

   vec3 pc = (p0 + p1 + p2)/3.0 ;  
 
   float length = 100. ; 

   vec3 e0 = p1 - p0; 
   vec3 e1 = p0 - p2;    
   vec3 no = normalize(cross( e1, e0 )); 

   vec3 a0 = normalize(e0); 
   vec3 a1 = normalize(e1); 

   uint face_index = gl_PrimitiveIDIn ;   

   // no way to select solids here : but this is typically used when focussing on a single mesh anyhow
   bool picked = ( PickFace.x > 0 &&  PickFace.y > 0 && face_index >= PickFace.x && face_index < PickFace.y ) ;


   fcolor = picked ? vec4(1.0,1.0,1.0,1.0)  :  vec4(1.0,0.0,0.0,1.0) ;
   gl_Position = ModelViewProjection * vec4 (pc, 1.0);
   EmitVertex();
   gl_Position = ModelViewProjection * vec4 (pc + no*length, 1.0);
   EmitVertex();
   EndPrimitive();

   fcolor = picked ? vec4(1.0,1.0,1.0,1.0)  :  vec4(0.0,1.0,0.0,1.0) ;
   gl_Position = ModelViewProjection * vec4 (pc, 1.0);
   EmitVertex();
   gl_Position = ModelViewProjection * vec4 (pc + a0*length, 1.0);
   EmitVertex();
   EndPrimitive();

   fcolor = picked ? vec4(1.0,1.0,1.0,1.0)  :  vec4(0.0,0.0,1.0,1.0) ;
   gl_Position = ModelViewProjection * vec4 (pc, 1.0);
   EmitVertex();
   gl_Position = ModelViewProjection * vec4 (pc + a1*length, 1.0);
   EmitVertex();
   EndPrimitive();
}

