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

#pragma debug(on)

//in vec3 colour;
in vec2 texcoord;

out vec4 frag_colour;

uniform  vec4 ScanParam ;
uniform vec4 ClipPlane ;
uniform ivec4 NrmParam ;

uniform sampler2D ColorTex ;

void main () 
{
   frag_colour = texture(ColorTex, texcoord);
   float depth = frag_colour.w ;  // alpha is hijacked for depth in pinhole_camera.cu material1_radiance.cu
   frag_colour.w = 1.0 ; 

   gl_FragDepth = depth  ;

   if(NrmParam.z == 1)
   {
        if(depth < ScanParam.x || depth > ScanParam.y ) discard ;
   } 
}


//
//  the input color is ignored 
//
//
// http://www.roxlu.com/2014/036/rendering-the-depth-buffer
//
// gl_FragDepth = 1.1 ;   // black
// gl_FragDepth = 1.0 ;   // black
// gl_FragDepth = 0.999 ; //  visible geometry
// gl_FragDepth = 0.0   ; //  visible geometry 
//
// frag_colour = vec4( depth, depth, depth, 1.0 );  
// vizualize fragment depth, the closer you get to geometry the darker it gets 
// reaching black just before being near clipped
//
