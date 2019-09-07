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

//
//  ColorDomain
//         x : 0.
//         y : total number of colors in color buffer (*)
//         z : number of psychedelic colors 
//         w : 0.
// 
// (*) total number of colors is required to convert a buffer index 
//     into float 0->1 for texture access
//
// offsets into the color buffer are obtained from dynamic define
// in order to match those of  GColors::setupCompositeColorBuffer
//

    switch(ColorParam.x)
    {
       case 0:
              fcolor = vec4(1.0,1.0,1.0,1.0) ; break;
       case 1:
              fcolor = texture(Colors, (float(flq[0].x) + MATERIAL_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) ; break; 
       case 2:
              fcolor = texture(Colors, (float(flq[1].x) + MATERIAL_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) ; break; 
       case 3:
              fcolor = texture(Colors, (float(flq[0].w) + FLAG_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) ; break; 
       case 4:
              fcolor = texture(Colors, (float(flq[1].w) + FLAG_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) ; break; 
       case 5:
              fcolor = vec4(vec3(polarization[0]), 1.0) ; break;
       case 6:
              fcolor = vec4(vec3(polarization[1]), 1.0) ; break;
    }


//
// material coloring  
//
//
//     flq[0].x  sorta working m1? colors 
//     flq[0].y  sorta working m2?
//     flq[0].z  all red, expected to be zero (yielding -1 off edge of texture, so picks up value of 1 based "1")
//     flq[0].w  variable colors, looks consistent with flags
//
//     flq[1].x  a bit more distinct m1  
//     flq[1].y
//     flq[1].z  
//     flq[1].w  not much variation  
//
//
//  history coloring
//
//  float idxcol  = (32.0 + float(flq[1].w) - 1.0 + 0.5)/ColorDomain.y ;    
//
//     flq[0].x   color variation
//     flq[0].y   color variation
//     flq[0].z   all dull grey, consistent with expected zero yielding "-1" and landing on buffer prefill 0x444444 
//     flq[0].w   very subtle coloring mostly along muon path,  
//
//     flq[1].x   more distinct color variation
//     flq[1].y   more distinct color variation
//     flq[1].z   all dull grey (buffer prefill) 
//     flq[1].w   more obvious, but still too much white : off by one maybe?
// 
//  problem is that the most common step flag is BT:boundary transmit (now cyan)
//  which makes flying point view not so obvious  
//

