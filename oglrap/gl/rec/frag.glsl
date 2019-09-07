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


in vec4 fcolor;
out vec4 frag_color;

//uniform  vec4 ScanParam ;
//uniform ivec4 NrmParam ;
//more efficient to skip in geometry shader rather than in fragment, if possible

void main () 
{
    frag_color = fcolor ;

    //if(NrmParam.z == 1)
    //{
    //    if(gl_FragCoord.z < ScanParam.x || gl_FragCoord.z > ScanParam.y ) discard ;
    //}

}


