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

#pragma once

// see OBoundaryLib::convertColors GColors::setupCompositeColorBuffer

rtTextureSampler<uchar4,2>  color_texture  ;
rtDeclareVariable(uint4, color_domain, , );

static __device__ __inline__ uchar4 color_lookup(unsigned int line)
{
    uchar4 col = tex2D(color_texture, line+0.5f, 0.5f );  
    col.w = 255u ;   // texture somehow mangled, assuming RGBA    GB are close but A is 0 
    return col ;
}

static __device__ __inline__ void color_dump()
{
    //rtPrintf("color_lookup:color_dump %u %u %u %u \n", color_domain.x, color_domain.y, color_domain.z, color_domain.w );  // color_lookup:color_dump 0 256 147 64 
    for(unsigned int i=0 ; i < color_domain.y ; i++) 
    {
        uchar4 c = color_lookup(i) ; 
        rtPrintf("color_dump %u : %u %u %u %u \n", i, c.x, c.y, c.z, c.w );
    }

}

