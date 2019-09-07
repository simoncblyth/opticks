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

//  http://www.ppsloan.org/publications/XYZJCGT.pdf
//
//  Below functions from listing 1 of above publication were 
//  obtained by fits to CIE XYZ color matching observations of 
//  human tristimulus response.
//
//  These allow conversion of a wavelength spectrum into CIE XYZ values
//  which can then be converted on to RGB
//

#include <math.h>
#include <stdio.h>

float xFit_1931( float wave )
{
    float t1 = (wave-442.0f)*((wave<442.0f)?0.0624f:0.0374f);
    float t2 = (wave-599.8f)*((wave<599.8f)?0.0264f:0.0323f);
    float t3 = (wave-501.1f)*((wave<501.1f)?0.0490f:0.0382f);
    return 0.362f*expf(-0.5f*t1*t1) + 1.056f*expf(-0.5f*t2*t2)- 0.065f*expf(-0.5f*t3*t3);
}

float yFit_1931( float wave )
{
    float t1 = (wave-568.8f)*((wave<568.8f)?0.0213f:0.0247f);
    float t2 = (wave-530.9f)*((wave<530.9f)?0.0613f:0.0322f);
    return 0.821f*expf(-0.5f*t1*t1) + 0.286f*expf(-0.5f*t2*t2);
}

float zFit_1931( float wave )
{
    float t1 = (wave-437.0f)*((wave<437.0f)?0.0845f:0.0278f);
    float t2 = (wave-459.0f)*((wave<459.0f)?0.0385f:0.0725f);
    return 1.217f*expf(-0.5f*t1*t1) + 0.681f*expf(-0.5f*t2*t2);
}


void dump()
{
    for(int iw=400 ; iw < 800 ; iw+=20 )
    {
       float w = (float)iw ; 
       float x = xFit_1931(w);
       float y = yFit_1931(w);
       float z = zFit_1931(w);
       printf(" %3d : %10.4f %10.4f %10.4f \n", iw, x, y, z );  
    }
}
