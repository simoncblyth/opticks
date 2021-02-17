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
#include "OpticksSwitches.h"


// TODO: get this from a GBndLib rather than duplicating 
enum {
    OMAT,
    OSUR,
    ISUR,
    IMAT 
};

__device__ void fill_state( State& s, int boundary, uint4 identity, float wavelength )
{
    // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
    // >0 outward going photon
    // <0 inward going photon
    //
    // NB the line is above the details of the payload (ie how many float4 per matsur) 
    //    it is just 
    //                boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 
    //

    int line = boundary > 0 ? (boundary - 1)*BOUNDARY_NUM_MATSUR : (-boundary - 1)*BOUNDARY_NUM_MATSUR  ; 

    // pick relevant lines depening on boundary sign, ie photon direction relative to normal
    // 
    int m1_line = boundary > 0 ? line + IMAT : line + OMAT ;   
    int m2_line = boundary > 0 ? line + OMAT : line + IMAT ;   
    int su_line = boundary > 0 ? line + ISUR : line + OSUR ;   

    //  consider photons arriving at PMT cathode surface
    //  geometry normals are expected to be out of the PMT 
    //
    //  boundary sign will be -ve : so line+3 outer-surface is the relevant one

    s.material1 = boundary_lookup( wavelength, m1_line, 0);  
    s.m1group2  = boundary_lookup( wavelength, m1_line, 1);  

    s.material2 = boundary_lookup( wavelength, m2_line, 0); 
    s.surface   = boundary_lookup( wavelength, su_line, 0);                 

    s.optical = optical_buffer[su_line] ;   // index/type/finish/value

    s.index.x = optical_buffer[m1_line].x ; // m1 index
    s.index.y = optical_buffer[m2_line].x ; // m2 index 
    s.index.z = optical_buffer[su_line].x ; // su index
    s.index.w = identity.w   ;

    s.identity = identity ; 

}

__device__ void  dump_state( State& s)
{
#ifdef WITH_PRINT
    rtPrintf(" dump_state:material1       %10.4f %10.4f %10.4f %10.4f ri/al/sl/rp \n", s.material1.x, s.material1.y, s.material1.z, s.material1.w );
    rtPrintf(" dump_state:material2       %10.4f %10.4f %10.4f %10.4f ri/al/sl/rp \n", s.material2.x, s.material2.y, s.material2.z, s.material2.w );
    rtPrintf(" dump_state:surface         %10.4f %10.4f %10.4f %10.4f dt/ab/rs/rd \n", s.surface.x, s.surface.y, s.surface.z, s.surface.w );
    rtPrintf(" dump_state:optical         %10u %10u %10u %10i ix/ty/fi/va \n", s.optical.x, s.optical.y, s.optical.z, s.optical.w );
    rtPrintf(" dump_state:index           %10u %10u %10u %10i m1/m2/su/se \n", s.index.x  , s.index.y,   s.index.z,   s.index.w );
    rtPrintf(" dump_state:identity        %10u %10u %10u %10i no/me/bd/se \n", s.identity.x  , s.identity.y,   s.identity.z,   s.identity.w );
#endif
}


