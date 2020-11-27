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

#include <optix.h>
#include <optix_math.h>

struct PerRayData_propagate
{
    float3 surface_normal ; 
    float distance_to_boundary ;
    int   boundary ; 
    uint4 identity ; 
    float cos_theta ;
};

/**

It would be good to squeeze this down to the size of 2*float4 


surface_normal
    essential for propagate.h eg for reflection

distance_to_boundary 
    rtIntersectionDistance is not available in raygen, so need in PRD 
    to pass from closest hit to raygen : essential for deciding history 

boundary
    signed 1-based index, currently occupying 32 bits, 16 bits easily enough

identity
    GVolume::getIdentity nodeIndex/tripletIdentity/shapeIdentity/sensorIndex
     
    16 bytes, but so useful : in principal could just use 4 bytes of nodeIndex and look 
    up the identity from identity buffers 

    actually this identity already has the 16 bits of unsigned boundary index
    within the shapeIdentity :  could just sign that in place  and avoid the separate boundary 

cos_theta
    sign is definitely needed, but is the value ? Actually the sign info  
    is already carried in the sign of the 1-based boundary index 

**/

