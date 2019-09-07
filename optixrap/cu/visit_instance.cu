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


#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, instance_bounding_radius , , );

//#define WITH_PRINT 1


RT_PROGRAM void visit_instance()
{
    const float distance = length( ray.origin ) ;  // Visit program ray.origin is in OBJECT frame
    const unsigned level = distance < instance_bounding_radius ? 0u : 1u ;  
    rtIntersectChild( level );
}

RT_PROGRAM void visit_instance_WORLD()
{
    /*
    Transform gymnastics here actually pointless... 
    No need to convert between frames, OBJECT -> WORLD 
    just directly use OBJECT frame ray.origin to yield 
    the same distance.

        0  1  2  3
        4  5  6  7
        8  9 10 11
       12 13 14 15 
    */
    
    float matrix[16] ;
    rtGetTransform( RT_OBJECT_TO_WORLD , matrix ) ;
    const float3 ipos = make_float3( matrix[3], matrix[7], matrix[11] );  // 12 13 14 was (0,0,0)
    const float3 orig = rtTransformPoint( RT_OBJECT_TO_WORLD, ray.origin );
    const float3 offset = orig - ipos  ;  
    const float distance = length( offset ) ; 
    const float distance1 = length( ray.origin ) ; 
    const unsigned level = distance < instance_bounding_radius ? 0u : 1u ;  

#ifdef WITH_PRINT
    rtPrintf("visit_instance_WORLD: level %d instance_bounding_radius %10.3f distance %10.3f distance1 %10.3f  orig (%10.3f %10.3f %10.3f) ipos (%10.3f %10.3f %10.3f)  \n", 
          level, 
          instance_bounding_radius,
          distance,
          distance1,
          orig.x, orig.y, orig.z,
          ipos.x, ipos.y, ipos.z
       ); 
    
#endif

    rtIntersectChild( level );
}



