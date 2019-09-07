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


#incl InstLODCullContext.h


layout( location = 4) in mat4 InstanceTransform ;

out mat4 ITransform ;    
flat out int objectVisible;

void main() 
{    
    vec4 InstancePosition = InstanceTransform[3] ; 
    vec4 IClip = ModelViewProjection * InstancePosition ;    
  
    //float f = 0.95f ; 
    float f = 1.1f ; 
    objectVisible = 
         ( IClip.x < IClip.w*f && IClip.x > -IClip.w*f  ) &&
         ( IClip.y < IClip.w*f && IClip.y > -IClip.w*f  ) &&
         ( IClip.z < IClip.w*f && IClip.z > -IClip.w*f  ) ? 1 : 0 ; 
    
    //objectVisible = InstancePosition.y > 200.f ? 1 : 0 ; 
    //objectVisible  = 1 ; 


    ITransform = InstanceTransform ; 
}   







