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

//uniform mat4 ModelViewProjection ;
//uniform mat4 ModelView ;
//uniform vec4 LODCUT ; 

layout(points) in; 
layout(points, max_vertices = 1) out;

in mat4 ITransform[1] ;
flat in int objectVisible[1];

layout(stream=0) out vec4 VizTransform0LOD0 ;
layout(stream=0) out vec4 VizTransform1LOD0 ;
layout(stream=0) out vec4 VizTransform2LOD0 ;
layout(stream=0) out vec4 VizTransform3LOD0 ;

layout(stream=1) out vec4 VizTransform0LOD1 ;
layout(stream=1) out vec4 VizTransform1LOD1 ;
layout(stream=1) out vec4 VizTransform2LOD1 ;
layout(stream=1) out vec4 VizTransform3LOD1 ;

layout(stream=2) out vec4 VizTransform0LOD2 ;
layout(stream=2) out vec4 VizTransform1LOD2 ;
layout(stream=2) out vec4 VizTransform2LOD2 ;
layout(stream=2) out vec4 VizTransform3LOD2 ;


void main()
{
    mat4 tr = ITransform[0] ;

    if(objectVisible[0] == 1)
    {
        vec4 InstancePosition = tr[3] ;
        vec4 IEye = ModelView * InstancePosition ;
        float distance = length(IEye.xyz) ;
        //float distance = InstancePosition.x  ;
        //float distance = IEye.y  ;
        // must use the uniforms

        int lod = distance < LODCUT.x ? 0 : ( distance < LODCUT.y ? 1 : 2 ) ;

        //int lod = distance < LODCUT.x ? 1 : 2  ; 

        switch(lod)
        {
           case 0:
                VizTransform0LOD0 = tr[0]  ;
                VizTransform1LOD0 = tr[1]  ;
                VizTransform2LOD0 = tr[2]  ;
                VizTransform3LOD0 = tr[3]  ;

                EmitStreamVertex(0);
                EndStreamPrimitive(0);
                break ;

           case 1:
                VizTransform0LOD1 = tr[0]  ;
                VizTransform1LOD1 = tr[1]  ;
                VizTransform2LOD1 = tr[2]  ;
                VizTransform3LOD1 = tr[3]  ;

                EmitStreamVertex(1);
                EndStreamPrimitive(1);
                break ;

           case 2:
                VizTransform0LOD2 = tr[0]  ;
                VizTransform1LOD2 = tr[1]  ;
                VizTransform2LOD2 = tr[2]  ;
                VizTransform3LOD2 = tr[3]  ;

                EmitStreamVertex(2);
                EndStreamPrimitive(2);
                break ;
        }
    }
}



