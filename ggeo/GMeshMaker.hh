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

#include "GGEO_API_EXPORT.hh"

class GMesh ; 
template <typename T> class NPY ; 

struct nbbox ; 


class GGEO_API GMeshMaker 
{
    public:
        static GMesh* Make( nbbox& bb ) ;
        static GMesh* MakeSphereLocal(NPY<float>* triangles, unsigned meshindex=0);  
        static GMesh* Make(NPY<float>* triangles, unsigned meshindex=0);

        // this one tries to avoid vertex duplication, but may have problems with normals
        // as a result, the GMesh assumption of same numbers of normals as verts aint so good  
        static GMesh* Make(NPY<float>* vtx3, NPY<unsigned>* tri3, unsigned meshindex=0);


};
 
