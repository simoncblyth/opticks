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

#include <vector>

struct nnode ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API NNodeSample 
{
    static void Tests(std::vector<nnode*>& nodes );
    static nnode* Sphere1(); 
    static nnode* Sphere2(); 
    static nnode* Union1(); 
    static nnode* Intersection1(); 
    static nnode* Difference1(); 
    static nnode* Difference2(); 
    static nnode* Union2(); 
    static nnode* Box(); 
    static nnode* SphereBoxUnion(); 
    static nnode* SphereBoxIntersection(); 
    static nnode* SphereBoxDifference(); 
    static nnode* BoxSphereDifference(); 

    static nnode* Sample(const char* name);
    static nnode* DifferenceOfSpheres(); 
    static nnode* Box3(); 

    static nnode* _Prepare(nnode* root); 
};


