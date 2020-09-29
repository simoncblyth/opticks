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
#include "plog/Severity.h"

class GNode ; 
template<class T> class NPY ;

/**
GTree
=============

Pulling out intended-to-be-common parts of GScene and GInstancer into GTree
to avoid duplicity issues. 

Used by: GMergedMesh and GInstancer 

Creates NPY buffers and populates with info from 
the instance placements lists.


**/

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GTree {
    public: 
        static const plog::Severity LEVEL ; 
        static NPY<float>*    makeInstanceTransformsBuffer(const std::vector<const GNode*>& placements);
        static NPY<unsigned>* makeInstanceIdentityBuffer(  const std::vector<const GNode*>& placements)  ;         // ?->InstanceVolumeIdentityBuffer
};





