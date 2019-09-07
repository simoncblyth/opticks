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

class GMergedMesh ; 

#include "GMesh.hh"
#include "GGEO_API_EXPORT.hh"
class GGEO_API GBBoxMesh : public GMesh {
public:
    //enum { NUM_VERTICES = 8, NUM_FACES = 6*2 } ;
    enum { NUM_VERTICES = 24, NUM_FACES = 6*2 } ;  // 6*4 = 24 : a blown apart box, 2 tri "face" per box facet 
public:
    static GBBoxMesh* create(GMergedMesh* mergedmesh);
private:
    void eight(); 
    void twentyfour(); 
public:
    static void twentyfour(gbbox& bb, gfloat3* vertices, guint3* faces, gfloat3* normals);
public:
    GBBoxMesh(GMergedMesh* mm) ; 
    virtual ~GBBoxMesh(); 
private:
    GMergedMesh* m_mergedmesh ; 
     
};



