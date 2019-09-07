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

#include "NGLM.hpp"
#include "NQuad.hpp"
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

template <typename FVec, typename IVec> struct NFieldGrid3 ; 
typedef NFieldGrid3<glm::vec3, glm::ivec3>  FG3 ; 


class NPY_API NOct
{
public:
    enum 
    {
        Node_None,
        Node_Internal,
        Node_Psuedo,
        Node_Leaf,
    };

    static int Corners( const glm::ivec3& ijk, FG3* f, const nvec4& ce, const int ncorner=8, const int size=1 );

    static void PopulateLeaf(int corners, NOct* leaf, FG3* f, const nvec4& ce );

    static void GenerateVertexIndices(NOct* node, std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals, const nbbox& bb, const nvec4& ce, FG3* fg);
    static void ContourCellProc(NOct* node, std::vector<int>& indexBuffer);
    static NOct* ConstructOctreeNodes(NOct* node, FG3* fg, const nvec4& ce, int& count);
    static NOct* SimplifyOctree(NOct* node, float threshold);



    NOct()
        : 
          type(Node_None),
          min(0, 0, 0),
          size(0)
    {
          for (int i = 0; i < 8; i++) children[i] = NULL ; 
    }

    int             type;
    glm::ivec3      min;
    int             size;
    NOct*           children[8];

};



inline bool operator == ( const NOct& a, const NOct& b)
{
    return a.type == b.type && 
           a.size == b.size &&
           a.min == b.min  ;
}




