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

#include "NOct.hpp"

#include "NGrid3.hpp"
#include "NField3.hpp"
#include "NFieldGrid3.hpp"

template<typename FVec, typename IVec> struct NFieldGrid3 ; 



int NOct::Corners( const glm::ivec3& ijk, FG3* fg, const nvec4& /*ce*/, const int /*ncorner*/, const int /*size*/ )
{
    glm::vec3 fpos = fg->grid->fpos(ijk); 
    int corners = fg->field->zcorners(fpos, fg->grid->elem ) ;  
    return corners ; 
}



void NOct::PopulateLeaf(int , NOct* leaf, FG3* , const nvec4& )
{
    leaf->type = NOct::Node_Leaf;
}


void NOct::GenerateVertexIndices(NOct* /*node*/, std::vector<glm::vec3>& /*vertices*/,std::vector<glm::vec3>& /*normals*/ , const nbbox& /*bb*/, const nvec4& /*ce*/, FG3* /*fg*/)
{
}

void NOct::ContourCellProc(NOct* /*node*/, std::vector<int>& /*indexBuffer*/)
{
}

NOct* NOct::ConstructOctreeNodes(NOct* node, FG3* /*fg*/, const nvec4& /*ce*/, int& /*count*/)
{
   return node ; 
}
NOct* NOct::SimplifyOctree(NOct* node, float /*threshold*/)
{
   return node ; 
}




