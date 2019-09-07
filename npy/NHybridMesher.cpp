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

#include <sstream>

#include "BFile.hh"

#include "NHybridMesher.hpp"
#include "NTrianglesNPY.hpp"
#include "NOpenMesh.hpp"

#ifdef OLD_PARAMETERS
#include "X_BParameters.hh"
#else
#include "NMeta.hpp"
#endif


#include "NNode.hpp"

#include "PLOG.hh"



#ifdef OLD_PARAMETERS
NHybridMesher::NHybridMesher(const nnode* node, X_BParameters* meta, const char* treedir)
#else
NHybridMesher::NHybridMesher(const nnode* node, NMeta* meta, const char* treedir)
#endif
    :
    m_mesh(MESH::Make(node, meta, treedir)),
    m_bbox( new nbbox(node->bbox()) ), 
    m_treedir(treedir ? strdup(treedir) : NULL )
{
}


std::string NHybridMesher::desc()
{
   std::stringstream ss ; 
   ss << "NHybridMesher"
      ;
   return ss.str(); 
}


NTrianglesNPY* NHybridMesher::operator()()
{
    NTrianglesNPY* tt = new NTrianglesNPY(m_mesh);  // NTriSource pull out the tris
    return tt ; 
}





