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

#include <glm/glm.hpp>
#include "NGenerator.hpp"


NGenerator::NGenerator(const nbbox& bb)
   :
   m_bb(bb),
   m_side({bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z }), 
   m_dist(0.f, 1.f),
   m_gen(m_rng, m_dist)
{
}

void NGenerator::operator()(nvec3& p)
{
    p.x = m_bb.min.x + m_gen()*m_side.x ; 
    p.y = m_bb.min.y + m_gen()*m_side.y ; 
    p.z = m_bb.min.z + m_gen()*m_side.z ;
}

void NGenerator::operator()(glm::vec3& p)
{
    p.x = m_bb.min.x + m_gen()*m_side.x ; 
    p.y = m_bb.min.y + m_gen()*m_side.y ; 
    p.z = m_bb.min.z + m_gen()*m_side.z ;
}


