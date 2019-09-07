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

#include <algorithm>   
#include <sstream>   

#include "X4SolidList.hh"

X4SolidList::X4SolidList()
{
}

void X4SolidList::addSolid(G4VSolid* solid)
{
    if(hasSolid(solid)) return ; 
    m_solidlist.push_back(solid); 
}
 
bool X4SolidList::hasSolid(G4VSolid* solid) const 
{
    return std::find(m_solidlist.begin(), m_solidlist.end(), solid) != m_solidlist.end()  ;
}

unsigned X4SolidList::getNumSolids() const 
{
    return m_solidlist.size() ; 
}

std::string X4SolidList::desc() const 
{
    std::stringstream ss ; 
    ss << "X4SolidList"
       << " NumSolids " << getNumSolids() 
       ;
    return ss.str(); 
}

 
