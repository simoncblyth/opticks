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

#include <cstring>
#include "CSurfaceTable.hh"

CSurfaceTable::CSurfaceTable(const char* name)
   :
   m_name(strdup(name))
{
}

const char* CSurfaceTable::getName()
{
    return m_name ; 
}

void CSurfaceTable::add(const G4OpticalSurface* surf)
{
    m_surfaces.push_back(surf);
}

unsigned CSurfaceTable::getNumSurf()
{
    return m_surfaces.size();
}

const G4OpticalSurface* CSurfaceTable::getSurface(unsigned index)
{
    return index < m_surfaces.size() ? m_surfaces[index] : NULL ; 
}

 
