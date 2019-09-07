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

#include "G4StepPoint.hh"
#include "CPoi.hh"
#include "CAction.hh"
#include "OpticksFlags.hh"
#include "OpStatus.hh"

#include "Format.hh"



#ifdef USE_CUSTOM_BOUNDARY
CPoi::CPoi(const G4StepPoint* point, unsigned flag, unsigned material, Ds::DsG4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage, const G4ThreeVector& origin) 
#else
CPoi::CPoi(const G4StepPoint* point, unsigned flag, unsigned material, G4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage, const G4ThreeVector& origin) 
#endif
   :
   m_point(new G4StepPoint(*point)), 
   m_flag(flag),
   m_material(material),
   m_boundary_status(boundary_status),
   m_stage(stage),
   m_action(0),
   m_origin(origin)
{
}

CPoi::~CPoi()
{
   delete m_point ;  
}



const G4StepPoint* CPoi::getPoint() const 
{
    return m_point ; 
}

unsigned CPoi::getFlag() const 
{
    return m_flag ;
}
unsigned CPoi::getMaterial() const 
{
    return m_material ;
}



#ifdef USE_CUSTOM_BOUNDARY
Ds::DsG4OpBoundaryProcessStatus CPoi::getBoundaryStatus() const
#else
G4OpBoundaryProcessStatus   CPoi::getBoundaryStatus() const
#endif
{
   return m_boundary_status ;  
}
CStage::CStage_t CPoi::getStage() const
{
   return m_stage ; 
}



std::string CPoi::description() const 
{
    std::stringstream ss ; 
    ss 
       << " CPoi " << OpticksFlags::Abbrev(m_flag)
       << "   " << std::setw(5) << OpStatus::OpBoundaryAbbrevString(m_boundary_status) 
       << std::endl 
       << ::Format(m_point, m_origin, "Poi", true ) 
       ;
    return ss.str(); 
}


