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

#include "CFG4_PUSH.hh"
#include "G4StepStatus.hh"
#include "CBoundaryProcess.hh"
#include "CStage.hh"
#include "CFG4_POP.hh"

#include "CFG4_API_EXPORT.hh"

class G4StepPoint ; 

#include <string>


class CFG4_API OpStatus 
{
    public:
        static std::string OpStepString(const G4StepStatus status);
#ifdef USE_CUSTOM_BOUNDARY
       static std::string  OpBoundaryString(const Ds::DsG4OpBoundaryProcessStatus status);
       static std::string OpBoundaryAbbrevString(const Ds::DsG4OpBoundaryProcessStatus status);
       static unsigned OpBoundaryFlag(const Ds::DsG4OpBoundaryProcessStatus status);
       static unsigned OpPointFlag(const G4StepPoint* point, const Ds::DsG4OpBoundaryProcessStatus bst, CStage::CStage_t stage);
#else
       static std::string  OpBoundaryString(const G4OpBoundaryProcessStatus status);
       static std::string OpBoundaryAbbrevString(const G4OpBoundaryProcessStatus status);
       static unsigned OpBoundaryFlag(const G4OpBoundaryProcessStatus status);
       static unsigned OpPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst, CStage::CStage_t stage);
#endif

       static  bool IsTerminalFlag(unsigned flag); 
};



