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

/**
G4OpticksScintillationStep
============================

enum definition of the 6*4 floats/integers of scintillation gensteps

**/


#include "G4OK_API_EXPORT.hh"

class G4OK_API G4OpticksScintillationStep {
    public:

    enum {

       _Id, 
       _ParentID,
       _Material,
       _NumPhotons,
      
       _x0_x,
       _x0_y,
       _x0_z,
       _t0,

       _DeltaPosition_x,
       _DeltaPosition_y,
       _DeltaPosition_z,
       _step_length,

       _code,
       _charge, 
       _weight, 
       _MeanVelocity,

       _scnt,  
       _slowerRatio,   
       _slowTimeConstant,    
       _slowerTimeConstant,

       _ScintillationTime,
       _ScintillationIntegralMax,
       _Spare1,
       _Spare2,

       SIZE

    };

};


