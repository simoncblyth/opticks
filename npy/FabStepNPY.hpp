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

// for both non-CUDA and CUDA compilation
typedef enum {
   F_UNDEF,
   F_POINT,
   F_NUM_TYPE
}  Fab_t ;

#ifndef __CUDACC__

#include "GenstepNPY.hpp"
#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

/**
FabStepNPY
=============

Frame targetting and NPY creation are handled in base class GenstepNPY, 
currently the only other GenstepNPY subclass is TorchStepNPY 

**/


class NPY_API FabStepNPY : public GenstepNPY 
{
    public:
        FabStepNPY(unsigned code, unsigned num_step, unsigned num_photons_per_step);
        void updateAfterSetFrameTransform();
    private:
        void addSteps(unsigned num_step); 
    private:
        unsigned m_num_photons_per_step ;

};

#include "NPY_TAIL.hh"
#endif


