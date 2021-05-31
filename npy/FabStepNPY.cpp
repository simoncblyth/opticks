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


#include "NStep.hpp"
#include "FabStepNPY.hpp"


FabStepNPY::FabStepNPY(unsigned genstep_type, unsigned num_step, unsigned num_photons_per_step) 
    :  
    GenstepNPY(genstep_type, num_step, NULL, false ),
    m_num_photons_per_step(num_photons_per_step)
{
    init();
}

void FabStepNPY::init()
{
    unsigned num_step = getNumStep();
    for(unsigned i=0 ; i < num_step ; i++)
    {   
        m_onestep->setMaterialLine(i*10);   
        m_onestep->setNumPhotons(m_num_photons_per_step); 
        addStep();
    } 
}

void FabStepNPY::updateAfterSetFrameTransform()
{
}
