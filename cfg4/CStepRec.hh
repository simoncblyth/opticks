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

#include <vector>


template <typename T> class NPY ; 

// g4-
class G4Step ; 
class G4StepPoint ; 

class CStep ;     // cg4-


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/** 
CStepRec : Records non-optical particle step points 
=====================================================

Non-optical step points are serialized into the nopstep buffer of the 
current G4 event.  As the nopstep pointer is updated by 
OpticksHub as each G4 event is created this class needs
to take no action on changing event. It just keeps
recording into the nopstep provided by the hub.

CStepRec::initEvent sets m_nopstep the pointer to the 
buffer in which to store the data


**/


class CFG4_API CStepRec {
   public:
       CStepRec(Opticks* ok, bool dynamic);
       void initEvent(NPY<float>* nopstep);    
   private:
       void setNopstep(NPY<float>* nopstep);
   public:
       // collect into vector of CStep*
       void collectStep(const G4Step* step, unsigned int step_id);   
       void storeStepsCollected(unsigned int event_id, unsigned int track_id, int particle_id);
       unsigned int getStoreCount();
   private:
       void storePoint(unsigned int event_id, unsigned int track_id, int particle_id, unsigned int point_id, const G4StepPoint* point);
   private:
       Opticks*                    m_ok ; 
       bool                        m_dynamic ; 
       std::vector<const CStep*>   m_steps ; 
       unsigned                    m_store_count ; 
       unsigned                    m_num_vals ; 
       float*                      m_vals ;
       NPY<float>*                 m_nopstep ; 

};

#include "CFG4_TAIL.hh"


