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

#include <string>

#include "plog/Severity.h"
#include "NGLM.hpp"

struct NStep ; 
template<typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"


/**
GenstepNPY
============

Base class of FabStepNPY and TorchStepNPY 

*setZenithAzimuth*
    Photons directions are generated using two random numbers in range 0:1 
    which are used scale the zenith and azimuth ranges.
    Default is a uniform sphere. Changing zenith ranges allows cones or
    rings to be generated and changing azimuth range allows 
    to chop the cone, ring or sphere.

                       mapped to 0:2pi of azimuth angle    
                    -------
           (0.f,1.f,0.f,1.f)
            --------
              mapped to 0:pi of zenith angle
**/


class NPY_API GenstepNPY {
   public:  
       static const plog::Severity LEVEL ; 
   public:  
       GenstepNPY(unsigned gentype, const char* config=NULL, bool is_default=false); 
       void addStep(); // increments m_step_index
       unsigned getNumStep() const ;

   public:
       // slots used by Geant4 only (not Opticks) from cfg4- 
       unsigned int getNumG4Event() const ;
       unsigned int getNumPhotonsPerG4Event() const ; 
       void setNumPhotonsPerG4Event(unsigned int n);
   public:
       NPY<float>* getNPY() const ;
       void         addActionControl(unsigned long long  action_control);

   public:  
       // thru to m_onestep 
       NStep*   getOneStep() const ;
       unsigned getNumPhotons() const ;
       void     setNumPhotons(unsigned num_photons);
       void     setMaterialLine(unsigned ml);
       void     setOriginTrackID(unsigned id);
   public:  
       // target setting needs external info regarding geometry 
       void setFrame(const char* s );
       void setFrame(unsigned vindex );
       const glm::ivec4&  getFrame() const ;
       int getFrameIndex() const ;
       void setFrameTransform(glm::mat4& transform);
       // targetting needs frame transform info which is done by GGeo::targetTorchStep(torchstep)

       void setFrameTransform(const char* s );       // directly from string of 16 comma delimited floats 
       virtual void updateAfterSetFrameTransform() = 0 ;   //  <-- provided by subclasses such as TorchstepNPY

       bool isFrameTargetted() const ;
       bool isDummyFrame() const ; 
       const glm::mat4& getFrameTransform() const ;

       std::string brief() const ; 
   private:
       void setFrameTargetted(bool targetted=true);
   public:  
        // methods invoked by update after frame transform is available
   public:  
       // need external help to set the MaterialLine
       void setMaterial(const char* s );
       const char* getConfig() const ;
       const char* getMaterial() const ;
       bool isDefault() const ; 

   protected: 
       NStep*       m_onestep ; 

  private:
       unsigned int m_gentype ; 
       const char*  m_config ;
       bool         m_is_default ; 
  private:
       const char*  m_material ;
       NPY<float>*  m_arr ; 
       unsigned int m_step_index ; 
  private:
       glm::ivec4   m_frame ;
       glm::mat4    m_frame_transform ; 
       bool         m_frame_targetted ; 
  private:
       unsigned int m_num_photons_per_g4event ;
 

};

#include "NPY_TAIL.hh"




