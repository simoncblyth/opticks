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
   T_UNDEF,
   T_SPHERE,
   T_POINT,
   T_DISC,
   T_DISC_INTERSECT_SPHERE,
   T_DISC_INTERSECT_SPHERE_DUMB,
   T_DISCLIN,
   T_DISCAXIAL,
   T_INVSPHERE,
   T_REFLTEST,
   T_INVCYLINDER,
   T_RING, 
   T_NUM_TYPE
}               Torch_t ;

typedef enum {
   M_UNDEF             = 0x0 ,
   M_SPOL              = 0x1 << 0,
   M_PPOL              = 0x1 << 1,
   M_FLAT_THETA        = 0x1 << 2, 
   M_FLAT_COSTHETA     = 0x1 << 3,
   M_FIXPOL            = 0x1 << 4,
   M_WAVELENGTH_SOURCE = 0x1 << 5,
   M_WAVELENGTH_COMB   = 0x1 << 6
}              Mode_t ; 


#ifndef __CUDACC__


#include <string>
#include "NGLM.hpp"

template<typename T> class NPY ; 

#include "plog/Severity.h"

#include "GenstepNPY.hpp"
#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

/**
TorchStepNPY
==============

Frame targetting and NPY creation are handled in base class GenstepNPY, 
currently the only other GenstepNPY subclass is FabStepNPY 


**/

class NPY_API TorchStepNPY : public GenstepNPY {
   public:
       typedef enum { TYPE, 
                      MODE, 
                      POLARIZATION, 
                      FRAME,  
                      TRANSFORM, 
                      SOURCE, 
                      TARGET, 
                      PHOTONS, 
                      MATERIAL, 
                      ZENITHAZIMUTH, 
                      WAVELENGTH, 
                      WEIGHT, 
                      TIME, 
                      RADIUS, 
                      DISTANCE, 
                      UNRECOGNIZED } Param_t ;


       static const plog::Severity LEVEL ; 
       static const char* DEFAULT_CONFIG ; 

       static const char* TYPE_; 
       static const char* MODE_; 
       static const char* POLARIZATION_; 
       static const char* FRAME_ ; 
       static const char* TRANSFORM_ ; 
       static const char* SOURCE_ ; 
       static const char* TARGET_ ; 
       static const char* PHOTONS_ ; 
       static const char* MATERIAL_ ; 
       static const char* ZENITHAZIMUTH_ ; 
       static const char* WAVELENGTH_ ; 
       static const char* WEIGHT_ ; 
       static const char* TIME_ ; 
       static const char* RADIUS_ ; 
       static const char* DISTANCE_ ; 

       static const char* T_UNDEF_ ; 
       static const char* T_SPHERE_ ; 
       static const char* T_POINT_ ; 
       static const char* T_DISC_ ; 
       static const char* T_DISCLIN_ ; 
       static const char* T_DISCAXIAL_ ; 
       static const char* T_DISC_INTERSECT_SPHERE_ ; 
       static const char* T_DISC_INTERSECT_SPHERE_DUMB_ ; 
       static const char* T_INVSPHERE_ ; 
       static const char* T_REFLTEST_ ; 
       static const char* T_INVCYLINDER_ ; 
       static const char* T_RING_ ; 

       static const char* M_SPOL_ ; 
       static const char* M_PPOL_ ; 
       static const char* M_FLAT_THETA_ ; 
       static const char* M_FLAT_COSTHETA_ ; 
       static const char* M_FIXPOL_ ; 
       static const char* M_WAVELENGTH_SOURCE_ ; 
       static const char* M_WAVELENGTH_COMB_ ; 

   private:
       static Param_t ParseParam(const char* k);
       // global scope enum types as CUDA only sees those
       static ::Mode_t  ParseMode(const char* k);
       static ::Torch_t ParseType(const char* k);
   public:  
       TorchStepNPY(unsigned genstep_type, const char* config=NULL); 
       void updateAfterSetFrameTransform();  
   private:
       void init();
       void set(TorchStepNPY::Param_t param, const char* s );
   public:  
       void setMode(const char* s );
       void setType(const char* s );
   public:
       // Type
       bool isIncidentSphere() const ;
       bool isDisc() const ;
       bool isDiscLinear() const ;
       bool isRing() const ;
       bool isPoint() const ;
       bool isSphere() const ;
       bool isReflTest() const ;
   public:
       // Mode   
       bool isSPolarized() const ;
       bool isPPolarized() const ;
       bool isFixPolarized() const ;
   public:

       std::string description() const ;
       void Summary(const char* msg="TorchStepNPY::Summary") const ;
       std::string desc(const char* msg="TorchStepNPY::desc") const ;
       void dump(const char* msg="TorchStepNPY::dump") const ;

   public:
       // local positions/vectors, frame transform is applied in *update* yielding world frame m_post m_dirw 
       void setSourceLocal(const char* s );
       void setTargetLocal(const char* s );
       void setPolarizationLocal(const char* s );
   public:
       glm::vec4 getSourceLocal() const ;
       glm::vec4 getTargetLocal() const ;
       glm::vec4 getPolarizationLocal() const ;
   public:  
       ::Mode_t  getMode() const ;
       ::Torch_t getType() const ;
       std::string getModeString() const ;
       const char* getTypeName() const ;

  private:
       // position and directions to which the frame transform is applied in update
       glm::vec4    m_source_local ; 
       glm::vec4    m_target_local ; 
       glm::vec4    m_polarization_local ; 
  private:
       glm::vec4    m_src ;
       glm::vec4    m_tgt ;
       glm::vec4    m_pol ;
       glm::vec3    m_dir ;

};

#include "NPY_TAIL.hh"


#endif


