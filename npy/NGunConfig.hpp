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
#include "NGLM.hpp"

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API NGunConfig {
   public:
       static const char* DEFAULT_CONFIG ; 

       static const char* COMMENT_; 
       static const char* PARTICLE_; 
       static const char* FRAME_ ; 
       static const char* POSITION_ ; 
       static const char* DIRECTION_ ; 
       static const char* POLARIZATION_ ; 
       static const char* TIME_ ; 
       static const char* ENERGY_ ; 
       static const char* NUMBER_ ; 
       static const char* UNRECOGNIZED_ ; 
   public:
       typedef enum { 
                      COMMENT,
                      PARTICLE, 
                      FRAME,  
                      POSITION, 
                      DIRECTION, 
                      POLARIZATION, 
                      TIME, 
                      ENERGY, 
                      NUMBER, 
                      UNRECOGNIZED
                    } Param_t ;
   private:
       Param_t parseParam(const char* k);
       void set(NGunConfig::Param_t param, const char* s );
       const char* getParam(NGunConfig::Param_t param);
   public:
       NGunConfig();
       void Summary(const char* msg="NGunConfig::Summary");
   public:
       void parse(const char* config=NULL);
       void parse(std::string config);
   private:
       void setComment(const char* s);
       void setParticle(const char* s);
       void setNumber(const char* s);
       void setNumber(unsigned int num);
       void setEnergy(const char* s );
       void setEnergy(float energy);
       void setTime(const char* s );
       void setTime(float time);
       void setFrame(const char* s);
       void setPositionLocal(const char* s );
       void setDirectionLocal(const char* s );
       void setPolarizationLocal(const char* s );
   public: 
       void setFrameTransform(const char* s ); // 16 comma delimited floats 
       void setFrameTransform(glm::mat4& transform);
       const glm::mat4& getFrameTransform();
   private:  
       // methods invoked after setFrameTransform
       void update();
       void setPosition(const glm::vec3& pos );
       void setDirection(const glm::vec3& dir );
       void setPolarization(const glm::vec3& pol );
   public:  
       const char* getComment();
       const char* getParticle();
       unsigned int getNumber();
       float       getTime();
       float       getEnergy();
       int         getFrame();
       glm::vec3   getPosition();
       glm::vec3   getDirection();
       glm::vec3   getPolarization();
   private:
       void init();
   private:
       const char* m_config ;
  private:
       // position and directions to which the frame transform is applied in update
       glm::vec4    m_position_local ; 
       glm::vec4    m_direction_local ; 
       glm::vec4    m_polarization_local ; 
  private:
       const char*  m_comment ;
       const char*  m_particle ;
       unsigned int m_number ;  
       float        m_time  ; 
       float        m_energy  ; 
       glm::ivec4   m_frame ;
       glm::vec3    m_position ; 
       glm::vec3    m_direction ; 
       glm::vec3    m_polarization ; 
   private:
       glm::mat4    m_frame_transform ; 

};

#include "NPY_TAIL.hh"

