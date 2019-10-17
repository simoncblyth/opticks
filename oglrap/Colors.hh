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

class Device ; 
template <typename T> class NPY ; 

#include "OGLRAP_API_EXPORT.hh"
#include "plog/Severity.h"

class OGLRAP_API Colors {
      static const plog::Severity LEVEL ; 
  public:
      Colors(Device* dev);
  public: 
      void setColorBuffer(NPY<unsigned char>* colorbuffer);
      unsigned int getNumColors();
  public: 
      void upload();
  private:
      Device*      m_device ; 
      GLuint       m_colors_tex ;
      bool         m_colors_uploaded ; 
      NPY<unsigned char>*     m_colorbuffer ; 


};      


