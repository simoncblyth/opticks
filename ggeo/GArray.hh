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

#include "GBuffer.hh"

// WHERE USED ? TODO: replace with NPY

#include "GGEO_API_EXPORT.hh"

template <class T>
class GGEO_API GArray : public GBuffer {
  public:
     GArray(unsigned int length, const T* values)
      :
       GBuffer(sizeof(T)*length, (void*)values, sizeof(T), 1, "GArray"),
       m_length(length)
     {
     }

     virtual ~GArray()
     {
     }

     unsigned int getLength()
     {
         return m_length ;
     }
     const T* getValues()
     {
         return (T*)m_pointer ;
     }

  private:
     unsigned int m_length ;
     const T* m_values ;

};



