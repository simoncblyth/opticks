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

#include <glm/fwd.hpp>
#include <map>
#include <string>
#include <vector>

class Types ; 
template <typename T> class NPY ; 
#include "NPY.hpp"


// host based indexing of photon data 

#include "NPY_API_EXPORT.hh"

class NPY_API BoundariesNPY {
   public:  
       BoundariesNPY(NPY<float>* photons); 
   public:
       void setTypes(Types* types);
       void setBoundaryNames(std::map<unsigned int, std::string> names);    
       // just used to give a name to a boundary code
   public:
       // boundary code integer is cos_theta signed by OptiX in cu/material1_propagate.cu
       void indexBoundaries();
       void dump(const char* msg="BoundariesNPY::dump");

   private:  
       static bool second_value_order(const std::pair<int,int>&a, const std::pair<int,int>&b);

   public:  
       glm::ivec4                                    getSelection();           
       // 1st four boundary codes provided by the selection
       std::vector< std::pair<int, std::string> >&   getBoundaries(); 

   protected:
       NPY<float>*                                  m_photons ; 
       std::map<unsigned int, std::string>          m_names ; 
       Types*                                       m_types ; 
       std::vector< std::pair<int, std::string> >   m_boundaries ; 
       unsigned int                                 m_total ; 

};




 
