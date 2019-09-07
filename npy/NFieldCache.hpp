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
#include <functional>
#include  <boost/unordered_map.hpp>
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

class NPY_API NFieldCache {
    public:
         typedef boost::unordered_map<unsigned int, float> UMAP ; 
         NFieldCache(std::function<float(float,float,float)> field, const nbbox& bb);
         float operator()(float x, float y, float z);
         unsigned getMortonCode(float x, float y, float z);
         std::string desc();
         void reset();
         std::function<float(float,float,float)> func();
    private:
         std::function<float(float,float,float)>   m_field ;

         nbbox m_bbox ; 
         nvec3 m_side ; 
         UMAP m_cache;
         unsigned m_calc ; 
         unsigned m_lookup ; 
  
};
