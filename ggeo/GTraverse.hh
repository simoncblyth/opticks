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

template<class T> class Counts ;

class GGeo ; 
class GBndLib ; 
class GMaterialLib ; 
class GNode ; 

/**
GTraverse
===========

Takes a quick spin over the GGeo GVolume tree counting material
names obtained from volume bpundaries.

**/


#include "GGEO_API_EXPORT.hh"
class GGEO_API GTraverse {
   public:
        GTraverse(GGeo* ggeo);
   public:
        void init();
        void traverse();
   private:
        void traverse( const GNode* node, unsigned int depth );
   private:
       GGeo*                  m_ggeo ; 
       GBndLib*               m_blib ; 
       GMaterialLib*          m_mlib ; 
       Counts<unsigned int>*  m_materials_count ; 
 
};


