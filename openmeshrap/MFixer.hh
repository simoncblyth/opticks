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

/**
MFixer
=========

Mending cleaved meshes. 

DevNotes
---------

* no longer needed

**/


class GGeo ; 
class GMeshLib ; 
class MTool ; 

#include "MESHRAP_API_EXPORT.hh"
class MESHRAP_API MFixer {
   public:
       MFixer(GGeo* ggeo);
       void setVerbose(bool verbose=true);
       void fixMesh();
   private:
       void init();
   private:
       GGeo*     m_ggeo ;
       GMeshLib* m_meshlib ; 
       MTool*    m_tool ; 
       bool      m_verbose ; 

};


