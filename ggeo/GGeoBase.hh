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

#include "GGEO_API_EXPORT.hh"

class GSourceLib ; 
class GScintillatorLib ; 
class GSurfaceLib ; 
class GMaterialLib ; 
class GBndLib ; 
class GGeoLib ; 
//class GPmtLib ; 
class GNodeLib ; 
class GMeshLib ; 
class GMergedMesh ; 

class GGEO_API GGeoBase {
    public:
        virtual GScintillatorLib* getScintillatorLib() const = 0 ; 
        virtual GSourceLib*       getSourceLib() const = 0 ; 
        virtual GSurfaceLib*      getSurfaceLib() const = 0 ; 
        virtual GMaterialLib*     getMaterialLib() const = 0 ; 

        virtual GBndLib*          getBndLib() const = 0 ; 
   //     virtual GPmtLib*          getPmtLib() const = 0 ; 
        virtual GGeoLib*          getGeoLib() const = 0 ;        // GMergedMesh 
        virtual GNodeLib*         getNodeLib() const = 0 ;       // GNode/GVolume pv,lv names
        virtual GMeshLib*         getMeshLib() const = 0 ;      

        virtual const char*       getIdentifier() const = 0 ; 
        virtual GMergedMesh*      getMergedMesh(unsigned index) const = 0 ; 

};
