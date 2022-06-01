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
SGeo
======

Protocol base used to facilitate lower level package access
to limited geometry information, by passing the higher level 
GGeo instance down to it cast down to this SGeo protocol base.

Also used by CSG/CSGFoundry::upload to record the CFBase directory 
of the last geometry uploaded to the device in a location that
is accessible from anywhere. (HMM: an alt approach would be to set an envvar for this ?)

**/

#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SGeo 
{
    private:
        static const char* LAST_UPLOAD_CFBASE ;
        static plog::Severity LEVEL ; 
    public:
        static void SetLastUploadCFBase(const char* cfbase);   
        static const char* LastUploadCFBase() ; 
        static const char* LastUploadCFBase_OutDir(); 
    public:
        virtual unsigned           getNumMeshes() const = 0 ; 
        virtual const char*        getMeshName(unsigned midx) const = 0 ;
        virtual int                getMeshIndexWithName(const char* name, bool startswith) const = 0 ;
        virtual ~SGeo(){};

};


