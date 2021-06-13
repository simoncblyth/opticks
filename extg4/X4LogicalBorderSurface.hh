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
#include <vector>
#include <utility>

class G4LogicalBorderSurface ;
class GBorderSurface ; 

#include "plog/Severity.h"

#include "X4_API_EXPORT.hh"

/**
X4LogicalBorderSurface
=======================

**/

class X4_API X4LogicalBorderSurface
{
        static const plog::Severity LEVEL ; 
    public:
        static GBorderSurface* Convert(const G4LogicalBorderSurface* src);
        static int GetItemIndex( const G4LogicalBorderSurface* item ) ;
        static std::string DescCandidateImplicitBorderSurface( const std::vector<std::pair<const void*, const void*>>& v_pvpv ) ; 


};


