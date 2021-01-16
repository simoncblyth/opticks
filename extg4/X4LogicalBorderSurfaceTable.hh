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

#include "X4_API_EXPORT.hh"

class GSurfaceLib ; 

#include "G4LogicalBorderSurface.hh"   // forced to include for the typedef 
#include "X4_API_EXPORT.hh"

#include "plog/Severity.h"

/**
X4LogicalBorderSurfaceTable
============================

**/

class X4_API X4LogicalBorderSurfaceTable 
{
        static const plog::Severity LEVEL ; 
    public:
        static void Convert(GSurfaceLib* dst);
        static const std::vector<G4LogicalBorderSurface*>* PrepareVector(const G4LogicalBorderSurfaceTable* tab) ; 
    private:
        X4LogicalBorderSurfaceTable(GSurfaceLib* dst);
        void init();
    private:
        const std::vector<G4LogicalBorderSurface*>* m_src ;  
        GSurfaceLib*                                m_dst ; 
        

};
