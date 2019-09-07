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

#include "G4UserRunAction.hh"
#include "plog/Severity.h"

class OpticksHub ; 

class CRunAction : public G4UserRunAction
{
        static const plog::Severity LEVEL ; 
    public:
        CRunAction(OpticksHub* hub);
        virtual ~CRunAction();
    public:
        void BeginOfRunAction(const G4Run*);
        void   EndOfRunAction(const G4Run*); 
    private:
        OpticksHub*  m_hub ; 
        unsigned     m_count ; 

};
