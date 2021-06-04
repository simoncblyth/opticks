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

struct CCtx ; 
struct CRecState ; 

#include "CRecorder.h"
#include "CFG4_API_EXPORT.hh"

/**
CPhoton
=========

Used CPU side only for Geant4 recording in OpticksEvent format.

Canonical m_photon instance is resident of CRecorder and a reference 
is held by CWriter and CDebug.

Builds seqhis, seqmat nibble by nibble just like GPU side generate.cu


**/


struct CFG4_API CPhoton
{
    const CCtx& _ctx ; 
    CRecState&    _state ; 

    unsigned _badflag ; 
    unsigned _slot_constrained ; 
    unsigned _material ; 
    uifchar4   _c4 ; 

    unsigned long long _seqhis ; 
    unsigned long long _seqmat ; 
    unsigned long long _mskhis ; 

    unsigned long long _his ; 
    unsigned long long _mat ; 
    unsigned long long _flag ; 

    unsigned long long _his_prior ; 
    unsigned long long _mat_prior ; 
    unsigned long long _flag_prior ; 


    CPhoton(const CCtx& ctx, CRecState& state);

    void clear();

    void add(unsigned flag, unsigned  material);
    void increment_slot() ; 

    bool is_rewrite_slot() const  ;
    bool is_flag_done() const ;
    bool is_done() const ;
    bool is_hard_truncate() ;
    void scrub_mskhis( unsigned flag );

    std::string desc() const ; 
    std::string brief() const ; 


};
 
