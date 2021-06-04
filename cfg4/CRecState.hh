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

struct CCtx ; 
#include <string>
#include "CFG4_API_EXPORT.hh"

/**
CRecState
=============

m_state member of CRecorder



**/

struct CFG4_API CRecState
{
    const CCtx& _ctx ; 

    unsigned _decrement_request ; 
    unsigned _decrement_denied ; 
    unsigned _topslot_rewrite ; 

    bool     _record_truncate ; 
    bool     _bounce_truncate ; 

    unsigned _slot ; 
    unsigned _step_action ; 

    CRecState(const CCtx& ctx);
    void clear();
    std::string desc() const ; 

    void     decrementSlot();  // NB not just live 
    unsigned constrained_slot();
    void     increment_slot_regardless();
    bool     is_truncate() const ; 



};
 
