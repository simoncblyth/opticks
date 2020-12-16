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


#include <cstring>
#include "SArgs.hh"

#include "BOpticks.hh"
#include "BOpticksKey.hh"
#include "BOpticksResource.hh"

#include "PLOG.hh"

BOpticks::BOpticks(int argc, char** argv, const char* argforced )
    :    
    m_firstarg( argc > 1 ? argv[1] : nullptr ), 
    m_sargs(new SArgs(argc, argv, argforced)), 
    m_argc(m_sargs->argc),
    m_argv(m_sargs->argv),
    m_envkey(m_sargs->hasArg("--envkey") ? BOpticksKey::SetKey(nullptr) : false),
    m_testgeo(false),
    m_resource(BOpticksResource::Get(NULL)),    // creates if no instance yet
    m_error(0)
{
}


const char* BOpticks::getPath(const char* rela, const char* relb, const char* relc ) const
{
    return m_resource->makeIdPathPath(rela, relb, relc );
}

int BOpticks::getError() const { 
    if( m_error > 0 ) LOG(fatal) << " MISSING OPTICKS_KEY " ; 
    return m_error ; 
}

const char* BOpticks::getFirstArg(const char* fallback) const 
{
    return m_firstarg ? m_firstarg : fallback ; 
}

const char* BOpticks::getArg(int n, const char* fallback) const   // argforce makes this problematic
{
    return n < m_argc ? m_argv[n] : fallback ; 
}
