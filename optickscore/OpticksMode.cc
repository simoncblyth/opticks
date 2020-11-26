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

#include <cassert>
#include <cstring>
#include <sstream>

#include "SSys.hh"
#include "Opticks.hh"
#include "OpticksMode.hh"
#include "PLOG.hh"


const plog::Severity OpticksMode::LEVEL = PLOG::EnvLevel("OpticksMode", "DEBUG" ); 

const char* OpticksMode::COMPUTE_ARG_ = "--compute" ; 
const char* OpticksMode::INTEROP_ARG_ = "--interop" ; 
const char* OpticksMode::NOVIZ_ARG_ = "--noviz" ; 

const char* OpticksMode::UNSET_MODE_  = "UNSET_MODE" ;
const char* OpticksMode::COMPUTE_MODE_  = "COMPUTE_MODE" ;
const char* OpticksMode::INTEROP_MODE_  = "INTEROP_MODE" ;

bool OpticksMode::isCompute() const { return (m_mode & COMPUTE_MODE) != 0 ; }
bool OpticksMode::isInterop() const { return (m_mode & INTEROP_MODE) != 0 ; } 

std::string OpticksMode::desc() const 
{
    std::stringstream ss ; 

    if(isCompute()) ss << COMPUTE_MODE_ ; 
    if(isInterop()) ss << INTEROP_MODE_ ; 

    if(m_compute_requested) ss << " compute_requested " ; 
    if(m_forced_compute)    ss << " forced_compute " ; 

    return ss.str();
}

unsigned int OpticksMode::Parse(const char* tag)  // static
{
    unsigned int mode = UNSET_MODE  ; 
    if(     strcmp(tag, INTEROP_MODE_)==0)  mode = INTEROP_MODE ; 
    else if(strcmp(tag, COMPUTE_MODE_)==0)  mode = COMPUTE_MODE ; 
    return mode ; 
}


int OpticksMode::getInteractivityLevel() const 
{
    int interactivity = SSys::GetInteractivityLevel() ;
    if(m_noviz) interactivity = 0 ; 
    if(isCompute()) interactivity = 0 ; 
    return interactivity  ;
}


/**
OpticksMode::OpticksMode(const char* tag)
--------------------------------------------

Used by OpticksEvent to instanciate from loaded metadata string.

**/

OpticksMode::OpticksMode(const char* tag)
    :
    m_mode(Parse(tag)),
    m_compute_requested(false),
    m_noviz(false),
    m_forced_compute(false)
{
    LOG(LEVEL) << " tag " << tag ;  
}

OpticksMode::OpticksMode(Opticks* ok) 
    : 
    m_mode(UNSET_MODE),
    m_compute_requested(ok->hasArg(COMPUTE_ARG_) && !ok->hasArg(INTEROP_ARG_)),   // "--interop" trumps "--compute" on same commandline
    m_noviz(ok->hasArg(NOVIZ_ARG_)),
    m_forced_compute(false)
{
    if(SSys::IsRemoteSession())
    {
        m_mode = COMPUTE_MODE ; 
        m_forced_compute = true ;  
    }
    else
    {
        m_mode = m_compute_requested ? COMPUTE_MODE : INTEROP_MODE ;
    }
    LOG(LEVEL) << desc() ;  
}



