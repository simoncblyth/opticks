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

#include "SLog.hh"
#include "PLOG.hh"

SLog::SLog(const char* label, const char* extra, plog::Severity level) 
   :
   m_label(strdup(label)),
   m_extra(strdup(extra)),
   m_level(level)
{
    pLOG(m_level,0) 
        << " ( "
        << m_label 
        << " " 
        << m_extra 
        ;  
}


const char* SLog::exename() // static
{
    return PLOG::instance->args.exename() ; 
}

void SLog::operator()(const char* msg)
{
    pLOG(m_level,0) 
        << " ) "
        << m_label 
        << " " 
        << m_extra 
        << " "
        << msg 
        ;  
}

void SLog::Nonce()
{
    LOG(verbose) << "verbose" ; 
    LOG(debug) << "debug" ; 
    LOG(info) << "info" ; 
    LOG(warning) << "warning" ; 
    LOG(error) << "error" ; 
    LOG(fatal) << "fatal" ; 
}



