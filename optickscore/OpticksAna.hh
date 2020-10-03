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
#include <map>
class Opticks ; 
template <typename T> class OpticksCfg ;

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"
#include "plog/Severity.h"

/**
OpticksAna
============

Canonical m_ana OpticksAna instance is ctor resident of m_ok Opticks, 
which in turn is ctor resident of top level managers 
such as OKG4Mgr and OKMgr.

::

    OpticksAnaTest --anakey tpmt --tag 10 --cat PmtInBox

**/

class OKCORE_API OpticksAna
{
       friend struct OpticksAnaTest ;   
       static const plog::Severity LEVEL ; 
       static const char* DEFAULT_EXEC ; 
       static const char* FALLBACK_SCRIPT_DIR ; 
    public:
       OpticksAna(Opticks* ok);
       void run();
   private:
       std::string getCommandLine(const char* anakey) const ;
       bool isKeyEnabled(const char* anakey) const ;
       const char* getScript(const char* anakey) const ;
       const char* getScriptResolved(const char* anakey) const ;
       bool isPythonScript(const char* anakey) const ;
       std::string getArgs(const char* anakey) const ;
       void setEnv();
   private:
       Opticks*             m_ok ; 
       OpticksCfg<Opticks>* m_cfg ; 
       std::map<std::string, std::string> m_scripts ; 

};

#include "OKCORE_HEAD.hh"


