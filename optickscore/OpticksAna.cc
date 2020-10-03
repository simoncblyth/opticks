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

#include <sstream>

#include "SStr.hh"
#include "SSys.hh"
#include "BFile.hh"
#include "BResource.hh"

#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksAna.hh"

#include "PLOG.hh"

const plog::Severity OpticksAna::LEVEL = debug ; 


const char* OpticksAna::DEFAULT_EXEC = "echo" ; 
const char* OpticksAna::FALLBACK_SCRIPT_DIR = "$PREFIX/py/opticks/ana" ; 

OpticksAna::OpticksAna(Opticks* ok) 
   :
   m_ok(ok),
   m_cfg(ok->getCfg())
{
   // for simple filtering 
   m_scripts["tpmt"] = "tpmt.py" ;  
   m_scripts["tevt"] = "tevt.py" ;  
   m_scripts["tboolean"] = "tboolean.py" ;  
}

bool OpticksAna::isKeyEnabled(const char* anakey) const 
{
     return anakey && m_scripts.count(anakey) == 1 ;
}

const char* OpticksAna::getScript(const char* anakey) const 
{
    return isKeyEnabled(anakey) ? strdup(m_scripts.at(anakey).c_str()) : NULL  ;  
}

bool OpticksAna::isPythonScript(const char* anakey) const 
{
    const char* script = getScript(anakey);
    return SStr::EndsWith(script, ".py"); 
}

const char* OpticksAna::getScriptResolved(const char* anakey) const 
{
    const char* script = getScript(anakey); 
    if( script == NULL ) return DEFAULT_EXEC ; 
    return BFile::ResolveScript(script, FALLBACK_SCRIPT_DIR );
}

std::string OpticksAna::getArgs(const char* /*anakey*/) const 
{
    const char* anakeyargs = m_ok->getAnaKeyArgs();
    std::stringstream ss ; 
    ss
         << "--tagoffset " << m_ok->getTagOffset() << " "
         << "--tag " << m_ok->getEventTag() << " "
         << "--cat " << m_ok->getInputUDet() << " "     
         << "--pfx " << m_ok->getEventPfx() << " "
         << "--src " << m_ok->getSourceType() << " "
         << "--show " 
         << ( anakeyargs ? anakeyargs : "" )
         ;

    return ss.str();
}

std::string OpticksAna::getCommandLine(const char* anakey) const 
{
    bool py = isPythonScript(anakey); 

    std::stringstream ss ; 
    if(py) ss << SSys::ResolvePython() << " " ; 
    ss
       << getScriptResolved(anakey) << " "
       << getArgs(anakey) << " "
       ;
    return ss.str();
}


void OpticksAna::run()
{
   const char* anakey = m_ok->getAnaKey();
   bool enabled = isKeyEnabled(anakey) ; 
   LOG(info)
       << " anakey " << anakey  
       << " enabled " << ( enabled ? "Y" : "N" )
       ; 

   if(!enabled) return ; 

   std::string cmdline = getCommandLine(anakey);

   LOG(info) << " cmdline " << cmdline ;  

   std::cout << std::endl ;  

   int rc = cmdline.empty() ? 0 : SSys::run(cmdline.c_str()) ; 

   std::cout << std::endl ;  
   
   const char* rcmsg = rc == 0 ? NULL : "OpticksAna::run non-zero RC from ana script"  ;

   int interactivity = m_ok->getInteractivityLevel() ; 

   LOG(info) 
       << " anakey " << anakey 
       << " cmdline " << cmdline
       << " interactivity " << interactivity
       << " rc " << rc
       << " rcmsg " << ( rcmsg ? rcmsg : "-" )
       ;

   if( rc != 0)  m_ok->setRC(rc, rcmsg);


   if( interactivity > 1 )
   SSys::WaitForInput("OpticksAna::run paused : hit RETURN to continue..." );    

}

