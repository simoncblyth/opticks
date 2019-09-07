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
#include <cstdlib>
#include <cassert>

#include "BFile.hh"

#include "BTxt.hh"
#include "BPropNames.hh"

#include "PLOG.hh"

BPropNames::BPropNames(const char* libname)
   :
   m_libname(strdup(libname)),
   m_txt(NULL) 
{
   read(); 
}


std::string BPropNames::libpath(const char* libname)
{
    std::string path = BFile::FormPath("$IDPATH", "GItemList", libname ); 
    return path ; 
}


void BPropNames::read()
{
    LOG(verbose) << "BPropNames::read" 
               << " libname " << ( m_libname ? m_libname : "NULL" )
               ;

    if(!m_txt)
    {

        if(strlen(m_libname) > 0 && m_libname[0] == '/' )
        {
            // absolute path for testing  
            m_txt = new BTxt(m_libname);
        } 
        else
        {
            // GItemList name like GMaterialLib 
            std::string path = libpath(m_libname) ; 
            LOG(verbose) << "BPropNames::read"
                       << " path " << path ; 

            m_txt = new BTxt(path.c_str());
        }
    }
    m_txt->read();
}


const char* BPropNames::getLine(unsigned int num)
{
   return m_txt ? m_txt->getLine(num) : NULL ; 
}
unsigned int  BPropNames::getNumLines()
{
   return m_txt ? m_txt->getNumLines() : 0 ;
}

unsigned int  BPropNames::getIndex(const char* line)
{
   assert(m_txt); 
   return m_txt->getIndex(line) ;
}



