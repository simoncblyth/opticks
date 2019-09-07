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

#include "PLOG.hh"
#include "Typ.hpp"

Typ::Typ()
{
}

void Typ::setMaterialNames(std::map<unsigned int, std::string> material_names)
{
    m_material_names = material_names ; 
}
void Typ::setFlagNames(std::map<unsigned int, std::string> flag_names)
{
    m_flag_names = flag_names ; 
}

std::string Typ::findMaterialName(unsigned int index)
{
    return m_material_names.count(index) == 1 ? m_material_names[index] : "?" ; 
}



void Typ::dumpMap(const char* msg, const std::map<unsigned, std::string>& m ) const 
{
    typedef std::map<unsigned, std::string> MUS ; 

    LOG(info) << msg ; 
    for(MUS::const_iterator it=m.begin() ; it != m.end() ; it++)
    {
       std::cout 
          << std::setw(5) << it->first
          << " : " 
          << std::setw(30) << it->second
          << std::endl 
          ;
    } 

}


void Typ::dump(const char* msg) const 
{
    LOG(info) << msg 
              << " num_material_names " << m_material_names.size()
              << " num_flag_names " << m_flag_names.size()
              ;

    dumpMap( "material_names", m_material_names );
    dumpMap( "flag_names", m_flag_names );
}




