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
#include <cstring>
#include <iomanip>

#include "NSensor.hpp"

#ifdef _MSC_VER
#define strdup _strdup
#endif


// kludge until fix the "csv" idmap to only put sensor labels on cathodes
const char* NSensor::CATHODE_NODE_NAME = "/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode" ;

const unsigned int NSensor::UNSET_INDEX = 0  ;

NSensor::NSensor(unsigned int index, unsigned int id, const char* node_name, unsigned int node_index)
       :
       m_index(index),
       m_id(id),
       m_node_name(node_name?strdup(node_name):NULL),
       m_node_index(node_index)
{
}

unsigned int NSensor::getIndex()
{
    return m_index ; 
}
unsigned int NSensor::getIndex1()
{
    return m_index + 1 ; 
}

unsigned int NSensor::RefIndex(NSensor* sensor)
{
    return sensor ? sensor->getIndex1() : NSensor::UNSET_INDEX  ;
}


unsigned int NSensor::getNodeIndex()
{
    return m_node_index ; 
}
unsigned int NSensor::getId()
{
    return m_id ; 
}
const char* NSensor::getNodeName()
{
    return m_node_name ; 
}


bool NSensor::isCathode()
{
    return m_node_name && strcmp(m_node_name, CATHODE_NODE_NAME) == 0 ; 
}

std::string NSensor::description()
{
    std::stringstream ss ; 

    ss << "NSensor "
       << " index " << std::setw(6) << m_index 
       << " idhex " << std::setw(6) << std::hex << m_id  
       << " iddec " << std::setw(6) << std::dec << m_id  
       << " node_index " << std::setw(6) << m_node_index 
       << " name " << m_node_name  
       << " " << ( isCathode() ? "CATHODE" : "NOT-CATHODE" ) 
       ;

    return ss.str();
}
