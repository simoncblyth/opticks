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

#include <vector>
#include <set>
#include <string>
#include <map>

class NSensor ; 


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API NSensorList {
    public:
        NSensorList();
        void load(const char* idmpath );
        unsigned int getNumSensors();
        void dump(const char* msg="NSensorList::dump");
        std::string description();

    public:
        NSensor* getSensor(unsigned int index);
        NSensor* findSensorForNode(unsigned int nodeIndex); // 0-based absolute node index, 0:world
    private:
        void read(const char* path);
        void add(NSensor* sensor);
        NSensor* createSensor_v1(std::vector<std::string>& elem ); // 6 columns
        NSensor* createSensor_v2(std::vector<std::string>& elem ); // 3 columns
        unsigned int parseHexString(std::string& str);

    private:
        std::vector<NSensor*>    m_sensors ; 
        std::set<unsigned int>   m_ids ; 
        std::map<unsigned int, NSensor*>   m_nid2sen ; 

};

#include "NPY_TAIL.hh"

