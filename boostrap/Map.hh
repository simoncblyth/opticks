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


#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

// TODO: merge Map with BMap that it uses ?

template <typename K, typename V> class Map ; 

template <typename K, typename V>
class BRAP_API Map {
    public:
        Map();

        static Map<K,V>* load(const char* dir, const char* name);     
        static Map<K,V>* load(const char* path);     

        void loadFromCache(const char* dir, const char* name);
        void loadFromCache(const char* path);

        void add(K key, V value);
        bool hasKey(K key) const ; 
        V get(K key, V fallback) const ; 
    
        Map<K,V>* makeSelection(const char* prefix, char delim=',');

        void save(const char* dir, const char* name);
        void save(const char* path);

        void dump(const char* msg="Map::dump") const ;
        std::map<K, V>& getMap(); 
    private:
        std::map<K, V> m_map ; 

};

#include "BRAP_TAIL.hh"


