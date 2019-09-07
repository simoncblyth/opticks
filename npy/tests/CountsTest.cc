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

#include "Counts.hpp"
#include "Index.hpp"
#include <map>

int main()
{
    typedef unsigned int T ; 


    std::map<std::string, T> m ;
    m["red"] = 1 ; 
    m["green"] = 2 ; 
    m["blue"] = 3 ; 

    Counts<T> c("hello");
    c.addMap(m);
    c.sort();
    c.dump();

    c.sort(false);
    c.dump();


    c.checkfind("green");


    c.add("purple");
    c.add("green", 10);
    c.add("red", 2);

    c.dump();
    c.sort();
    c.dump();



    Counts<T> t("test");
    t.add("red",  1); 
    t.add("green",2); 
    t.add("blue", 3); 
    t.dump();

    const char* reldir = NULL ; 
    Index* idx = t.make_index("testindex", reldir);
    idx->dump();
 




}
